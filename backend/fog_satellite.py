"""
Satellite fog along a route (Meteosat-9 night fog maps).

The heavy work (download, crop, fog detection, distance and edge-speed grids)
is done every ~30 min by satellite_worker/ (GitHub Actions), which uploads a
small scene file + latest.json to the public Supabase Storage bucket
`fog-satellite`. This module only:

  1. keeps the newest scene in memory (~0.5 MB; re-checks the manifest every
     2 min, downloads a scene only when it changes) - fits the 512 MB budget;
  2. for each waypoint, reads a 3x3 pixel window (~15 km) of the fog map and
     the distance / edge-speed grids at that pixel -> a status per waypoint.

Statuses: fog | patchy | approaching | clear | unknown | daytime | stale |
          no_data | outside
Best-effort: any failure returns `no_data`, never an exception to the caller.
"""
import io
import json
import math
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
import requests

BUCKET = "fog-satellite"
MANIFEST_TTL_S = 120          # how often we re-check latest.json
STALE_AFTER_MIN = 75          # newest scene older than this -> "stale"
WINDOW = 1                    # 1 -> 3x3 pixels around the waypoint
FOG_MIN_PIXELS = 3            # >= this many fog pixels in the window -> "fog"
UNKNOWN_MAX_PIXELS = 4        # more blind pixels than this -> "unknown"
APPROACH_MAX_KM = 50.0        # only extrapolate fog edges closer than this
APPROACH_SLACK_MIN = 15       # fog arriving up to 15 min after the driver still counts
OUTLOOK_AFTER_MIN = 180       # satellite extrapolation not trusted beyond 3 h

_lock = threading.Lock()
_state = {"checked": 0.0, "manifest": None, "scene_path": None, "scene": None, "error": None}


def _base_url():
    """FOG_SAT_BASE_URL overrides (tests / local folder via file://)."""
    override = os.environ.get("FOG_SAT_BASE_URL")
    if override:
        return override.rstrip("/")
    supa = os.environ.get("SUPABASE_URL", "").rstrip("/")
    return f"{supa}/storage/v1/object/public/{BUCKET}" if supa else ""


def _get(path):
    base = _base_url()
    if not base:
        raise RuntimeError("SUPABASE_URL not configured")
    if base.startswith("file://"):
        with open(os.path.join(base[7:], path), "rb") as f:
            return f.read()
    r = requests.get(f"{base}/{path}", params={"v": int(time.time() // 60)}, timeout=15)
    r.raise_for_status()
    return r.content


def _refresh(force=False):
    """Load the newest manifest/scene if our copy is older than MANIFEST_TTL_S."""
    with _lock:
        if not force and time.time() - _state["checked"] < MANIFEST_TTL_S:
            return
        _state["checked"] = time.time()
        try:
            manifest = json.loads(_get("latest.json"))
            _state["manifest"], _state["error"] = manifest, None
            scenes = manifest.get("scenes") or []
            path = scenes[0]["path"] if scenes else None
            if path and path != _state["scene_path"]:
                with np.load(io.BytesIO(_get(path))) as z:
                    _state["scene"] = {
                        "fog": z["fog"].astype(np.uint8),
                        "dist_km": z["dist_km"].astype(np.float32),
                        "speed_kmh": z["speed_kmh"].astype(np.float32),
                    }
                _state["scene_path"] = path
            elif not path:
                _state["scene"], _state["scene_path"] = None, None
        except Exception as e:  # best-effort: keep the last good scene
            _state["error"] = f"{type(e).__name__}: {e}"
            print(f"[fog_satellite] refresh failed: {_state['error']}")


def _parse_time(s):
    return datetime.strptime(s[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)


def status():
    """Scene metadata for /fog/satellite/status and the response header."""
    _refresh()
    m = _state["manifest"] or {}
    scenes = m.get("scenes") or []
    newest = scenes[0] if scenes else None
    age = None
    if newest:
        age = round((datetime.now(timezone.utc) - _parse_time(newest["time"])).total_seconds() / 60, 1)
    if _state["error"] and not m:
        state = "no_data"
    elif m.get("status") == "day":
        state = "daytime"
    elif newest is None:
        state = "no_data"
    elif age is not None and age > STALE_AFTER_MIN:
        state = "stale"
    else:
        state = "ok"
    return {
        "state": state,
        "scene_time": newest["time"] if newest else None,
        "scene_age_min": age,
        "fog_share": newest.get("fog_share") if newest else None,
        "unknown_share": newest.get("unknown_share") if newest else None,
        "satellite": m.get("satellite"),
        "error": _state["error"],
    }


def _pixel(grid, lat, lon):
    row = int(round((grid["lat_max"] - lat) / grid["res"]))
    col = int(round((lon - grid["lon_min"]) / grid["res"]))
    return row, col


def _f(x):
    x = float(x)
    return None if not math.isfinite(x) else round(x, 1)


def classify_point(scene, grid, classes, lat, lon, eta_mins):
    """Status for one waypoint from an in-memory scene (pure, unit-tested)."""
    fog_c, unk_c = classes.get("fog", 1), classes.get("unknown", 2)
    fog = scene["fog"]
    row, col = _pixel(grid, lat, lon)
    if not (0 <= row < fog.shape[0] and 0 <= col < fog.shape[1]):
        return {"status": "outside"}
    win = fog[max(0, row - WINDOW): row + WINDOW + 1, max(0, col - WINDOW): col + WINDOW + 1]
    n_fog = int((win == fog_c).sum())
    n_unk = int((win == unk_c).sum())
    dist = _f(scene["dist_km"][row, col])
    speed = _f(scene["speed_kmh"][row, col])
    out = {"fog_pixels": n_fog, "window_pixels": int(win.size), "dist_km": dist,
           "edge_speed_kmh": speed, "fog_eta_mins": None,
           "outlook": eta_mins > OUTLOOK_AFTER_MIN}
    if n_unk > UNKNOWN_MAX_PIXELS:
        return {**out, "status": "unknown"}
    if n_fog >= FOG_MIN_PIXELS:
        return {**out, "status": "fog"}
    if n_fog > 0:
        return {**out, "status": "patchy"}
    if dist is not None and speed and speed > 0 and dist <= APPROACH_MAX_KM:
        fog_eta = dist / speed * 60.0
        out["fog_eta_mins"] = round(fog_eta)
        if eta_mins <= OUTLOOK_AFTER_MIN and fog_eta <= eta_mins + APPROACH_SLACK_MIN:
            return {**out, "status": "approaching"}
    return {**out, "status": "clear"}


def route_fog(waypoints):
    """waypoints: [{lat, lon, eta_mins}] -> {scene: status(), points: [...]}"""
    meta = status()
    pts = []
    scene = _state["scene"]
    m = _state["manifest"] or {}
    for w in waypoints:
        lat, lon, eta = float(w["lat"]), float(w["lon"]), float(w.get("eta_mins") or 0)
        base = {"lat": lat, "lon": lon, "eta_mins": eta}
        if meta["state"] != "ok" or scene is None:
            pts.append({**base, "status": meta["state"]})
            continue
        try:
            pts.append({**base, **classify_point(scene, m["grid"], m.get("classes") or {}, lat, lon, eta)})
        except Exception as e:
            print(f"[fog_satellite] point failed: {e}")
            pts.append({**base, "status": "no_data"})
    return {"scene": meta, "points": pts}
