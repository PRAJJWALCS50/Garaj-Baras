"""
Satellite fog along a route (Meteosat-9 night fog maps).

The heavy work (download, crop, fog detection, distance and edge-speed grids)
is done every ~30 min by satellite_worker/ (GitHub Actions), which uploads a
small scene file + latest.json to the public Supabase Storage bucket
`fog-satellite`. This module only:

  1. keeps the newest scene in memory (~0.5 MB; re-checks the manifest every
     2 min, downloads a scene only when it changes) - fits the 512 MB budget;
  2. for each waypoint, reads a 3x3 pixel window (~15 km) of the fog map, then
     looks at every separate fog PATCH within 50 km (the worker tracks patches
     between scenes: born / tracked / merged / split, with growth and drift):
     a tracked patch whose growth + drift brings it to the waypoint before the
     driver -> "approaching"; a patch that just appeared nearby -> "forming"
     (new fog has no speed yet - it formed, it did not move).

Statuses: fog | patchy | approaching | forming | clear | unknown | daytime |
          stale | no_data | outside
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
APPROACH_MAX_KM = 50.0        # only patches closer than this are considered
FORMING_WARN_KM = 15.0        # a just-born patch this close -> "forming"
MIN_APPROACH_KMH = 0.5        # slower than this is "not really coming"
APPROACH_SLACK_MIN = 15       # fog arriving up to 15 min after the driver still counts
OUTLOOK_AFTER_MIN = 180       # satellite extrapolation not trusted beyond 3 h

PATCH_COLS = ("id", "pixels", "area_km2", "lat", "lon", "status", "age",
              "growth_kmh", "drift_e_kmh", "drift_n_kmh")
PATCH_STATUS = {0: "born", 1: "tracked", 2: "merged", 3: "split", 4: "untracked"}

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
                    scene = {
                        "fog": z["fog"].astype(np.uint8),
                        "dist_km": z["dist_km"].astype(np.float32),
                        "patch_id": z["patch_id"].astype(np.int32),
                        "patches": {c: z[f"p_{c}"] for c in PATCH_COLS},
                    }
                scene["patch_row"] = {int(pid): i for i, pid in enumerate(scene["patches"]["id"])}
                _state["scene"] = scene
                _state["scene_path"] = path
            elif not path:
                _state["scene"], _state["scene_path"] = None, None
        except Exception as e:  # best-effort: keep the last good scene
            _state["error"] = f"{type(e).__name__}: {e}"
            print(f"[fog_satellite] refresh failed: {_state['error']}")


REGION_CENTRE = (27.5, 80.0)  # middle of the Indo-Gangetic Plain


def _sun_up_now(now=None):
    """True when the sun is above -3 deg at the middle of the plain (night method unusable)."""
    now = now or datetime.now(timezone.utc)
    doy = now.timetuple().tm_yday
    hour = now.hour + now.minute / 60
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g))
    lat, lon = map(math.radians, REGION_CENTRE)
    ha = math.radians((hour * 60 + eqt + 4 * REGION_CENTRE[1]) / 4 - 180)
    elev = math.degrees(math.asin(math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)))
    return elev > -3.0


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
    elif (newest is None or (age is not None and age > STALE_AFTER_MIN)) and _sun_up_now():
        state = "daytime"           # no fresh scene because it is day, not because something broke
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


def _nearby_patches(scene, grid, lat, lon, row, col):
    """[(patch_row, dist_km, unit vector patch->waypoint (east, north))] within APPROACH_MAX_KM."""
    res_km = grid["res"] * 111.32
    dr = int(math.ceil(APPROACH_MAX_KM / res_km))
    dc = int(math.ceil(APPROACH_MAX_KM / (res_km * math.cos(math.radians(lat)))))
    ids = scene["patch_id"]
    r0, r1 = max(0, row - dr), min(ids.shape[0], row + dr + 1)
    c0, c1 = max(0, col - dc), min(ids.shape[1], col + dc + 1)
    sub = ids[r0:r1, c0:c1]
    mask = sub > 0
    if not mask.any():
        return []
    rr, cc = np.nonzero(mask)
    plat = grid["lat_max"] - (rr + r0) * grid["res"]
    plon = grid["lon_min"] + (cc + c0) * grid["res"]
    de = (lon - plon) * 111.32 * math.cos(math.radians(lat))      # pixel -> waypoint, km east
    dn = (lat - plat) * 111.32                                    # km north
    d = np.hypot(de, dn)
    pid = sub[mask]
    order = np.argsort(d)
    uniq, first = np.unique(pid[order], return_index=True)        # nearest pixel of each patch
    out = []
    for p, k in zip(uniq, first):
        k = order[k]
        if d[k] <= APPROACH_MAX_KM and int(p) in scene["patch_row"]:
            n = max(float(d[k]), 1e-6)
            out.append((scene["patch_row"][int(p)], float(d[k]), (float(de[k]) / n, float(dn[k]) / n)))
    return sorted(out, key=lambda x: x[1])


def _approach_kmh(patches, i, unit):
    """Growth + drift toward the waypoint (km/h), or None if the patch has no motion yet."""
    g = float(patches["growth_kmh"][i])
    de, dn = float(patches["drift_e_kmh"][i]), float(patches["drift_n_kmh"][i])
    parts = []
    if math.isfinite(g):
        parts.append(g)
    if math.isfinite(de) and math.isfinite(dn):
        parts.append(de * unit[0] + dn * unit[1])     # drift component toward the waypoint
    return sum(parts) if parts else None


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
    out = {"fog_pixels": n_fog, "window_pixels": int(win.size),
           "dist_km": _f(scene["dist_km"][row, col]), "fog_eta_mins": None,
           "nearest_patch": None, "soonest_patch": None, "patches_nearby": 0,
           "outlook": eta_mins > OUTLOOK_AFTER_MIN}
    if n_unk > UNKNOWN_MAX_PIXELS:
        return {**out, "status": "unknown"}
    if n_fog >= FOG_MIN_PIXELS:
        return {**out, "status": "fog"}
    if n_fog > 0:
        return {**out, "status": "patchy"}

    P = scene["patches"]
    near = _nearby_patches(scene, grid, lat, lon, row, col)
    out["patches_nearby"] = len(near)
    soonest, forming = None, False
    for k, (i, dist, unit) in enumerate(near):
        status = PATCH_STATUS.get(int(P["status"][i]), "untracked")
        speed = _approach_kmh(P, i, unit)          # None for born patches (no history)
        eta = dist / speed * 60.0 if speed is not None and speed > MIN_APPROACH_KMH else None
        info = {"id": int(P["id"][i]), "status": status, "dist_km": round(dist, 1),
                "area_km2": round(float(P["area_km2"][i])),
                "approach_kmh": None if speed is None else round(speed, 1),
                "eta_mins": None if eta is None else round(eta)}
        if k == 0:
            out["nearest_patch"] = info
        if status == "born" and dist <= FORMING_WARN_KM:
            forming = True
        if eta is not None and (soonest is None or eta < soonest["eta_mins"]):
            soonest = info
    out["soonest_patch"] = soonest
    if soonest is not None:
        out["fog_eta_mins"] = soonest["eta_mins"]
        if eta_mins <= OUTLOOK_AFTER_MIN and soonest["eta_mins"] <= eta_mins + APPROACH_SLACK_MIN:
            return {**out, "status": "approaching"}
    if forming:
        return {**out, "status": "forming"}
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
