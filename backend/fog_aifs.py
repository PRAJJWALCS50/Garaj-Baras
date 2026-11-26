"""
ECMWF AIFS fog forecast along a route.

aifs_worker/ (GitHub Actions, hourly) downloads each new AIFS Single run
(4 per day, 6-hourly steps, 0.25 deg), keeps 2 m dew-point depression, 10 m
wind and low cloud over North India for steps 0..72 h, and uploads
aifs/run_YYYYMMDDHH.npz (~50 KB) + aifs/latest.json to the public Supabase
bucket `fog-satellite`. This module only:

  1. keeps the newest run in memory (re-checks the manifest every 10 min);
  2. for each waypoint, takes the time the driver will be there
     (now + eta), blends the two 6-hourly steps around that time (linear in
     time) and the 4 grid cells around the point (bilinear), then applies the
     fog rule written in the manifest (same thresholds as the worker).

Statuses: fog | possible | clear | outside (not in the box) |
          beyond (after the last forecast step) | no_data
Run state: ok | stale (run older than 24 h - worker failing) | no_data.
Best-effort: never raises to the caller.
"""
import io
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

BUCKET = "fog-satellite"
MANIFEST = "aifs/latest.json"
MANIFEST_TTL_S = 600
STALE_AFTER_H = 24

_lock = threading.Lock()
_state = {"checked": 0.0, "manifest": None, "path": None, "run": None, "error": None}


def _now():
    return datetime.now(timezone.utc)


def _base_url():
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
    with _lock:
        ttl = MANIFEST_TTL_S if _state["run"] is not None else 60   # retry sooner while empty
        if not force and time.time() - _state["checked"] < ttl:
            return
        _state["checked"] = time.time()
        try:
            man = json.loads(_get(MANIFEST))
            if man.get("path") != _state["path"]:
                with np.load(io.BytesIO(_get(man["path"]))) as z:
                    _state["run"] = {k: z[k] for k in ("lats", "lons", "steps_h", "dpd", "wind", "lcc")}
                _state["path"] = man["path"]
            _state["manifest"], _state["error"] = man, None
        except Exception as e:      # keep the previous run if we had one
            _state["error"] = f"{type(e).__name__}: {e}"


def _run_time(man):
    return datetime.strptime(man["run"][:13], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)


def status():
    _refresh()
    man, run = _state["manifest"], _state["run"]
    if not man or run is None:
        return {"state": "no_data", "error": _state["error"]}
    rt = _run_time(man)
    age_h = (_now() - rt).total_seconds() / 3600
    return {"state": "stale" if age_h > STALE_AFTER_H else "ok", "run": man["run"],
            "age_h": round(age_h, 1), "model": man.get("model"),
            "last_valid": (rt + timedelta(hours=int(run["steps_h"][-1]))).strftime("%Y-%m-%dT%H:%MZ"),
            "error": _state["error"]}


def classify(dpd, wind, lcc, rule):
    """Same rule as aifs_worker/fogrule.py, thresholds from the manifest."""
    if (dpd <= rule["fog_dpd_k"] and wind <= rule["fog_wind_ms"]) or (
            lcc >= rule["lowcloud_lcc"] and dpd <= rule["lowcloud_dpd_k"] and wind <= rule["lowcloud_wind_ms"]):
        return "fog"
    if dpd <= rule["possible_dpd_k"] and wind <= rule["possible_wind_ms"]:
        return "possible"
    return "clear"


def _bilinear(field, lats, lons, lat, lon):
    """field (ny, nx) on lats N->S, lons W->E."""
    fy = (lats[0] - lat) / (lats[0] - lats[1])
    fx = (lon - lons[0]) / (lons[1] - lons[0])
    y0 = int(np.clip(np.floor(fy), 0, len(lats) - 2)); x0 = int(np.clip(np.floor(fx), 0, len(lons) - 2))
    wy, wx = float(np.clip(fy - y0, 0, 1)), float(np.clip(fx - x0, 0, 1))
    f = field[y0:y0 + 2, x0:x0 + 2].astype(np.float64)
    return float((f[0, 0] * (1 - wx) + f[0, 1] * wx) * (1 - wy) + (f[1, 0] * (1 - wx) + f[1, 1] * wx) * wy)


def classify_point(run, man, lat, lon, eta_mins, now):
    lats, lons, steps = run["lats"], run["lons"], run["steps_h"]
    if not (lats[-1] <= lat <= lats[0] and lons[0] <= lon <= lons[-1]):
        return {"status": "outside"}
    valid = now + timedelta(minutes=float(eta_mins or 0))
    lead_h = (valid - _run_time(man)).total_seconds() / 3600
    if lead_h > steps[-1]:
        return {"status": "beyond", "valid": valid.strftime("%Y-%m-%dT%H:%MZ")}
    lead_h = max(lead_h, float(steps[0]))
    i = int(np.clip(np.searchsorted(steps, lead_h, side="right") - 1, 0, len(steps) - 2))
    w = float(np.clip((lead_h - steps[i]) / (steps[i + 1] - steps[i]), 0, 1))
    val = {}
    for k in ("dpd", "wind", "lcc"):
        a = _bilinear(run[k][i], lats, lons, lat, lon)
        b = _bilinear(run[k][i + 1], lats, lons, lat, lon)
        val[k] = a + (b - a) * w
    return {"status": classify(val["dpd"], val["wind"], val["lcc"], man["rule"]),
            "valid": valid.strftime("%Y-%m-%dT%H:%MZ"),
            "dpd_k": round(val["dpd"], 2), "wind_ms": round(val["wind"], 1),
            "lcc": round(val["lcc"], 2), "lead_h": round(lead_h, 1)}


def route_fog(waypoints):
    try:
        st = status()
        if st["state"] == "no_data":
            return {"run": st, "points": [{"lat": w.get("lat"), "lon": w.get("lon"), "status": "no_data"}
                                          for w in waypoints]}
        man, run, now = _state["manifest"], _state["run"], _now()
        points = []
        for w in waypoints:
            lat, lon = float(w["lat"]), float(w["lon"])
            points.append({"lat": lat, "lon": lon, "eta_mins": w.get("eta_mins"),
                           **classify_point(run, man, lat, lon, w.get("eta_mins"), now)})
        return {"run": st, "points": points}
    except Exception as e:
        return {"run": {"state": "no_data", "error": f"{type(e).__name__}: {e}"},
                "points": [{"lat": w.get("lat"), "lon": w.get("lon"), "status": "no_data"} for w in waypoints]}
