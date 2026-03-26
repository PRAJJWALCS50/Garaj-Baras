from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import FileResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from pydantic import BaseModel  # type: ignore
from typing import List
import os
import time
import threading
import traceback

from radar import (
    get_recent_frames,
    get_all_frames,
    get_radar_lag_mins,
    refresh_frames_if_stale,
    GIF_SAVE_PATH,
    RADAR_TTL_SEC,
)  # type: ignore
from optical_flow import get_movement_vector, build_clutter_mask  # type: ignore
from prediction import (generate_waypoints, check_route_rain,   # type: ignore
                        haversine_km)
from georef import latlon_to_pixel, is_within_radar  # type: ignore
from fuzzy import enrich_results  # type: ignore

app = FastAPI(
    title="Garaj Baras API",
    description="Rain prediction for Delhi NCR routes",
    version="1.0.0"
)

# Serve extracted radar PNGs (and allow clients to fetch them)
try:
    from radar import FRAMES_FOLDER  # type: ignore

    if os.path.isdir(FRAMES_FOLDER):
        app.mount("/radar/frames", StaticFiles(directory=FRAMES_FOLDER), name="radar_frames")
except Exception:
    # Best-effort: API still works without static mounting (e.g. missing folder on first boot)
    pass

# Allow all origins for now (frontend will call this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    spacing_km: float = 2.0
    # OSRM routes: waypoint every N minutes of driving (ignored for straight-line fallback)
    eta_spacing_minutes: float = 2.0

class WaypointResult(BaseModel):
    lat: float
    lon: float
    eta_mins: float
    effective_eta: float
    rain_expected: bool
    confidence: str
    label: str
    color: str
    dbz: float
    message: str
    in_radar_bounds: bool

class PredictResponse(BaseModel):
    route_distance_km: float
    total_waypoints: int
    rain_waypoints: int
    clear_waypoints: int
    first_rain_eta: float | None
    first_rain_label: str | None
    rain_direction_from: str
    rain_direction_to: str
    rain_speed_kmh: float
    radar_lag_mins: float
    radar_freshness: str
    radar_message: str
    waypoints: List[dict]


class WaypointInput(BaseModel):
    lat: float
    lon: float
    eta_mins: float


class PredictWaypointsRequest(BaseModel):
    waypoints: List[WaypointInput]

# Global state - loaded once at startup
# Lazy-cache refreshed by TTL
radar_cache = {
    "clutter_mask": None,
    "last_loaded": None
}

_radar_state_lock = threading.Lock()
RADAR_CACHE_TTL_SEC = RADAR_TTL_SEC  # keep a single source of truth


def _load_radar_state(ttl_sec: float = RADAR_CACHE_TTL_SEC, *, force: bool = False) -> dict:
    """
    Lazy cache manager used by /predict.

    Fresh path (GIF < ttl_sec): does NOT download or re-extract frames; reuses in-memory cache.
    Stale path: refreshes GIF, clears old PNGs, extracts frames + timestamps, then recomputes
    clutter mask + movement vector once.
    """
    now = time.time()

    # Fast path: valid in-memory state AND GIF still fresh
    try:
        gif_fresh = os.path.exists(GIF_SAVE_PATH) and (now - os.path.getmtime(GIF_SAVE_PATH) < ttl_sec)
    except Exception:
        gif_fresh = False

    if (not force) and gif_fresh and radar_cache.get("frame_data") and radar_cache.get("movement") and radar_cache.get("clutter_mask") is not None:
        return radar_cache

    with _radar_state_lock:
        # Re-check inside lock
        try:
            gif_fresh = os.path.exists(GIF_SAVE_PATH) and (now - os.path.getmtime(GIF_SAVE_PATH) < ttl_sec)
        except Exception:
            gif_fresh = False

        if (not force) and gif_fresh and radar_cache.get("frame_data") and radar_cache.get("movement") and radar_cache.get("clutter_mask") is not None:
            return radar_cache

        # If stale/missing, refresh (download + clear PNGs + extract frames)
        frame_data, did_refresh = refresh_frames_if_stale(ttl_sec=ttl_sec, force=force, clear_pngs=True)
        if did_refresh:
            all_frame_data = frame_data
        else:
            # If GIF is fresh but we don't have in-memory cache (e.g. server restart),
            # do a *no-download* extract from the existing GIF once.
            # (This is still processing, but it honors the “don’t download” requirement.)
            from radar import extract_frames, FRAMES_FOLDER, GIF_SAVE_PATH  # type: ignore
            all_frame_data = extract_frames(GIF_SAVE_PATH, FRAMES_FOLDER) if gif_fresh else get_all_frames()

        recent_frame_data = all_frame_data[-6:] if len(all_frame_data) > 6 else all_frame_data
        all_paths = [p for (p, _ts) in all_frame_data]
        clutter_mask = build_clutter_mask(all_paths)
        dx, dy, dir_from, dir_to, speed = get_movement_vector(
            recent_frame_data, clutter_mask=clutter_mask
        )
        latest_frame = recent_frame_data[-1][0] if recent_frame_data else None
        latest_ts = recent_frame_data[-1][1] if recent_frame_data else None
        lag_info = get_radar_lag_mins(latest_ts)

        radar_cache.update({
            "frame_data": all_frame_data,
            "recent_frame_data": recent_frame_data,
            "clutter_mask": clutter_mask,
            "movement": (dx, dy, dir_from, dir_to, speed),
            "latest_frame": latest_frame,
            "latest_ts": latest_ts,
            "lag_info": lag_info,
            "last_loaded": time.time(),
            "gif_mtime": os.path.getmtime(GIF_SAVE_PATH) if os.path.exists(GIF_SAVE_PATH) else None,
        })
        return radar_cache


# ENDPOINT 1: Health Check

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Garaj Baras API",
        "version": "1.0.0"
    }

# ENDPOINT 2: Current Rain Movement

@app.get("/movement")
def get_movement():
    """
    Returns current rain movement direction 
    and speed over Delhi NCR
    """
    try:
        all_frame_data = get_all_frames()
        recent_frame_data = get_recent_frames(n=6)
        all_paths = [p for (p, _ts) in all_frame_data]
        clutter_mask = build_clutter_mask(all_paths)
        
        dx, dy, dir_from, dir_to, speed = get_movement_vector(
            recent_frame_data, clutter_mask=clutter_mask
        )
        latest_ts = recent_frame_data[-1][1] if recent_frame_data else None
        lag_info = get_radar_lag_mins(latest_ts)
        
        return {
            "direction_from": dir_from,
            "direction_to": dir_to,
            "speed_kmh": round(float(speed), 1),  # type: ignore
            "dx": round(float(dx), 2),  # type: ignore
            "dy": round(float(dy), 2),  # type: ignore
            "radar_lag_mins": lag_info["lag_mins"],
            "radar_freshness": lag_info["freshness"],
            "message": lag_info["message"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Movement detection failed: {str(e)}"
        )

# ENDPOINT 3: Latest Radar Frames (PNG + GIF)

@app.get("/radar/gif")
def get_radar_gif():
    """
    Returns the latest downloaded Delhi radar GIF (if present).
    Use with /frames/latest to ensure the GIF/frames are refreshed when stale.
    """
    if not os.path.exists(GIF_SAVE_PATH):
        raise HTTPException(status_code=404, detail="Radar GIF not found on server yet.")
    return FileResponse(GIF_SAVE_PATH, media_type="image/gif", filename="delhi_radar.gif")


@app.get("/frames/latest")
def get_latest_frames(n: int = 6, force: bool = True):
    """
    Fetch latest radar frames.

    - If frames are stale (older than RADAR_TTL_SEC) it refreshes the GIF and re-extracts frames.
    - If `force=true`, it refreshes even if the GIF is still fresh.

    Returns frame URLs served from `/radar/frames/<filename>`.
    """
    if n <= 0:
        n = 6
    n = min(int(n), 60)

    state = _load_radar_state(ttl_sec=RADAR_CACHE_TTL_SEC, force=bool(force))
    frame_data = state.get("frame_data") or []
    latest_ts = state.get("latest_ts")
    lag_info = state.get("lag_info") or {"lag_mins": 25.0, "freshness": "stale", "message": "Radar ~25 mins old (estimate)"}

    last_n = frame_data[-n:] if len(frame_data) > n else frame_data
    frames = []
    for fp, ts in last_n:
        frames.append({
            "filename": os.path.basename(fp),
            "url": f"/radar/frames/{os.path.basename(fp)}",
            "timestamp_ist": ts.isoformat() if ts else None,
        })

    return {
        "count": len(frames),
        "requested": int(n),
        "force": bool(force),
        "gif_url": "/radar/gif",
        "latest_timestamp_ist": latest_ts.isoformat() if latest_ts else None,
        "radar_lag_mins": lag_info.get("lag_mins"),
        "radar_freshness": lag_info.get("freshness"),
        "radar_message": lag_info.get("message"),
        "frames": frames,
    }

# ENDPOINT 4: Predict Rain For Provided Waypoints

@app.post("/predict_waypoints")
def predict_waypoints(payload: PredictWaypointsRequest):
    """
    Predict rain for a frontend-provided set of waypoints with explicit ETAs.

    Use-case:
      Frontend computes a road route geometry + ETAs from user-provided avg speed,
      then asks the backend to score rain intensity at those points/times.
    """
    try:
        if not payload.waypoints:
            raise HTTPException(status_code=400, detail="No waypoints provided.")

        # Validate coordinates are in India roughly (and ETAs are non-negative)
        for wp in payload.waypoints:
            if not (6 < wp.lat < 38 and 68 < wp.lon < 98):
                raise HTTPException(
                    status_code=400,
                    detail=f"Coordinates ({wp.lat},{wp.lon}) outside India bounds",
                )
            if wp.eta_mins < 0:
                raise HTTPException(status_code=400, detail="ETA minutes must be >= 0.")

        # Load radar state with TTL lazy-cache
        state = _load_radar_state(ttl_sec=RADAR_CACHE_TTL_SEC, force=False)
        clutter_mask = state["clutter_mask"]
        dx, dy, dir_from, dir_to, speed = state["movement"]
        latest_frame = state["latest_frame"]
        lag_info = state["lag_info"]

        # Convert to pixels and build (lat,lon,eta) tuples for enrichment
        waypoints_pixels = []
        waypoints_latlon = []
        for wp in payload.waypoints:
            px, py = latlon_to_pixel(wp.lat, wp.lon)
            waypoints_pixels.append((px, py, float(wp.eta_mins)))
            waypoints_latlon.append((float(wp.lat), float(wp.lon), float(wp.eta_mins)))

        max_eta = max((eta for (_la, _lo, eta) in waypoints_latlon), default=0.0)

        results = check_route_rain(
            waypoints_pixels,
            dx,
            dy,
            latest_frame,
            eta_minutes=max_eta,
            clutter_mask=clutter_mask,
            lag_mins=lag_info["lag_mins"],
        )

        enriched = enrich_results(
            results,
            waypoints_latlon,
            latest_frame,
            dx,
            dy,
            lag_info=lag_info,
        )

        for e in enriched:
            e["in_radar_bounds"] = is_within_radar(e["lat"], e["lon"])

        rain_wps = [e for e in enriched if e["rain_expected"]]
        clear_wps = [e for e in enriched if not e["rain_expected"]]
        first_rain = next((e for e in enriched if e["rain_expected"]), None)

        return {
            "total_waypoints": len(enriched),
            "rain_waypoints": len(rain_wps),
            "clear_waypoints": len(clear_wps),
            "first_rain_eta": first_rain["eta_mins"] if first_rain else None,
            "first_rain_label": first_rain["label"] if first_rain else None,
            "rain_direction_from": dir_from,
            "rain_direction_to": dir_to,
            "rain_speed_kmh": round(float(speed), 1),  # type: ignore
            "radar_lag_mins": lag_info["lag_mins"],
            "radar_freshness": lag_info["freshness"],
            "radar_message": lag_info["message"],
            "waypoints": enriched,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Waypoint prediction failed: {str(e)}\n{traceback.format_exc()}",
        )


# ENDPOINT 5: Predict Rain On Route

@app.post("/predict")
def predict_rain(route: RouteRequest):
    """
    Main endpoint.
    Takes start + end coordinates.
    Returns rain prediction for every 
    waypoint along the route.
    """
    try:
        # Validate coordinates are in India roughly
        for lat, lon in [
            (route.start_lat, route.start_lon),
            (route.end_lat, route.end_lon)
        ]:
            if not (6 < lat < 38 and 68 < lon < 98):
                raise HTTPException(
                    status_code=400,
                    detail=f"Coordinates ({lat},{lon}) outside India bounds"
                )

        # Load radar state with TTL lazy-cache (no download/re-extract if GIF fresh)
        state = _load_radar_state(ttl_sec=RADAR_CACHE_TTL_SEC, force=False)
        clutter_mask = state["clutter_mask"]
        dx, dy, dir_from, dir_to, speed = state["movement"]
        latest_frame = state["latest_frame"]
        lag_info = state["lag_info"]

        # Generate waypoints
        waypoints_latlon = generate_waypoints(
            (route.start_lat, route.start_lon),
            (route.end_lat, route.end_lon),
            spacing_km=route.spacing_km,
            eta_spacing_minutes=route.eta_spacing_minutes,
        )

        total_dist = haversine_km(
            route.start_lat, route.start_lon,
            route.end_lat, route.end_lon
        )

        # Convert to pixels
        waypoints_pixels = []
        for lat, lon, eta in waypoints_latlon:
            px, py = latlon_to_pixel(lat, lon)
            waypoints_pixels.append((px, py, eta))

        max_eta = waypoints_latlon[-1][2]

        # Run prediction
        results = check_route_rain(
            waypoints_pixels, dx, dy,
            latest_frame,
            eta_minutes=max_eta,
            clutter_mask=clutter_mask,
            lag_mins=lag_info["lag_mins"]
        )

        # Enrich with fuzzy intensity
        enriched = enrich_results(
            results, waypoints_latlon,
            latest_frame, dx, dy,
            lag_info=lag_info
        )

        # Add bounds check to each waypoint
        for e in enriched:
            e["in_radar_bounds"] = is_within_radar(
                e["lat"], e["lon"]
            )

        # Build summary
        rain_wps = [e for e in enriched if e["rain_expected"]]
        clear_wps = [e for e in enriched if not e["rain_expected"]]

        first_rain = next(
            (e for e in enriched if e["rain_expected"]), None
        )

        return {
            "route_distance_km": round(float(total_dist), 1),  # type: ignore
            "total_waypoints": len(enriched),
            "rain_waypoints": len(rain_wps),
            "clear_waypoints": len(clear_wps),
            "first_rain_eta": first_rain["eta_mins"] if first_rain else None,
            "first_rain_label": first_rain["label"] if first_rain else None,
            "rain_direction_from": dir_from,
            "rain_direction_to": dir_to,
            "rain_speed_kmh": round(float(speed), 1),  # type: ignore
            "radar_lag_mins": lag_info["lag_mins"],
            "radar_freshness": lag_info["freshness"],
            "radar_message": lag_info["message"],
            "waypoints": enriched
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        )

# RUN INSTRUCTIONS:
# cd backend
# .\venv\Scripts\activate
# uvicorn main:app --reload --port 8000
