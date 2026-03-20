from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel  # type: ignore
from typing import List
import traceback

from radar import get_recent_frames, get_all_frames, get_radar_lag_mins  # type: ignore
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

# Global state - loaded once at startup
# Refreshed every request for now (can cache later)
radar_cache = {
    "clutter_mask": None,
    "last_loaded": None
}

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

# ENDPOINT 3: Predict Rain On Route

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

        # Load radar data
        all_frame_data = get_all_frames()
        recent_frame_data = get_recent_frames(n=6)
        all_paths = [p for (p, _ts) in all_frame_data]
        clutter_mask = build_clutter_mask(all_paths)

        # Get movement
        dx, dy, dir_from, dir_to, speed = get_movement_vector(
            recent_frame_data, clutter_mask=clutter_mask
        )
        latest_frame = recent_frame_data[-1][0] if recent_frame_data else None
        latest_ts = recent_frame_data[-1][1] if recent_frame_data else None
        lag_info = get_radar_lag_mins(latest_ts)

        # Generate waypoints
        waypoints_latlon = generate_waypoints(
            (route.start_lat, route.start_lon),
            (route.end_lat, route.end_lon),
            spacing_km=route.spacing_km
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
