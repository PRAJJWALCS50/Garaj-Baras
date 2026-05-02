# Garaj Baras - cloud_cover.py
#
# Lightweight cloud-cover cross-check for IMD radar predictions.
#
# IMD radar occasionally flags "rain" on ground clutter / RF noise / bright
# banding where no rain actually exists. To filter those cases, we fetch a
# cloud-cover forecast from Open-Meteo (free, no API key, high rate limit)
# at each rain-flagged waypoint's ETA hour. If the sky there is basically
# clear, we trust the weather model over IMD and flip the waypoint to "No Rain".
#
# Design notes:
#   - Waypoints are quantized to a 0.25° (~28 km) grid, so the handful of
#     waypoints on a typical NCR route collapse to ONE Open-Meteo call.
#   - Results are cached in-process for 15 minutes per grid cell.
#   - Network failures / timeouts are fail-open: caller falls back to IMD.
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests  # type: ignore


logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
IST = timezone(timedelta(hours=5, minutes=30))

# How long we trust a cached forecast. Clouds don't move so fast that a
# 15-min-old forecast is misleading for a 1-2h route prediction.
_CACHE_TTL_SEC = 15 * 60

# Grid size in degrees for locality key. 0.25° latitude ≈ 28 km, longitude
# ≈ 24 km at Delhi's latitude. Short NCR routes typically collapse to 1 cell.
_GRID_DEG = 0.25

# Network budget per fetch. Predict latency must not balloon if this API
# is slow; 3s is generous for a static JSON response from Open-Meteo.
_REQUEST_TIMEOUT_SEC = 3.0

_cache_lock = threading.Lock()
_cache: dict[tuple[float, float], tuple[float, dict]] = {}


def _round_to_grid(lat: float, lon: float) -> tuple[float, float]:
    """Quantize lat/lon to the grid so nearby waypoints share one cache entry."""
    return (
        round(lat / _GRID_DEG) * _GRID_DEG,
        round(lon / _GRID_DEG) * _GRID_DEG,
    )


def _fetch_forecast(lat: float, lon: float) -> Optional[dict]:
    """
    Fetch cloud-cover forecast from Open-Meteo for the given grid cell.

    Returns a dict with keys 'times' (list[datetime in IST]) and 'cloud_cover'
    (list[int 0..100]). Returns None on any failure (network, HTTP, parse).
    """
    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "cloud_cover",
                "forecast_days": 2,
                "timezone": "Asia/Kolkata",
            },
            timeout=_REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("cloud_cover fetch failed for (%.2f,%.2f): %s", lat, lon, e)
        return None

    hourly = data.get("hourly") or {}
    times_iso = hourly.get("time") or []
    cloud = hourly.get("cloud_cover") or []

    if not times_iso or not cloud or len(times_iso) != len(cloud):
        logger.warning("cloud_cover response malformed for (%.2f,%.2f)", lat, lon)
        return None

    # Open-Meteo returns ISO without tz suffix when timezone=Asia/Kolkata.
    # Parse and attach IST explicitly for robust comparisons downstream.
    try:
        times = [
            datetime.fromisoformat(t).replace(tzinfo=IST)
            for t in times_iso
        ]
    except Exception as e:
        logger.warning("cloud_cover time parse failed: %s", e)
        return None

    return {"times": times, "cloud_cover": cloud}


def _get_forecast_cached(lat: float, lon: float) -> Optional[dict]:
    """Cache-wrapped forecast fetch. Key is the grid cell, TTL is 15 min."""
    key = _round_to_grid(lat, lon)
    now = time.time()

    with _cache_lock:
        hit = _cache.get(key)
    if hit and (now - hit[0]) < _CACHE_TTL_SEC:
        return hit[1]

    data = _fetch_forecast(key[0], key[1])
    if data is None:
        return None

    with _cache_lock:
        _cache[key] = (now, data)
    return data


def cloud_cover_at(lat: float, lon: float, when: datetime) -> Optional[int]:
    """
    Return cloud cover percentage (0-100) at ``lat``/``lon`` for the forecast
    hour closest to ``when``. Returns None if data is unavailable.
    """
    data = _get_forecast_cached(lat, lon)
    if not data:
        return None

    times: list[datetime] = data["times"]
    cc: list = data["cloud_cover"]
    if not times:
        return None

    # Compare in UTC to avoid tz edge-cases if ``when`` is naive.
    target = when if when.tzinfo else when.replace(tzinfo=IST)
    target_utc = target.astimezone(timezone.utc)

    best_i = 0
    best_d = float("inf")
    for i, t in enumerate(times):
        d = abs((t.astimezone(timezone.utc) - target_utc).total_seconds())
        if d < best_d:
            best_d = d
            best_i = i

    val = cc[best_i]
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    # Smoke test: fetch Delhi cloud cover for the next couple hours.
    now_ist = datetime.now(IST)
    lat, lon = 28.5562, 77.1000  # IGI
    for offset in (0, 30, 60, 120):
        when = now_ist + timedelta(minutes=offset)
        cc = cloud_cover_at(lat, lon, when)
        print(f"+{offset:>3}m @ {when.strftime('%H:%M')} IST -> cloud_cover={cc}%")
