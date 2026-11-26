"""
Pure-NumPy fog products (no satellite libraries needed -> unit-testable).

    classify()     T3.9, T10.8, sun elevation  -> fog map  (0 clear, 1 fog, 2 unknown)
    distance_km()  fog map                     -> km to the nearest fog pixel
    edge_speed()   distance now / 30 / 60 min  -> km/h the fog edge is approaching
"""
import math
from datetime import datetime

import numpy as np
from scipy import ndimage

import config as C


def grid_axes(grid=C.GRID):
    """Pixel-centre latitudes (north -> south) and longitudes (west -> east)."""
    lats = np.round(np.arange(grid["lat_max"], grid["lat_min"] - 1e-9, -grid["res"]), 6)
    lons = np.round(np.arange(grid["lon_min"], grid["lon_max"] + 1e-9, grid["res"]), 6)
    return lats, lons


def sun_elevation(when: datetime, lats, lons):
    """Solar elevation (deg) for every pixel, NOAA approximation. `when` is UTC."""
    doy = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60 + when.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * np.asarray(lons)[None, :]        # true solar time, min
    ha = np.radians(tst / 4 - 180)
    lat = np.radians(np.asarray(lats))[:, None]
    cos_zen = np.sin(lat) * math.sin(decl) + np.cos(lat) * math.cos(decl) * np.cos(ha)
    return 90 - np.degrees(np.arccos(np.clip(cos_zen, -1, 1)))


def classify(t039, t108, sun_elev):
    """Per-pixel fog map. Unknown wins over everything we cannot trust."""
    t039 = np.asarray(t039, dtype=np.float32)
    t108 = np.asarray(t108, dtype=np.float32)
    btd = t108 - t039
    out = np.full(t108.shape, C.CLEAR, dtype=np.uint8)
    out[btd > C.BTD_FOG_K] = C.FOG
    blind = (
        ~np.isfinite(btd)
        | (t108 < C.T108_HIGH_CLOUD_K)            # high cloud on top
        | (np.asarray(sun_elev) > C.SUN_ELEV_NIGHT_DEG)   # daylight/twilight
    )
    out[blind] = C.UNKNOWN
    return out


def pixel_km(grid=C.GRID):
    """(north-south, east-west) pixel size in km at the grid's mid latitude."""
    mid = math.radians((grid["lat_min"] + grid["lat_max"]) / 2)
    ns = grid["res"] * 111.32
    return ns, ns * math.cos(mid)


def distance_km(fogmap, grid=C.GRID):
    """km from every pixel to the nearest FOG pixel (0 on fog), capped."""
    fog = fogmap == C.FOG
    if not fog.any():
        return np.full(fogmap.shape, C.DIST_MAX_KM, dtype=np.float32)
    d = ndimage.distance_transform_edt(~fog, sampling=pixel_km(grid))
    return np.minimum(d, C.DIST_MAX_KM).astype(np.float32)


def edge_speed(d_now, d_prev1=None, gap1_min=None, d_prev2=None, gap2_min=None):
    """
    Approach speed of the fog edge (km/h, positive = getting closer).

    d_prev1 is the frame ~30 min before d_now, d_prev2 ~30 min before d_prev1.
    With both, the two estimates must agree, otherwise the pixel gets NaN
    (one noisy frame must not create a fake 20 km jump). With only d_prev1 the
    single estimate is used. NaN where there is no usable history, where the
    pixel is already foggy, or where fog is far away / receding.
    """
    nan = np.full(d_now.shape, np.nan, dtype=np.float32)
    if d_prev1 is None or not gap1_min or gap1_min > C.SPEED_MAX_GAP_MIN:
        return nan
    s1 = (d_prev1 - d_now) / (gap1_min / 60.0)
    if d_prev2 is not None and gap2_min and gap2_min <= C.SPEED_MAX_GAP_MIN:
        s2 = (d_prev2 - d_prev1) / (gap2_min / 60.0)
        agree = (np.sign(s1) == np.sign(s2)) & (
            np.abs(s1 - s2) <= np.maximum(C.SPEED_AGREE_KMH, 0.5 * np.abs(s1)))
        speed = np.where(agree, (s1 + s2) / 2, np.nan)
    else:
        speed = s1
    valid = (d_now > 0) & (d_now < C.DIST_MAX_KM) & (speed > 0) & (speed <= C.SPEED_MAX_KMH)
    return np.where(valid, speed, np.nan).astype(np.float32)


def scene_stats(fogmap):
    total = fogmap.size
    return {
        "fog_share": round(float((fogmap == C.FOG).sum() / total), 4),
        "unknown_share": round(float((fogmap == C.UNKNOWN).sum() / total), 4),
        "clear_share": round(float((fogmap == C.CLEAR).sum() / total), 4),
    }
