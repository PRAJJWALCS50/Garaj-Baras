"""
Pure-NumPy fog rule for AIFS fields (no GRIB libraries -> unit-testable).
The backend has a copy of `classify` that reads the thresholds from the manifest.
"""
import numpy as np

import config as C


def ingredients(t2m, d2m, u10, v10, lcc):
    """Raw AIFS fields -> (dew-point depression K, wind m/s, low cloud 0..1)."""
    t2m = np.asarray(t2m, np.float32)
    d2m = np.asarray(d2m, np.float32)
    dpd = np.maximum(t2m - d2m, 0.0)
    wind = np.hypot(np.asarray(u10, np.float32), np.asarray(v10, np.float32))
    lcc = np.asarray(lcc, np.float32)
    if np.nanmax(lcc, initial=0) > 1.5:     # AIFS open data gives low cloud in %
        lcc = lcc / 100.0
    return dpd, wind, np.clip(lcc, 0, 1)


def classify(dpd, wind, lcc, rule=C.RULE):
    """0 clear / 1 possible / 2 fog, same shape as the inputs."""
    dpd, wind, lcc = (np.asarray(a, np.float32) for a in (dpd, wind, lcc))
    fog = ((dpd <= rule["fog_dpd_k"]) & (wind <= rule["fog_wind_ms"])) | (
        (lcc >= rule["lowcloud_lcc"]) & (dpd <= rule["lowcloud_dpd_k"]) & (wind <= rule["lowcloud_wind_ms"]))
    possible = (dpd <= rule["possible_dpd_k"]) & (wind <= rule["possible_wind_ms"])
    out = np.full(dpd.shape, C.CLEAR, np.uint8)
    out[possible] = C.POSSIBLE
    out[fog] = C.FOG
    return out


def risk(dpd, wind):
    """0..100 'how close to fog' score, for display only."""
    a = np.clip((2.5 - np.asarray(dpd, np.float32)) / 2.0, 0, 1)
    b = np.clip((6.0 - np.asarray(wind, np.float32)) / 3.0, 0, 1)
    return np.round(100 * a * b).astype(np.uint8)
