"""
AIFS fog worker - settings.

ECMWF AIFS Single (machine-learning model, free open data): 4 runs a day
(00/06/12/18 UTC), 6-hourly steps, 0.25 deg grid. AIFS has no visibility field,
so fog is diagnosed from the classic fog ingredients at each step:
  * air almost saturated  -> small dew-point depression (2 m temperature - 2 m dew point)
  * light wind            -> 10 m wind speed (fog needs calm air to form / survive)
  * low cloud             -> AIFS low cloud cover (fog is low cloud touching the ground)

The rule below is a first guess from the fog literature, NOT yet calibrated on
METAR. It is written into every manifest, and the backend reads it from there,
so worker and backend always use the same thresholds.
"""

# ── Region: same box as the satellite fog map (North India / IGP) ───────────
BOX = {"lon_min": 72.0, "lon_max": 89.0, "lat_min": 23.0, "lat_max": 33.0}

# ── AIFS open data ──────────────────────────────────────────────────────────
MODEL = "aifs-single"
SOURCES = ("ecmwf", "aws", "azure")      # tried in order (ECMWF server can be busy)
PARAMS = ("2t", "2d", "10u", "10v", "lcc")
STEPS = tuple(range(0, 73, 6))           # 0..72 h: covers a long drive even from a 12 h old run

# ── Fog rule (per grid cell, per time) ──────────────────────────────────────
RULE = {
    "fog_dpd_k": 1.0,        # dew-point depression <= 1 K ...
    "fog_wind_ms": 3.0,      # ... and wind <= 3 m/s                         -> FOG
    "lowcloud_lcc": 0.8,     # low cloud >= 80 % ...
    "lowcloud_dpd_k": 1.5,   # ... with dpd <= 1.5 K and wind <= 4 m/s        -> FOG
    "lowcloud_wind_ms": 4.0,
    "possible_dpd_k": 2.0,   # dpd <= 2 K and wind <= 5 m/s                  -> POSSIBLE
    "possible_wind_ms": 5.0,
}
CLEAR, POSSIBLE, FOG = 0, 1, 2

# ── Storage: same public Supabase bucket as the satellite, own folder ───────
BUCKET = "fog-satellite"
PREFIX = "aifs"
MANIFEST = "aifs/latest.json"
KEEP_RUNS = 2
