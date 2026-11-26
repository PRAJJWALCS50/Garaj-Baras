"""
Satellite fog worker - settings.

Everything tunable lives here. The backend (backend/fog_satellite.py) reads the
grid geometry from each scene's manifest, so changing the grid here does not
require a backend change.
"""

# ── Region: North India / Indo-Gangetic Plain, regular lat/lon grid ─────────
GRID = {
    "lon_min": 72.0, "lon_max": 89.0,   # Punjab/Rajasthan edge -> West Bengal
    "lat_min": 23.0, "lat_max": 33.0,
    "res": 0.05,                        # degrees (~5.5 km N-S, ~4.9 km E-W)
}

# ── Satellite: EUMETSAT Meteosat-9 over the Indian Ocean (45.5 E), SEVIRI ────
EUMETSAT_COLLECTION = "EO:EUM:DAT:MSG:HRSEVIRI-IODC"
SATPY_READER = "seviri_l1b_native"
CH_039 = "IR_039"   # 3.9 um  (fog droplets look colder here at night)
CH_108 = "IR_108"   # 10.8 um (cloud-top / surface temperature)
SEARCH_BACK_MIN = 50          # look for the newest scan in the last N minutes
RESAMPLE_RADIUS_M = 10_000

# ── Night-time fog test (per pixel) ─────────────────────────────────────────
# Fog / low stratus at night: T10.8 - T3.9 above BTD_FOG_K, and the cloud top
# is warm (near the ground). Cold tops = high cloud = we cannot see below.
BTD_FOG_K = 2.5               # first guess; tune against airport METAR
T108_HIGH_CLOUD_K = 255.0     # colder than this -> "unknown" (blind)
SUN_ELEV_NIGHT_DEG = -3.0     # sun higher than this -> 3.9 um contaminated -> "unknown"

# Pixel classes written to the fog map
CLEAR, FOG, UNKNOWN = 0, 1, 2

# ── Products ────────────────────────────────────────────────────────────────
DIST_MAX_KM = 150.0           # distance-to-fog is capped here
SPEED_MAX_KMH = 40.0          # fog edges faster than this are treated as noise
SPEED_MAX_GAP_MIN = 75        # frames further apart than this -> no speed
SPEED_AGREE_KMH = 6.0         # two speed estimates must agree within this (or 50%)

# ── Storage (Supabase Storage, public-read bucket) ──────────────────────────
BUCKET = "fog-satellite"
MANIFEST = "latest.json"      # newest scene + the two before it
SCENE_DIR = "scenes"          # scenes/YYYYMMDD_HHMM.npz (history kept)
