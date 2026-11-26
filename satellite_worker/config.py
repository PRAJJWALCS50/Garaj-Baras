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
# Fog / low stratus at night: T10.8 - T3.9 above a threshold, and the cloud top
# is warm (near the ground). Cold tops = high cloud = we cannot see below.
#
# The threshold is found PER SCENE: the BTD histogram of a fog night has two humps
# (clear land ~2 K, fog ~5.5 K on 3-4 Jan 2025); the valley between them is the
# cut-off. Clear-night background shifts with humidity and ground (west Rajasthan
# desert reads 3.2-4.1 K even when clear), so a fixed 2.5 K marked half the
# region as fog. If no valley is found, BTD_FOG_K is used.
BTD_FOG_K = 4.25              # fallback (valley on the 3-4 Jan 2025 test night)
BTD_SEARCH_K = (3.25, 5.25)   # look for the valley in this range
BTD_CLAMP_K = (3.75, 5.0)     # never use a threshold outside this range
CLEAN_MIN_NEIGHBOURS = 3      # fog pixel with fewer fog neighbours (of 8) -> noise, removed
FILL_MIN_NEIGHBOURS = 6       # clear pixel with this many fog neighbours -> hole, filled
T108_HIGH_CLOUD_K = 255.0     # colder than this -> "unknown" (blind)
SUN_ELEV_NIGHT_DEG = -3.0     # sun higher than this -> 3.9 um contaminated -> "unknown"

# Pixel classes written to the fog map
CLEAR, FOG, UNKNOWN = 0, 1, 2

# ── Products ────────────────────────────────────────────────────────────────
DIST_MAX_KM = 150.0           # distance-to-fog is capped here

# ── Fog patch tracking ──────────────────────────────────────────────────────
# Touching fog pixels (8 neighbours) form a patch. Each patch is matched with the
# previous scene by OVERLAP (fog barely moves, so a patch overlaps itself):
#   no overlap -> born, one parent -> tracked, several parents -> merged,
#   parent with several children -> split, no usable previous scene -> untracked
PATCH_MIN_PIXELS = 3          # smaller fog specks are not tracked (still count as fog)
PATCH_OVERLAP_MIN = 0.3       # overlap / (smaller of the two patches) to call them linked
PATCH_MAX_GAP_MIN = 75        # previous scene older than this -> no tracking
PATCH_GROWTH_MAX_KMH = 40.0   # faster radius change = noise -> no growth value
PATCH_DRIFT_MAX_KMH = 60.0    # faster centre movement = noise -> no drift value
PATCH_SMOOTH = 0.5            # weight of the new growth/drift vs the patch's previous value

# ── Storage (Supabase Storage, public-read bucket) ──────────────────────────
BUCKET = "fog-satellite"
MANIFEST = "latest.json"      # newest scene + the two before it
SCENE_DIR = "scenes"          # scenes/YYYYMMDD_HHMM.npz (history kept)
