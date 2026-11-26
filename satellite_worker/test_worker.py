"""
Offline tests for the satellite fog pipeline (no EUMETSAT, no satpy, no Supabase).

    python -m pytest satellite_worker/test_worker.py -q

The end-to-end test fakes the satellite: a fog blanket centred on the Ganga at
Garhmukteshwar grows over three 30-min scenes. The REAL worker code turns those
scenes into fog maps / distance / tracked fog patches in a local folder, and the
REAL backend module (backend/fog_satellite.py) answers a Moradabad -> Delhi route.
"""
import importlib
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C   # noqa: E402
import fogmask as F  # noqa: E402

LATS, LONS = F.grid_axes()
NIGHT = datetime(2025, 1, 10, 21, 0)      # 02:30 IST


# ── fogmask unit tests ──────────────────────────────────────────────────────
def test_grid_shape():
    assert LATS[0] == 33.0 and LATS[-1] == 23.0 and LONS[0] == 72.0 and LONS[-1] == 89.0
    assert (len(LATS), len(LONS)) == (201, 341)


def test_sun_elevation_day_night():
    noon = F.sun_elevation(datetime(2025, 1, 10, 6, 30), [28.6], [77.2])[0, 0]
    night = F.sun_elevation(NIGHT, [28.6], [77.2])[0, 0]
    assert 35 < noon < 45          # Delhi winter noon ~38 deg
    assert night < -40


def test_classify_rules():
    rng = np.random.default_rng(0)
    t108 = np.full((80, 80), 285.0, np.float32)
    btd = rng.normal(2.2, 0.4, t108.shape).astype(np.float32)      # clear land ~2.2 K
    btd[10:50, 10:50] = rng.normal(5.5, 0.4, (40, 40))              # fog blanket ~5.5 K
    btd[60:70, 60:70] = 2.0
    t108[60:70, 60:70] = 240.0                                      # cold high cloud
    btd[0, 0] = np.nan
    btd[75, 5] = 6.0                                                # single-pixel speck
    sun = np.full(t108.shape, -50.0)
    out, info = F.classify(t108 - btd, t108, sun, return_info=True)
    assert info["threshold_method"] == "valley" and 3.75 <= info["btd_threshold_k"] <= 5.0
    assert (out[15:45, 15:45] == C.FOG).all()
    assert (out[55:58, 5:50] == C.CLEAR).all()
    assert (out[60:70, 60:70] == C.UNKNOWN).all() and out[0, 0] == C.UNKNOWN
    assert out[75, 5] == C.CLEAR                                    # speck removed
    assert (F.classify(t108 - btd, t108, np.full(t108.shape, 10.0)) == C.UNKNOWN).all()
    # no fog hump (clear night) -> fallback threshold, nothing marked fog
    clear, info = F.classify(t108 - rng.normal(2.2, 0.4, t108.shape), t108, sun, return_info=True)
    assert info["threshold_method"] == "fallback" and (clear != C.FOG).all()


def test_distance():
    fog = np.zeros((50, 50), np.uint8)
    fog[25, 10] = C.FOG
    d = F.distance_km(fog)
    ns, ew = F.pixel_km()
    assert d[25, 10] == 0 and abs(d[25, 20] - 10 * ew) < 0.01
    assert (F.distance_km(np.zeros((5, 5), np.uint8)) == C.DIST_MAX_KM).all()


# ── fog patches ─────────────────────────────────────────────────────────────
def disk(lat, lon, radius_km):
    yy = (LATS[:, None] - lat) * 111.32
    xx = (LONS[None, :] - lon) * 111.32 * np.cos(np.radians(lat))
    return np.hypot(yy, xx) <= radius_km


def fogmap(*masks):
    out = np.zeros((len(LATS), len(LONS)), np.uint8)
    for m in masks:
        out[m] = C.FOG
    return out


def step(fm, prev=None, gap=30, next_id=1):
    labels, props = F.label_patches(fm, LATS, LONS)
    ids_prev, tab_prev = prev if prev else (None, None)
    return F.track_patches(labels, props, ids_prev, tab_prev, gap if prev else None, next_id)


def by_status(table, name):
    return np.flatnonzero(table["status"] == F.STATUS[name])


def test_label_patches_ignores_specks():
    fm = fogmap(disk(28, 78, 15), disk(26, 82, 15))
    fm[100, 300] = C.FOG                                   # 1-pixel speck
    labels, props = F.label_patches(fm, LATS, LONS)
    assert labels.max() == 2 and len(props["pixels"]) == 2
    assert all(600 < a < 800 for a in props["area_km2"])   # pi*15^2 ~ 707 km2


def test_tracked_patch_growth_and_drift():
    ids1, t1, nid = step(fogmap(disk(28.8, 78.1, 10)))
    assert by_status(t1, "untracked").size == 1            # first scene: nothing to compare with
    ids2, t2, nid = step(fogmap(disk(28.8, 78.1, 14)), (ids1, t1), 30, nid)
    j = by_status(t2, "tracked")
    assert j.size == 1 and t2["id"][j[0]] == t1["id"][0]   # same id carried over
    assert 6 <= t2["growth_kmh"][j[0]] <= 10                # radius +4 km in 30 min ~ 8 km/h
    assert abs(t2["drift_e_kmh"][j[0]]) < 2
    # moves 6 km east in 30 min, same size -> drift ~12 km/h east
    ids3, t3, nid = step(fogmap(disk(28.8, 78.1 + 6 / (111.32 * np.cos(np.radians(28.8))), 14)), (ids2, t2), 30, nid)
    j = by_status(t3, "tracked")[0]
    assert 3 <= t3["drift_e_kmh"][j] <= 12                  # smoothed with the previous (0) drift


def test_born_merged_split_untracked():
    ids1, t1, nid = step(fogmap(disk(28.0, 78.0, 10), disk(28.0, 78.35, 10), disk(26.0, 82.0, 12)))
    # the two near patches join; the far one splits into two; a new patch appears
    split_a = disk(25.95, 81.93, 7)
    split_b = disk(26.05, 82.07, 7)
    ids2, t2, nid = step(fogmap(disk(28.0, 78.17, 22), split_a, split_b, disk(30.0, 75.0, 8)), (ids1, t1), 30, nid)
    assert by_status(t2, "merged").size == 1
    assert by_status(t2, "split").size == 2
    born = by_status(t2, "born")
    assert born.size == 1 and np.isnan(t2["growth_kmh"][born[0]])   # new fog has no speed
    assert np.isnan(t2["growth_kmh"][by_status(t2, "split")]).all()  # ambiguous -> no speed
    assert len(set(t2["id"].tolist())) == len(t2["id"])             # ids unique
    # previous scene too old -> untracked, not "born" (no false "forming" after outages)
    _, t3, _ = step(fogmap(disk(30.0, 75.0, 8)), (ids2, t2), 120, nid)
    assert by_status(t3, "untracked").size == 1 and by_status(t3, "born").size == 0


def _scene_for_backend(ids, table, fm):
    import fog_satellite as B
    return {"fog": fm, "dist_km": F.distance_km(fm), "patch_id": ids,
            "patches": {c: table[c] for c in B.PATCH_COLS},
            "patch_row": {int(p): i for i, p in enumerate(table["id"])}}


def test_new_patch_is_forming_not_a_fake_speed():
    """User's case: 12:00 nearest fog 20 km away; 12:30 a NEW patch appears 5 km away.
    Must NOT be read as fog moving 30 km/h; must say 'fog forming nearby'."""
    sys.path.insert(0, str(HERE.parent / "backend"))
    import fog_satellite as B
    wp = (28.84, 78.77)                                       # Moradabad
    km_lon = 111.32 * np.cos(np.radians(wp[0]))
    patch_a = lambda r: disk(wp[0], wp[1] - (20 + r) / km_lon, r)     # edge 20 km west
    ids1, t1, nid = step(fogmap(patch_a(10)))
    # NEW patch, edge ~7 km north (just outside the 3x3 window, so the point itself is not foggy)
    new_b = disk(wp[0] + (7 + 6) / 111.32, wp[1], 6)
    fm2 = fogmap(patch_a(10), new_b)
    ids2, t2, nid = step(fm2, (ids1, t1), 30, nid)
    assert by_status(t2, "born").size == 1
    scene = _scene_for_backend(ids2, t2, fm2)
    grid = {**C.GRID}
    r = B.classify_point(scene, grid, {"fog": 1, "unknown": 2}, wp[0], wp[1], eta_mins=20)
    assert r["status"] == "forming", r
    assert r["nearest_patch"]["status"] == "born" and r["nearest_patch"]["approach_kmh"] is None
    assert r["fog_eta_mins"] is None                          # no fake 30 km/h arrival
    # without the new patch: patch A is not growing -> just clear
    fm1 = fogmap(patch_a(10))
    ids1b, t1b, _ = step(fm1, (ids1, t1), 30, nid)
    r = B.classify_point(_scene_for_backend(ids1b, t1b, fm1), grid, {"fog": 1, "unknown": 2}, wp[0], wp[1], 20)
    assert r["status"] == "clear" and r["fog_eta_mins"] is None


def test_growing_patch_is_approaching():
    sys.path.insert(0, str(HERE.parent / "backend"))
    import fog_satellite as B
    wp = (28.84, 78.77)
    km_lon = 111.32 * np.cos(np.radians(wp[0]))
    centre = (wp[0], wp[1] - 40 / km_lon)                     # centre 40 km west
    ids1, t1, nid = step(fogmap(disk(*centre, 20)))           # edge 20 km away
    fm2 = fogmap(disk(*centre, 26))                           # +6 km radius in 30 min -> ~12 km/h
    ids2, t2, nid = step(fm2, (ids1, t1), 30, nid)
    scene = _scene_for_backend(ids2, t2, fm2)
    cls = {"fog": 1, "unknown": 2}
    late = B.classify_point(scene, C.GRID, cls, wp[0], wp[1], eta_mins=90)    # driver in 90 min
    early = B.classify_point(scene, C.GRID, cls, wp[0], wp[1], eta_mins=10)   # driver in 10 min
    assert late["status"] == "approaching" and 50 <= late["fog_eta_mins"] <= 110, late
    assert early["status"] == "clear"


# ── end-to-end: fake satellite -> worker -> local store -> backend ──────────
GANGA = (28.80, 78.10)       # Garhmukteshwar
ROUTE = [  # Moradabad -> Delhi, request at the newest scan time
    ("Moradabad", 28.84, 78.77, 0),
    ("Gajraula", 28.84, 78.24, 45),
    ("Garhmukteshwar", 28.79, 78.10, 75),
    ("Hapur", 28.73, 77.78, 105),
    ("Ghaziabad", 28.67, 77.43, 140),
    ("Delhi", 28.61, 77.21, 165),
]


def fake_scene(radius_km, cloud_over=None):
    """Brightness temperatures for a circular fog blanket around the Ganga."""
    ns, ew = F.pixel_km()
    yy = (LATS[:, None] - GANGA[0]) * 111.32
    xx = (LONS[None, :] - GANGA[1]) * 111.32 * np.cos(np.radians(28))
    r = np.hypot(yy, xx)
    t108 = np.full(r.shape, 284.0, np.float32)
    t039 = t108 - 0.5                               # clear land: small BTD
    t039[r <= radius_km] = t108[r <= radius_km] - 4.0   # fog: BTD 4 K
    if cloud_over:                                  # cold high cloud over a town
        la, lo = cloud_over
        cloud = (np.abs(LATS[:, None] - la) < 0.15) & (np.abs(LONS[None, :] - lo) < 0.15)
        t108[cloud] = 235.0
        t039[cloud] = 233.0
    return t039, t108


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Run the real worker 3 times (fog radius 20 -> 28 -> 36 km, 30 min apart)."""
    import run as worker
    frames = [(datetime(2025, 1, 10, 20, 0), 20.0), (datetime(2025, 1, 10, 20, 30), 28.0),
              (datetime(2025, 1, 10, 21, 0), 36.0)]
    current = {}

    fake = types.ModuleType("fetch")
    fake.collection = lambda: object()
    fake.nearest_product = lambda coll, when: when
    fake.newest_product = lambda coll, when: when
    fake.scan_time = lambda p: p

    def download_native(product, workdir):
        p = Path(workdir) / "scene.nat"
        p.write_bytes(b"x" * 1000)
        return p
    fake.download_native = download_native
    def to_grid(nat):
        a = current["arrays"]
        return a if isinstance(a, dict) else {"t039": a[0], "t108": a[1]}
    fake.to_grid = to_grid
    monkeypatch.setitem(sys.modules, "fetch", fake)

    for t, radius in frames:
        current["arrays"] = fake_scene(radius, cloud_over=(28.67, 77.43))   # cloud over Ghaziabad
        args = types.SimpleNamespace(local=str(tmp_path), time=t.strftime("%Y-%m-%dT%H:%M"),
                                     quicklook=None, force=False)
        assert worker.process(args) == 0

    # backend reads the same folder; pretend "now" is right after the newest scan
    sys.path.insert(0, str(HERE.parent / "backend"))
    monkeypatch.setenv("FOG_SAT_BASE_URL", f"file://{tmp_path}")
    import fog_satellite
    fog_satellite = importlib.reload(fog_satellite)
    monkeypatch.setattr(fog_satellite, "datetime", _FrozenDatetime)
    return tmp_path, fog_satellite


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        from datetime import timezone
        return datetime(2025, 1, 10, 21, 5, tzinfo=timezone.utc)


def test_end_to_end_route(pipeline):
    folder, fog_satellite = pipeline
    import json
    manifest = json.loads((folder / "latest.json").read_text())
    assert manifest["status"] == "ok" and len(manifest["scenes"]) == 3
    assert manifest["scenes"][0]["time"] == "2025-01-10T21:00:00Z"

    res = fog_satellite.route_fog({"lat": la, "lon": lo, "eta_mins": eta} for _, la, lo, eta in ROUTE)
    assert res["scene"]["state"] == "ok"
    got = {name: p for (name, *_), p in zip(ROUTE, res["points"])}
    for name, p in got.items():
        print(name, p["status"], p.get("dist_km"), p.get("nearest_patch"), p.get("fog_eta_mins"))

    assert got["Garhmukteshwar"]["status"] == "fog"          # inside the blanket
    assert got["Gajraula"]["status"] in ("approaching", "fog", "patchy")
    assert got["Hapur"]["status"] in ("fog", "patchy", "approaching")
    assert got["Ghaziabad"]["status"] == "unknown"           # high cloud -> never "clear"
    assert got["Delhi"]["status"] == "clear"                 # far from fog
    # the blanket is ONE tracked patch growing 8 km per 30 min -> ~16 km/h
    moradabad = got["Moradabad"]
    assert moradabad["nearest_patch"]["status"] == "tracked"
    assert 10 <= moradabad["nearest_patch"]["approach_kmh"] <= 22
    assert moradabad["status"] == "clear"        # driver leaves before the fog arrives


def test_backend_states(pipeline, monkeypatch):
    folder, fog_satellite = pipeline
    import json
    # stale: newest scene 2 h old
    class Late(datetime):
        @classmethod
        def now(cls, tz=None):
            from datetime import timezone
            return datetime(2025, 1, 10, 23, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(fog_satellite, "datetime", Late)
    fog_satellite._state["checked"] = 0
    assert fog_satellite.route_fog([{"lat": 28.8, "lon": 78.1, "eta_mins": 0}])["points"][0]["status"] == "stale"
    # daytime manifest
    m = json.loads((folder / "latest.json").read_text())
    (folder / "latest.json").write_text(json.dumps({**m, "status": "day", "scenes": []}))
    fog_satellite._state["checked"] = 0
    assert fog_satellite.route_fog([{"lat": 28.8, "lon": 78.1, "eta_mins": 0}])["points"][0]["status"] == "daytime"
    # outside the grid (Bengaluru) with a valid scene
    (folder / "latest.json").write_text(json.dumps(m))
    monkeypatch.setattr(fog_satellite, "datetime", _FrozenDatetime)
    fog_satellite._state["checked"] = 0
    assert fog_satellite.route_fog([{"lat": 12.97, "lon": 77.59, "eta_mins": 0}])["points"][0]["status"] == "outside"


def test_backend_no_config(monkeypatch):
    sys.path.insert(0, str(HERE.parent / "backend"))
    monkeypatch.delenv("FOG_SAT_BASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    import fog_satellite
    fog_satellite = importlib.reload(fog_satellite)
    res = fog_satellite.route_fog([{"lat": 28.8, "lon": 78.1, "eta_mins": 0}])
    assert res["scene"]["state"] == "no_data" and res["points"][0]["status"] == "no_data"


# ── Daytime method ──────────────────────────────────────────────────────────
def day_scene(sun=35.0, seed=0):
    """Dark land (refl ~0.12) with a bright fog sheet, a cold cloud, cumulus and snow."""
    rng = np.random.default_rng(seed)
    shape = (len(LATS), len(LONS))
    sun_e = np.full(shape, sun, np.float32)
    s = np.sin(np.radians(sun))
    refl = rng.normal(0.12, 0.02, shape).astype(np.float32)        # clear ground
    nir = refl * 1.5                                                # land: brighter at 1.6 um
    t108 = np.full(shape, 290.0, np.float32)
    fog = np.zeros(shape, bool); fog[60:120, 80:200] = True        # fog sheet
    refl[fog] = rng.normal(0.55, 0.01, fog.sum()); nir[fog] = refl[fog] * 0.7; t108[fog] = 278.0
    cold = np.zeros(shape, bool); cold[150:180, 20:60] = True      # mid-level cloud
    refl[cold] = 0.6; nir[cold] = 0.4; t108[cold] = 260.0
    cu = np.zeros(shape, bool); cu[150:190, 250:320] = True        # lumpy cumulus, warm top
    refl[cu] = np.where(rng.random(cu.sum()) > 0.5, 0.75, 0.3); nir[cu] = refl[cu] * 0.7; t108[cu] = 280.0
    snow = np.zeros(shape, bool); snow[0:15, 0:60] = True           # Himalayan snow
    refl[snow] = 0.7; nir[snow] = 0.08; t108[snow] = 272.0
    t039 = t108 + 15                                                # 3.9 um swamped by sunlight
    vis = refl * s * 100                                            # satpy percent, before sun correction
    nir16 = nir * s * 100
    return dict(t039=t039, t108=t108, vis06=vis, nir16=nir16), sun_e, dict(fog=fog, cold=cold, cu=cu, snow=snow)


def test_daytime_fog_from_visible():
    g, sun, m = day_scene()
    out, info = F.classify(g["t039"], g["t108"], sun, return_info=True, vis06=g["vis06"], nir16=g["nir16"])
    inner = np.zeros_like(m["fog"]); inner[62:118, 82:198] = True
    assert (out[inner] == C.FOG).mean() > 0.98, "fog sheet found by day"
    assert (out[m["cold"]] == C.UNKNOWN).mean() > 0.9, "cold bright cloud hides the ground"
    assert (out[m["cu"]] == C.FOG).mean() < 0.05, "lumpy cumulus is not fog"
    assert (out[m["snow"]] != C.FOG).all(), "snow is not fog"
    clear = ~(m["fog"] | m["cold"] | m["cu"] | m["snow"])
    assert (out[clear] == C.CLEAR).mean() > 0.97
    assert info["day_share"] == 1.0 and C.VIS_CLAMP[0] <= info["vis_threshold"] <= C.VIS_CLAMP[1]


def test_low_sun_correction_keeps_fog_bright():
    g, sun, m = day_scene(sun=10.0)        # early morning: raw reflectance ~ 1/6 as bright
    out = F.classify(g["t039"], g["t108"], sun, vis06=g["vis06"], nir16=g["nir16"])
    assert (out[65:115, 85:195] == C.FOG).mean() > 0.98


def test_twilight_is_unknown_and_day_without_vis_is_unknown():
    g, _, _ = day_scene()
    twilight = np.full(g["t108"].shape, 3.0, np.float32)
    assert (F.classify(g["t039"], g["t108"], twilight, vis06=g["vis06"]) == C.UNKNOWN).all()
    day = np.full(g["t108"].shape, 35.0, np.float32)
    assert (F.classify(g["t039"], g["t108"], day) == C.UNKNOWN).all()


def test_mixed_scene_night_west_day_east():
    """Dawn sweeps west->east: night pixels use BTD, day pixels use the visible test."""
    g, _, m = day_scene()
    sun = np.where(np.arange(len(LONS))[None, :] < 170, -10.0, 30.0) * np.ones((len(LATS), 1), np.float32)
    t039 = g["t039"].copy()
    night = sun <= C.SUN_ELEV_NIGHT_DEG
    t039[night] = g["t108"][night] - 0.5                 # clear night land
    t039[night & m["fog"]] = g["t108"][night & m["fog"]] - 5.0   # night fog: BTD 5 K
    out, info = F.classify(t039, g["t108"], sun, return_info=True, vis06=g["vis06"], nir16=g["nir16"])
    assert (out[65:115, 85:165] == C.FOG).mean() > 0.98, "night half of the sheet"
    assert (out[65:115, 175:195] == C.FOG).mean() > 0.98, "day half of the sheet"
    assert 0 < info["day_share"] < 1 and 0 < info["night_share"] < 1


def test_daytime_scene_end_to_end(tmp_path, monkeypatch):
    """A 10:30 IST scan runs through the real worker (no day skip) and the backend sees fog."""
    import run as worker
    g, _, _ = day_scene()
    fake = types.ModuleType("fetch")
    fake.collection = lambda: object()
    fake.nearest_product = lambda coll, when: when
    fake.newest_product = lambda coll, when: when
    fake.scan_time = lambda p: p

    def download_native(product, wd):
        p = Path(wd) / "s.nat"
        p.write_bytes(b"x")
        return p
    fake.download_native = download_native

    def to_grid(nat):
        # real sun elevation over the grid at that time, so the scene is consistent
        sun = F.sun_elevation(datetime(2025, 1, 4, 5, 0), LATS, LONS)
        s = np.sin(np.radians(np.clip(sun, 1, 90)))
        return {**g, "vis06": g["vis06"] / np.sin(np.radians(35.0)) * s,
                "nir16": g["nir16"] / np.sin(np.radians(35.0)) * s}
    fake.to_grid = to_grid
    monkeypatch.setitem(sys.modules, "fetch", fake)
    args = types.SimpleNamespace(local=str(tmp_path), time="2025-01-04T05:00", quicklook=None, force=False)
    assert worker.process(args) == 0
    man = json.loads((tmp_path / C.MANIFEST).read_text())
    sc = man["scenes"][0]
    assert man["status"] == "ok" and sc["day_share"] > 0.9 and sc["fog_share"] > 0.05
    z = np.load(tmp_path / sc["path"])
    assert "vis06" in z.files and (z["fog"] == C.FOG).sum() > 1000
