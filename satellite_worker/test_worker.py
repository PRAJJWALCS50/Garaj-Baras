"""
Offline tests for the satellite fog pipeline (no EUMETSAT, no satpy, no Supabase).

    python -m pytest satellite_worker/test_worker.py -q

The end-to-end test fakes the satellite: a fog blanket centred on the Ganga at
Garhmukteshwar grows over three 30-min scenes. The REAL worker code turns those
scenes into fog maps / distance / edge speed in a local folder, and the REAL
backend module (backend/fog_satellite.py) answers a Moradabad -> Delhi route.
"""
import importlib
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
    t108 = np.array([[285, 285, 240, 285]], dtype=np.float32)
    t039 = np.array([[281, 284.5, 236, np.nan]], dtype=np.float32)
    sun = np.full(t108.shape, -50.0)
    out = F.classify(t039, t108, sun)
    # BTD 4 -> fog, BTD 0.5 -> clear, cold top -> unknown, missing -> unknown
    assert out.tolist() == [[C.FOG, C.CLEAR, C.UNKNOWN, C.UNKNOWN]]
    assert (F.classify(t039, t108, np.full(t108.shape, 10.0)) == C.UNKNOWN).all()


def test_distance_and_speed():
    fog = np.zeros((50, 50), np.uint8)
    fog[25, 10] = C.FOG
    d = F.distance_km(fog)
    ns, ew = F.pixel_km()
    assert d[25, 10] == 0 and abs(d[25, 20] - 10 * ew) < 0.01
    assert (F.distance_km(np.zeros((5, 5), np.uint8)) == C.DIST_MAX_KM).all()
    # edge approaches 8 km per 30 min in both intervals -> 16 km/h
    s = F.edge_speed(np.full((2, 2), 20.0), np.full((2, 2), 28.0), 30, np.full((2, 2), 36.0), 30)
    assert np.allclose(s, 16.0)
    # disagreeing intervals (one noisy frame) -> no speed
    s = F.edge_speed(np.full((2, 2), 20.0), np.full((2, 2), 28.0), 30, np.full((2, 2), 60.0), 30)
    assert np.isnan(s).all()
    # receding fog / no history / too-old history -> no speed
    assert np.isnan(F.edge_speed(np.full((2, 2), 30.0), np.full((2, 2), 20.0), 30)).all()
    assert np.isnan(F.edge_speed(np.full((2, 2), 30.0))).all()
    assert np.isnan(F.edge_speed(np.full((2, 2), 20.0), np.full((2, 2), 28.0), 120)).all()


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
    fake.to_grid = lambda nat: current["arrays"]
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
        print(name, p["status"], p.get("dist_km"), p.get("edge_speed_kmh"), p.get("fog_eta_mins"))

    assert got["Garhmukteshwar"]["status"] == "fog"          # inside the blanket
    assert got["Gajraula"]["status"] in ("approaching", "fog", "patchy")
    assert got["Hapur"]["status"] in ("fog", "patchy", "approaching")
    assert got["Ghaziabad"]["status"] == "unknown"           # high cloud -> never "clear"
    assert got["Delhi"]["status"] == "clear"                 # far from fog
    # fog edge grew 8 km / 30 min in both intervals -> ~16 km/h where measurable
    speeds = [p["edge_speed_kmh"] for p in res["points"] if p.get("edge_speed_kmh")]
    assert speeds and all(10 <= s <= 22 for s in speeds)


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
