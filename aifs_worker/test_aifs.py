"""
Offline tests: python -m pytest aifs_worker/test_aifs.py
No network, no GRIB libraries - fetch is replaced by a fake module.
"""
import importlib
import json
import os
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C   # noqa: E402
import fogrule as R  # noqa: E402

LATS = np.arange(33.0, 22.99, -0.25, dtype=np.float32)
LONS = np.arange(72.0, 89.01, 0.25, dtype=np.float32)
DELHI, JAIPUR, LUCKNOW = (28.61, 77.21), (26.91, 75.79), (26.85, 80.95)


def test_rule_basic_cases():
    dpd = np.array([0.5, 0.5, 1.2, 1.8, 1.8, 3.0])
    wind = np.array([2.0, 4.0, 3.5, 3.5, 6.0, 1.0])
    lcc = np.array([0.0, 0.0, 0.9, 0.0, 0.0, 1.0])
    assert R.classify(dpd, wind, lcc).tolist() == [C.FOG, C.POSSIBLE, C.FOG, C.POSSIBLE, C.CLEAR, C.CLEAR]


def test_ingredients_units():
    dpd, wind, lcc = R.ingredients([280.0], [279.4], [3.0], [4.0], [85.0])
    assert abs(dpd[0] - 0.6) < 1e-4 and abs(wind[0] - 5.0) < 1e-6 and abs(lcc[0] - 0.85) < 1e-6
    dpd, _, _ = R.ingredients([280.0], [280.3], [0], [0], [0])      # supersaturated rounding
    assert dpd[0] == 0.0


def fake_fields(run):
    """Fog belt over the plain (Delhi->Lucknow) at night steps, dry desert at Jaipur."""
    n = len(C.STEPS)
    shape = (n, len(LATS), len(LONS))
    t2 = np.full(shape, 285.0, np.float32)
    d2 = t2 - 6.0
    u = np.full(shape, 2.0, np.float32); v = np.zeros(shape, np.float32)
    lcc = np.zeros(shape, np.float32)
    belt = ((LATS[:, None] > 26.0) & (LATS[:, None] < 29.5) & (LONS[None, :] > 76.8) & (LONS[None, :] < 82.0))
    for i, s in enumerate(C.STEPS):
        hour_ist = (run + timedelta(hours=s, minutes=330)).hour
        if hour_ist < 9 or hour_ist >= 22:            # night / early morning
            d2[i][belt] = t2[i][belt] - 0.4
            lcc[i][belt] = 95.0
    return {"2t": t2, "2d": d2, "10u": u, "10v": v, "lcc": lcc}


@pytest.fixture
def worker_out(tmp_path, monkeypatch):
    import run as worker
    run = datetime(2026, 11, 26, 18)
    fake = types.ModuleType("fetch")
    fake.newest_run = lambda: (run, "ecmwf")

    def download(workdir, r, src=None):
        p = Path(workdir) / "a.grib2"
        p.write_bytes(b"x" * 100)
        return p, "ecmwf"
    fake.download = download
    fake.read_box = lambda grib: (LATS, LONS, fake_fields(run))
    monkeypatch.setitem(sys.modules, "fetch", fake)
    args = types.SimpleNamespace(local=str(tmp_path), run=None, quicklook=None, force=False)
    assert worker.process(args) == 0
    assert worker.process(args) == 0           # second call: already processed
    return tmp_path, run


def test_worker_writes_manifest_and_run(worker_out):
    folder, run = worker_out
    man = json.loads((folder / C.MANIFEST).read_text())
    assert man["run"] == "2026-11-26T18:00Z" and man["rule"] == C.RULE
    assert len(man["steps"]) == len(C.STEPS)
    z = np.load(folder / man["path"])
    assert z["fog"].shape == (len(C.STEPS), len(LATS), len(LONS))
    # +12 h = 06 UTC = 11:30 IST: daytime, belt dry -> no fog; +6 h = 05:30 IST: fog
    by = {s["step_h"]: s for s in man["steps"]}
    assert by[6]["fog_share"] > 0.02 and by[18]["fog_share"] == 0


def backend_with(folder, monkeypatch, now):
    monkeypatch.setenv("FOG_SAT_BASE_URL", f"file://{folder}")
    sys.path.insert(0, str(HERE.parent / "backend"))
    import fog_aifs
    importlib.reload(fog_aifs)
    fog_aifs._now = lambda: now
    return fog_aifs


def test_backend_route_forecast(worker_out, monkeypatch):
    folder, run = worker_out
    # leave Delhi at 03:00 IST (21:30 UTC): Delhi now, Lucknow ~7 h later (10:00 IST)
    fa = backend_with(folder, monkeypatch, datetime(2026, 11, 26, 21, 30, tzinfo=timezone.utc))
    out = fa.route_fog([
        {"lat": DELHI[0], "lon": DELHI[1], "eta_mins": 0},
        {"lat": JAIPUR[0], "lon": JAIPUR[1], "eta_mins": 60},
        {"lat": LUCKNOW[0], "lon": LUCKNOW[1], "eta_mins": 120},
        {"lat": LUCKNOW[0], "lon": LUCKNOW[1], "eta_mins": 420},
        {"lat": 19.0, "lon": 72.8, "eta_mins": 10},                 # Mumbai: outside
        {"lat": DELHI[0], "lon": DELHI[1], "eta_mins": 60 * 80},   # beyond +72 h
    ])
    st = [p["status"] for p in out["points"]]
    assert out["run"]["state"] == "ok"
    assert st == ["fog", "clear", "fog", "clear", "outside", "beyond"], st
    assert out["points"][0]["dpd_k"] < 1 and out["points"][0]["valid"].startswith("2026-11-26T21:30")


def test_backend_time_interpolation_between_steps(worker_out, monkeypatch):
    """Halfway between a fog step and a dry step the fields are blended, not jumped."""
    folder, run = worker_out
    fa = backend_with(folder, monkeypatch, datetime(2026, 11, 27, 3, 0, tzinfo=timezone.utc))
    p = fa.route_fog([{"lat": DELHI[0], "lon": DELHI[1], "eta_mins": 0}])["points"][0]
    # 03 UTC: between +6 h (00 UTC, dpd 0.4) and +12 h (06 UTC, dpd 6)
    assert 2.5 < p["dpd_k"] < 4.0 and p["status"] == "clear"


def test_backend_stale_and_no_data(worker_out, monkeypatch, tmp_path_factory):
    folder, run = worker_out
    fa = backend_with(folder, monkeypatch, datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc))
    assert fa.status()["state"] == "stale"            # run 42 h old
    empty = tmp_path_factory.mktemp("empty")
    fa = backend_with(empty, monkeypatch, datetime(2026, 11, 26, 21, 0, tzinfo=timezone.utc))
    out = fa.route_fog([{"lat": DELHI[0], "lon": DELHI[1], "eta_mins": 0}])
    assert out["run"]["state"] == "no_data" and out["points"][0]["status"] == "no_data"
