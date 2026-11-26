"""
AIFS fog worker - one run = one AIFS forecast.

Hourly (GitHub Actions, .github/workflows/fog_aifs.yml):
  1. find the newest AIFS Single run on ECMWF open data; already done? -> stop
  2. download 2 m temperature + dew point, 10 m wind, low cloud, steps 0..72 h
  3. crop to North India, per step: dew-point depression, wind, low cloud,
     fog class (clear / possible / fog)
  4. upload aifs/run_YYYYMMDDHH.npz (~50 KB) and aifs/latest.json

Local test on your PC (no Supabase needed):
  python aifs_worker/run.py --local aifs_test --quicklook aifs_test/map.png
"""
import argparse
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C          # noqa: E402
import fogrule as R         # noqa: E402
import store as S           # noqa: E402


def log(msg):
    print(f"[aifs-worker {datetime.now(timezone.utc):%H:%M:%S}Z] {msg}", flush=True)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:00Z")


def process(args):
    import fetch
    store = S.LocalStore(args.local) if args.local else S.SupabaseStore()
    manifest = S.read_manifest(store) or {}
    run_req = datetime.strptime(args.run, "%Y-%m-%dT%H") if args.run else None

    run, src = (run_req, None) if run_req else fetch.newest_run()
    if manifest.get("run") == iso(run) and not args.force:
        log(f"run {iso(run)} already processed")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        grib, src = fetch.download(Path(tmp), run, src)
        size_mb = grib.stat().st_size / 1e6
        log(f"downloaded AIFS run {iso(run)} from {src} ({size_mb:.0f} MB)")
        lats, lons, f = fetch.read_box(grib)

    dpd, wind, lcc = R.ingredients(f["2t"], f["2d"], f["10u"], f["10v"], f["lcc"])
    cls = R.classify(dpd, wind, lcc)
    steps = []
    for i, s in enumerate(C.STEPS):
        valid = run + timedelta(hours=s)
        steps.append({"step_h": s, "valid": iso(valid),
                      "fog_share": round(float((cls[i] == C.FOG).mean()), 4),
                      "possible_share": round(float((cls[i] == C.POSSIBLE).mean()), 4)})
    log("fog share per step: " + ", ".join(f"+{s['step_h']}h {s['fog_share']:.2f}" for s in steps))

    path = f"{C.PREFIX}/run_{run:%Y%m%d%H}.npz"
    store.put(path, S.pack({
        "lats": lats, "lons": lons, "steps_h": np.array(C.STEPS, np.int16),
        "dpd": dpd.astype(np.float16), "wind": wind.astype(np.float16),
        "lcc": lcc.astype(np.float16), "fog": cls,
    }))
    S.write_manifest(store, {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "ECMWF AIFS Single (open data, CC-BY-4.0)",
        "run": iso(run), "path": path, "source": src,
        "box": C.BOX, "grid_deg": 0.25, "steps": steps,
        "rule": C.RULE, "classes": {"clear": C.CLEAR, "possible": C.POSSIBLE, "fog": C.FOG},
    })
    log(f"uploaded {path} and {C.MANIFEST}")
    if args.quicklook:
        quicklook(lats, lons, cls, dpd, run, args.quicklook)
        log(f"quicklook saved to {args.quicklook}")
    return 0


def quicklook(lats, lons, cls, dpd, run, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    ext = [lons[0] - .125, lons[-1] + .125, lats[-1] - .125, lats[0] + .125]
    show = [i for i, s in enumerate(C.STEPS) if s <= 30][:6]
    fig, ax = plt.subplots(2, len(show), figsize=(4 * len(show), 6.5))
    towns = {"Delhi": (28.61, 77.21), "Lucknow": (26.85, 80.95), "Patna": (25.61, 85.14),
             "Jaipur": (26.91, 75.79), "Amritsar": (31.63, 74.87), "Varanasi": (25.32, 82.97)}
    for j, i in enumerate(show):
        valid = run + timedelta(hours=C.STEPS[i])
        ist = valid + timedelta(hours=5, minutes=30)
        ax[0, j].imshow(dpd[i].astype(float), extent=ext, cmap="viridis_r", vmin=0, vmax=8)
        ax[0, j].set_title(f"dew-pt depression K\n{ist:%d %b %H:%M} IST", fontsize=9)
        ax[1, j].imshow(cls[i], extent=ext, cmap=ListedColormap(["#e8f5e9", "#b0bec5", "#455a64"]), vmin=0, vmax=2)
        ax[1, j].set_title("clear · possible · FOG", fontsize=9)
        for a in ax[:, j]:
            for n, (la, lo) in towns.items():
                a.plot(lo, la, "k.", ms=3)
                a.annotate(n, (lo, la), fontsize=6)
    fig.suptitle(f"ECMWF AIFS run {iso(run)}")
    fig.tight_layout()
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="AIFS fog worker")
    ap.add_argument("--local", help="store products in this folder instead of Supabase")
    ap.add_argument("--run", help="process this run instead of the newest, e.g. 2026-11-26T00")
    ap.add_argument("--quicklook", help="also save a PNG of the fog forecast")
    ap.add_argument("--force", action="store_true", help="reprocess even if already done")
    args = ap.parse_args()
    if not args.local and not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY")):
        log("not configured, missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(0 if os.environ.get("GITHUB_ACTIONS") else 2)
    sys.exit(process(args))


if __name__ == "__main__":
    main()
