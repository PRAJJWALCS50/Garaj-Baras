"""
Satellite fog worker - one run = one scene.

Every ~30 min (GitHub Actions, see .github/workflows/fog_satellite.yml):
  1. daytime over the whole region?  -> record "day" and stop (method is night-only)
  2. find the newest Meteosat-9 scan; already processed? -> stop
  3. download it, keep only 3.9 / 10.8 um over North India
  4. fog map + distance-to-fog + fog-edge speed (vs the 2 previous scenes)
  5. upload scenes/<time>.npz and update latest.json  (Supabase Storage)

Local test on your PC (no Supabase needed):
  python satellite_worker/run.py --local fog_test --time 2025-01-10T21:00 --quicklook fog_test/map.png
"""
import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C          # noqa: E402
import fogmask as F         # noqa: E402
import store as S           # noqa: E402


def log(msg):
    print(f"[fog-worker {datetime.now(timezone.utc):%H:%M:%S}Z] {msg}", flush=True)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:00Z")


def parse_iso(s):
    return datetime.strptime(s[:16], "%Y-%m-%dT%H:%M")


def make_store(args):
    if args.local:
        return S.LocalStore(args.local)
    return S.SupabaseStore()


def manifest_base(now):
    return {
        "updated": iso(now),
        "satellite": "Meteosat-9 IODC (SEVIRI)",
        "grid": C.GRID,
        "classes": {"clear": C.CLEAR, "fog": C.FOG, "unknown": C.UNKNOWN},
        "thresholds": {"btd_fog_k": C.BTD_FOG_K, "t108_high_cloud_k": C.T108_HIGH_CLOUD_K,
                       "sun_elev_night_deg": C.SUN_ELEV_NIGHT_DEG},
    }


def load_dist(store, entry):
    raw = store.get(entry["path"])
    return None if raw is None else S.unpack(raw)["dist_km"].astype(np.float32)


def process(args):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    store = make_store(args)
    manifest = S.read_manifest(store) or {}
    lats, lons = F.grid_axes()
    target = parse_iso(args.time) if args.time else now

    # 1. whole region in daylight? nothing to see with the night method
    if not args.time and F.sun_elevation(target, lats[::20], lons[::20]).min() > C.SUN_ELEV_NIGHT_DEG:
        if manifest.get("status") != "day":
            S.write_manifest(store, {**manifest_base(now), "status": "day", "scenes": []})
        log("daytime over the whole region - night fog method not applicable, skipped")
        return 0

    # 2. newest (or requested) scan
    import fetch
    coll = fetch.collection()
    product = fetch.nearest_product(coll, target) if args.time else fetch.newest_product(coll, target)
    if product is None:
        log("no Meteosat-9 scan found in the search window")
        return 1
    t = fetch.scan_time(product).replace(second=0, microsecond=0)
    old = manifest.get("scenes") or []
    if any(s["time"] == iso(t) for s in old) and not args.force:
        log(f"scene {iso(t)} already processed")
        return 0

    # 3. download + crop
    with tempfile.TemporaryDirectory() as tmp:
        nat = fetch.download_native(product, Path(tmp))
        size_mb = nat.stat().st_size / 1e6
        log(f"downloaded {nat.name} ({size_mb:.0f} MB)")
        t039, t108 = fetch.to_grid(nat)

    # 4. products
    fog = F.classify(t039, t108, F.sun_elevation(t, lats, lons))
    dist = F.distance_km(fog)
    prev = sorted((s for s in old if parse_iso(s["time"]) < t), key=lambda s: s["time"], reverse=True)
    d1 = load_dist(store, prev[0]) if len(prev) > 0 else None
    d2 = load_dist(store, prev[1]) if len(prev) > 1 else None
    gap1 = (t - parse_iso(prev[0]["time"])).total_seconds() / 60 if d1 is not None else None
    gap2 = ((parse_iso(prev[0]["time"]) - parse_iso(prev[1]["time"])).total_seconds() / 60
            if d1 is not None and d2 is not None else None)
    speed = F.edge_speed(dist, d1, gap1, d2, gap2)
    stats = F.scene_stats(fog)
    log(f"scene {iso(t)}: {stats}, speed pixels={int(np.isfinite(speed).sum())}")

    # 5. upload
    path = f"{C.SCENE_DIR}/{t:%Y%m%d_%H%M}.npz"
    store.put(path, S.pack({
        "fog": fog,
        "dist_km": dist.astype(np.float16),
        "speed_kmh": speed.astype(np.float16),
        "btd": (t108 - t039).astype(np.float16),
        "t108": t108.astype(np.float16),
    }))
    entry = {"time": iso(t), "path": path, "raw_mb": round(size_mb, 1), **stats}
    scenes = sorted([entry] + [s for s in old if s["time"] != entry["time"]],
                    key=lambda s: s["time"], reverse=True)[:3]
    S.write_manifest(store, {**manifest_base(now), "status": "ok", "scenes": scenes})
    log(f"uploaded {path} and {C.MANIFEST}")

    if args.quicklook:
        quicklook(fog, t108 - t039, iso(t), args.quicklook)
        log(f"quicklook saved to {args.quicklook}")
    return 0


def quicklook(fog, btd, title, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    g = C.GRID
    ext = [g["lon_min"], g["lon_max"], g["lat_min"], g["lat_max"]]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    im = ax[0].imshow(btd.astype(float), extent=ext, cmap="RdBu_r", vmin=-6, vmax=6)
    ax[0].set_title("BTD = T10.8 - T3.9 (K)  red = fog/low cloud")
    fig.colorbar(im, ax=ax[0], shrink=0.8)
    ax[1].imshow(fog, extent=ext, cmap=ListedColormap(["#e8f5e9", "#607d8b", "#ffcc80"]), vmin=0, vmax=2)
    ax[1].set_title("fog map: green clear · grey FOG · orange unknown")
    towns = {"Amritsar": (31.63, 74.87), "Delhi": (28.61, 77.21), "Moradabad": (28.84, 78.77),
             "Lucknow": (26.85, 80.95), "Varanasi": (25.32, 82.97), "Patna": (25.61, 85.14),
             "Jaipur": (26.91, 75.79), "Agra": (27.18, 78.01)}
    for a in ax:
        for n, (la, lo) in towns.items():
            a.plot(lo, la, "k.", ms=4)
            a.annotate(n, (lo, la), fontsize=7)
    fig.suptitle(f"Meteosat-9 IODC  {title}")
    fig.tight_layout()
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Satellite fog worker (one scene per run)")
    ap.add_argument("--local", help="store products in this folder instead of Supabase")
    ap.add_argument("--time", help="process the scan nearest this UTC time, e.g. 2025-01-10T21:00")
    ap.add_argument("--quicklook", help="also save a PNG picture of the fog map")
    ap.add_argument("--force", action="store_true", help="reprocess even if already done")
    args = ap.parse_args()

    missing = [k for k in ("EUMETSAT_KEY", "EUMETSAT_SECRET") if not os.environ.get(k)]
    if not args.local:
        missing += [k for k in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY") if not os.environ.get(k)]
    if missing:
        log(f"not configured, missing: {', '.join(missing)}")
        # scheduled runs stay green until the secrets are added
        sys.exit(0 if os.environ.get("GITHUB_ACTIONS") else 2)
    sys.exit(process(args))


if __name__ == "__main__":
    main()
