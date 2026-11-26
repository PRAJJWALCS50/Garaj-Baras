"""
Satellite fog worker - one run = one scene.

Every ~30 min (GitHub Actions, see .github/workflows/fog_satellite.yml):
  1. find the newest Meteosat-9 scan; already processed? -> stop
  2. download it, keep 0.6 / 1.6 / 3.9 / 10.8 um over North India
  3. fog map: night pixels -> 3.9/10.8 um BTD test, day pixels -> visible test,
     dawn/dusk -> unknown
  4. fog map + distance-to-fog + fog patches tracked against the previous
     scene (born / tracked / merged / split, growth and drift per patch)
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
        "thresholds": {"btd_fog_k_fallback": C.BTD_FOG_K, "btd_method": "per-scene histogram valley",
                       "t108_high_cloud_k": C.T108_HIGH_CLOUD_K,
                       "sun_elev_night_deg": C.SUN_ELEV_NIGHT_DEG,
                       "sun_elev_day_deg": C.SUN_ELEV_DAY_DEG,
                       "vis_fog_refl_fallback": C.VIS_FOG_REFL, "vis_method": "per-scene histogram valley",
                       "t108_low_cloud_k": C.T108_LOW_CLOUD_K},
    }


PATCH_COLS = ("id", "pixels", "area_km2", "lat", "lon", "status", "age",
              "growth_kmh", "drift_e_kmh", "drift_n_kmh")


def load_patches(store, entry):
    """Previous scene's patch-id map and patch table, or (None, None)."""
    raw = store.get(entry["path"])
    if raw is None:
        return None, None
    z = S.unpack(raw)
    if "patch_id" not in z:
        return None, None
    return z["patch_id"].astype(np.int32), {c: z[f"p_{c}"] for c in PATCH_COLS}


def process(args):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    store = make_store(args)
    manifest = S.read_manifest(store) or {}
    lats, lons = F.grid_axes()
    target = parse_iso(args.time) if args.time else now

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
        g = fetch.to_grid(nat)
    t039, t108 = g["t039"], g["t108"]

    # 4. products
    fog, thr_info = F.classify(t039, t108, F.sun_elevation(t, lats, lons), return_info=True,
                               vis06=g.get("vis06"), nir16=g.get("nir16"))
    dist = F.distance_km(fog)
    labels, props = F.label_patches(fog, lats, lons)
    prev = sorted((s for s in old if parse_iso(s["time"]) < t), key=lambda s: s["time"], reverse=True)
    prev_ids, prev_table, gap = None, None, None
    if prev:
        prev_ids, prev_table = load_patches(store, prev[0])
        gap = (t - parse_iso(prev[0]["time"])).total_seconds() / 60
        # around dawn/dusk the method changes under the patches -> do not
        # turn that into fake growth; restart tracking instead
        if abs(thr_info["day_share"] - prev[0].get("day_share", thr_info["day_share"])) > C.DAYNIGHT_SWITCH_MAX:
            log("day/night method switch since previous scene - tracking restarted")
            prev_ids, prev_table, gap = None, None, None
    ids, table, next_id = F.track_patches(labels, props, prev_ids, prev_table, gap,
                                          int(manifest.get("next_patch_id", 1)))
    stats = {**F.scene_stats(fog), **thr_info}
    counts = {k: int((table["status"] == v).sum()) for k, v in F.STATUS.items()}
    stats["patches"] = int(len(table["id"]))
    log(f"scene {iso(t)}: {stats}, patch status {counts}")

    # 5. upload
    path = f"{C.SCENE_DIR}/{t:%Y%m%d_%H%M}.npz"
    store.put(path, S.pack({
        "fog": fog,
        "dist_km": dist.astype(np.float16),
        "patch_id": ids,
        **{f"p_{c}": table[c] for c in PATCH_COLS},
        "btd": (t108 - t039).astype(np.float16),
        "t108": t108.astype(np.float16),
        **({"vis06": F.sun_corrected(g["vis06"], F.sun_elevation(t, lats, lons)).astype(np.float16)}
           if g.get("vis06") is not None and thr_info["day_share"] > 0 else {}),
    }))
    entry = {"time": iso(t), "path": path, "raw_mb": round(size_mb, 1), **stats}
    scenes = sorted([entry] + [s for s in old if s["time"] != entry["time"]],
                    key=lambda s: s["time"], reverse=True)[:3]
    S.write_manifest(store, {**manifest_base(now), "status": "ok", "scenes": scenes,
                             "next_patch_id": next_id})
    log(f"uploaded {path} and {C.MANIFEST}")

    if args.quicklook:
        vis = (F.sun_corrected(g["vis06"], F.sun_elevation(t, lats, lons))
               if g.get("vis06") is not None and thr_info["day_share"] > 0.5 else None)
        quicklook(fog, t108 - t039, iso(t), args.quicklook, vis=vis)
        log(f"quicklook saved to {args.quicklook}")
    return 0


def quicklook(fog, btd, title, png, vis=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    g = C.GRID
    ext = [g["lon_min"], g["lon_max"], g["lat_min"], g["lat_max"]]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    if vis is not None:
        im = ax[0].imshow(np.clip(vis.astype(float), 0, 1), extent=ext, cmap="gray", vmin=0, vmax=0.8)
        ax[0].set_title("visible reflectance (sun-corrected)  white = fog/cloud")
    else:
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

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch import credentials_present
    missing = [] if credentials_present() else ["EUMETSAT_USERNAME/EUMETSAT_PASSWORD"]
    if not args.local:
        missing += [k for k in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY") if not os.environ.get(k)]
    if missing:
        log(f"not configured, missing: {', '.join(missing)}")
        # scheduled runs stay green until the secrets are added
        sys.exit(0 if os.environ.get("GITHUB_ACTIONS") else 2)
    sys.exit(process(args))


if __name__ == "__main__":
    main()
