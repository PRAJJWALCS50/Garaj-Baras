"""
Get one Meteosat-9 scene from the EUMETSAT Data Store and turn it into the two
channels we need on our North India grid.

    newest_product(time)  -> the scan nearest/just before `time`
    download_native()     -> the scene's .nat file (whole Earth disk, large)
    to_grid()             -> (t039, t108) float32 arrays on config.GRID
Needs EUMETSAT_KEY / EUMETSAT_SECRET (environment variables).
"""
import os
import shutil
import zipfile
from datetime import timedelta
from pathlib import Path

import numpy as np

import config as C


def collection():
    import eumdac
    key, secret = os.environ.get("EUMETSAT_KEY"), os.environ.get("EUMETSAT_SECRET")
    if not key or not secret:
        raise RuntimeError("EUMETSAT_KEY / EUMETSAT_SECRET not set")
    token = eumdac.AccessToken((key, secret))
    return eumdac.DataStore(token).get_collection(C.EUMETSAT_COLLECTION)


def _start(p):
    return p.sensing_start.replace(tzinfo=None)


def newest_product(coll, when, back_min=C.SEARCH_BACK_MIN):
    """Latest scan that started within `back_min` minutes before `when` (UTC)."""
    prods = list(coll.search(dtstart=when - timedelta(minutes=back_min), dtend=when))
    return max(prods, key=_start) if prods else None


def nearest_product(coll, when, window_min=12):
    """Scan nearest to `when` (for historical tests)."""
    prods = list(coll.search(dtstart=when - timedelta(minutes=window_min),
                             dtend=when + timedelta(minutes=window_min)))
    return min(prods, key=lambda p: abs((_start(p) - when).total_seconds())) if prods else None


def scan_time(product):
    return _start(product)


def download_native(product, workdir: Path) -> Path:
    """Fetch only the .nat entry when the API allows it, else the zip."""
    try:
        nat = [e for e in product.entries if e.lower().endswith(".nat")]
    except Exception:
        nat = []
    if nat:
        target = workdir / Path(nat[0]).name
        with product.open(entry=nat[0]) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        return target
    zpath = workdir / "product.zip"
    with product.open() as src, open(zpath, "wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".nat"))
        z.extract(name, workdir)
    zpath.unlink()
    return workdir / name


def area_definition(grid=C.GRID):
    from pyresample import create_area_def
    from fogmask import grid_axes
    lats, lons = grid_axes(grid)
    h = grid["res"] / 2
    return create_area_def(
        "igp", {"proj": "longlat", "datum": "WGS84"},
        area_extent=[grid["lon_min"] - h, grid["lat_min"] - h, grid["lon_max"] + h, grid["lat_max"] + h],
        shape=(len(lats), len(lons)),
    )


def to_grid(nat_path: Path):
    """Calibrated brightness temperatures (K) of 3.9 and 10.8 um on our grid."""
    from satpy import Scene
    scn = Scene(filenames=[str(nat_path)], reader=C.SATPY_READER)
    scn.load([C.CH_039, C.CH_108])
    res = scn.resample(area_definition(), resampler="nearest", radius_of_influence=C.RESAMPLE_RADIUS_M)
    t039 = np.asarray(res[C.CH_039].values, dtype=np.float32)
    t108 = np.asarray(res[C.CH_108].values, dtype=np.float32)
    return t039, t108
