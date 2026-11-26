"""
Get one Meteosat-9 scene from the EUMETSAT Data Store and turn it into the two
channels we need on our North India grid.

    newest_product(time)  -> the scan nearest/just before `time`
    download_native()     -> the scene's .nat file (whole Earth disk, large)
    to_grid()             -> (t039, t108) float32 arrays on config.GRID
Login (EUMETSAT's current method, EUMDAC guide v6.1, Nov 2026): your EUMETSAT
User Portal username + password -> OAuth2 token. Environment variables:
    EUMETSAT_USERNAME / EUMETSAT_PASSWORD   (preferred; also EUMDAC_USER / EUMDAC_PW)
    EUMETSAT_KEY / EUMETSAT_SECRET          (legacy API key: search works, downloads may 403)
"""
import os
import shutil
import zipfile
from datetime import timedelta
from pathlib import Path

import numpy as np

import config as C


def credentials_present():
    env = os.environ.get
    return bool((env("EUMETSAT_USERNAME") or env("EUMDAC_USER")) and (env("EUMETSAT_PASSWORD") or env("EUMDAC_PW"))) \
        or bool(env("EUMETSAT_KEY") and env("EUMETSAT_SECRET"))


def make_token():
    """OAuth2 token from username/password (current method), else legacy API key."""
    import eumdac
    env = os.environ.get
    user = env("EUMETSAT_USERNAME") or env("EUMDAC_USER")
    pw = env("EUMETSAT_PASSWORD") or env("EUMDAC_PW")
    oauth = getattr(getattr(eumdac, "token", None), "OAuth2Token", None)
    if user and pw and oauth is not None:
        tok = oauth()
        res = tok.set_credentials(eumdac.token.Creds(user=user, pw=pw))
        return res if res is not None else tok
    key, secret = env("EUMETSAT_KEY"), env("EUMETSAT_SECRET")
    if key and secret:
        return eumdac.AccessToken((key, secret))
    if user and pw:
        raise RuntimeError("installed eumdac has no OAuth2Token - run: pip install -U eumdac")
    raise RuntimeError("no EUMETSAT credentials (EUMETSAT_USERNAME/PASSWORD or EUMETSAT_KEY/SECRET)")


def collection():
    import eumdac
    return eumdac.DataStore(make_token()).get_collection(C.EUMETSAT_COLLECTION)


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
