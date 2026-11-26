"""
Download the newest AIFS Single run (GRIB2, global) and crop it to our box.
Needs: ecmwf-opendata, eccodes (pip wheels include the ecCodes library).
"""
from datetime import datetime
from pathlib import Path

import numpy as np

import config as C


def _client(source):
    from ecmwf.opendata import Client
    return Client(source=source, model=C.MODEL)


def newest_run():
    """Newest AIFS run that has every step we need -> (run_time, source)."""
    last = None
    for src in C.SOURCES:
        try:
            return _client(src).latest(type="fc", stream="oper", step=max(C.STEPS),
                                       param=list(C.PARAMS)), src
        except Exception as e:          # busy server / mirror behind
            last = e
    raise RuntimeError(f"no AIFS run found on any mirror: {last}")


def download(workdir: Path, run: datetime, first_source=None):
    """Download one run (all params/steps) -> (grib_path, source). Tries each mirror."""
    order = [first_source] + [s for s in C.SOURCES if s != first_source] if first_source else list(C.SOURCES)
    target = Path(workdir) / f"aifs_{run:%Y%m%d%H}.grib2"
    last = None
    for src in order:
        try:
            _client(src).retrieve(date=run.strftime("%Y%m%d"), time=run.hour, type="fc", stream="oper",
                                  step=list(C.STEPS), param=list(C.PARAMS), target=str(target))
            return target, src
        except Exception as e:
            last = e
    raise RuntimeError(f"AIFS download failed on all mirrors: {last}")


def read_box(grib: Path, box=C.BOX):
    """
    -> (lats N->S, lons W->E, fields) where fields[param] is (n_steps, ny, nx)
    in the order of C.STEPS.
    """
    import eccodes
    out, axes = {}, None
    with open(grib, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                name = eccodes.codes_get(h, "shortName")
                step = int(eccodes.codes_get(h, "endStep"))
                ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
                lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
                lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
                dlat = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
                dlon = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
                vals = eccodes.codes_get_values(h).reshape(nj, ni)
                missing = eccodes.codes_get(h, "missingValue")
            finally:
                eccodes.codes_release(h)
            name = {"t2m": "2t", "d2m": "2d", "u10": "10u", "v10": "10v"}.get(name, name)
            if name not in C.PARAMS or step not in C.STEPS:
                continue
            lats = lat0 - dlat * np.arange(nj)
            lons = (lon0 + dlon * np.arange(ni) + 180) % 360 - 180
            r = np.flatnonzero((lats <= box["lat_max"] + 1e-6) & (lats >= box["lat_min"] - 1e-6))
            c = np.flatnonzero((lons >= box["lon_min"] - 1e-6) & (lons <= box["lon_max"] + 1e-6))
            c = c[np.argsort(lons[c])]
            sub = vals[np.ix_(r, c)].astype(np.float32)
            sub[sub == missing] = np.nan
            axes = (lats[r].astype(np.float32), lons[c].astype(np.float32))
            out.setdefault(name, {})[step] = sub
    miss = [(p, s) for p in C.PARAMS for s in C.STEPS if s not in out.get(p, {})]
    if miss:
        raise RuntimeError(f"GRIB is missing {len(miss)} fields, e.g. {miss[:3]}")
    return axes[0], axes[1], {p: np.stack([out[p][s] for s in C.STEPS]) for p in C.PARAMS}
