"""
Pure-NumPy fog products (no satellite libraries needed -> unit-testable).

    classify()     T3.9, T10.8, sun elevation  -> fog map  (0 clear, 1 fog, 2 unknown)
                   night: BTD test; day: visible-reflectance test (VIS0.6, NIR1.6)
    distance_km()  fog map                     -> km to the nearest fog pixel
    label_patches() fog map                    -> separate fog patches (touching pixels)
    track_patches() patches now vs previous    -> persistent id, born/tracked/merged/split,
                                                  growth (km/h) and drift (km/h) per patch
"""
import math
from datetime import datetime

import numpy as np
from scipy import ndimage

import config as C


def grid_axes(grid=C.GRID):
    """Pixel-centre latitudes (north -> south) and longitudes (west -> east)."""
    lats = np.round(np.arange(grid["lat_max"], grid["lat_min"] - 1e-9, -grid["res"]), 6)
    lons = np.round(np.arange(grid["lon_min"], grid["lon_max"] + 1e-9, grid["res"]), 6)
    return lats, lons


def sun_elevation(when: datetime, lats, lons):
    """Solar elevation (deg) for every pixel, NOAA approximation. `when` is UTC."""
    doy = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60 + when.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * np.asarray(lons)[None, :]        # true solar time, min
    ha = np.radians(tst / 4 - 180)
    lat = np.radians(np.asarray(lats))[:, None]
    cos_zen = np.sin(lat) * math.sin(decl) + np.cos(lat) * math.cos(decl) * np.cos(ha)
    return 90 - np.degrees(np.arccos(np.clip(cos_zen, -1, 1)))


def _valley(vals, edges, search, clamp, fallback):
    """Valley between the clear hump and the fog hump of a histogram, or fallback."""
    vals = vals[np.isfinite(vals)]
    if vals.size < 1000:
        return fallback, "fallback"
    hist, _ = np.histogram(vals, bins=edges)
    smooth = np.convolve(hist, np.ones(3) / 3, mode="same")
    centres = (edges[:-1] + edges[1:]) / 2
    inside = (centres >= search[0]) & (centres <= search[1])
    k = np.flatnonzero(inside)[np.argmin(smooth[inside])]
    left, right = smooth[centres < centres[k]].max(initial=0), smooth[centres > centres[k]].max(initial=0)
    # a real valley needs a clear hump on the left AND a fog hump on the right
    if smooth[k] < 0.8 * min(left, right):
        return float(np.clip(centres[k], *clamp)), "valley"
    return fallback, "fallback"


def fog_threshold(btd, usable):
    """
    Night: scene-specific BTD threshold (dynamic threshold). Falls back to BTD_FOG_K.
    Returns (threshold, how) with how in {"valley", "fallback"}.
    """
    return _valley(btd[usable], np.arange(0.0, 8.0 + 1e-9, 0.25),
                   C.BTD_SEARCH_K, C.BTD_CLAMP_K, C.BTD_FOG_K)


def vis_threshold(refl, usable):
    """Day: scene-specific reflectance threshold. Falls back to VIS_FOG_REFL."""
    return _valley(refl[usable], np.arange(0.0, 1.0 + 1e-9, 0.025),
                   C.VIS_SEARCH, C.VIS_CLAMP, C.VIS_FOG_REFL)


def local_std(a):
    """3x3 standard deviation (NaN-safe: NaN -> large)."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=10.0)
    m = ndimage.uniform_filter(a, 3, mode="nearest")
    m2 = ndimage.uniform_filter(a * a, 3, mode="nearest")
    return np.sqrt(np.maximum(m2 - m * m, 0))


def sun_corrected(vis06, sun_elev):
    """Visible reflectance (satpy % or fraction) -> fraction, divided by sin(elevation)."""
    v = np.asarray(vis06, dtype=np.float32)
    if np.nanmax(v, initial=0) > 2.0:            # satpy gives percent
        v = v / 100.0
    s = np.sin(np.radians(np.clip(np.asarray(sun_elev, dtype=np.float32), 1.0, 90.0)))
    return v / s


def clean_mask(fog):
    """Remove isolated fog specks and fill small holes (8-neighbour counts)."""
    n = ndimage.convolve(fog.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - fog
    out = fog.copy()
    out[fog & (n < C.CLEAN_MIN_NEIGHBOURS)] = False
    out[~fog & (n >= C.FILL_MIN_NEIGHBOURS)] = True
    return out


def classify(t039, t108, sun_elev, return_info=False, vis06=None, nir16=None):
    """
    Per-pixel fog map. Unknown wins over everything we cannot trust.
      night (sun <= SUN_ELEV_NIGHT_DEG): BTD = T10.8 - T3.9 above the scene valley
      day   (sun >= SUN_ELEV_DAY_DEG):   bright, smooth, warm-topped, not snow
      dawn/dusk in between, or day without the visible channel: unknown
    """
    t039 = np.asarray(t039, dtype=np.float32)
    t108 = np.asarray(t108, dtype=np.float32)
    sun = np.broadcast_to(np.asarray(sun_elev, dtype=np.float32), t108.shape)
    btd = t108 - t039
    night = sun <= C.SUN_ELEV_NIGHT_DEG
    day = sun >= C.SUN_ELEV_DAY_DEG
    warm = t108 >= C.T108_HIGH_CLOUD_K
    out = np.full(t108.shape, C.CLEAR, dtype=np.uint8)
    info = {"day_share": round(float(day.mean()), 3), "night_share": round(float(night.mean()), 3)}

    # night method
    thr, how = fog_threshold(btd, night & warm)
    fog_n = clean_mask(np.nan_to_num(btd, nan=-99.0) > thr) & night
    info.update(btd_threshold_k=round(thr, 2), threshold_method=how)
    blind = ~night | ~np.isfinite(btd)

    # day method
    fog_d = np.zeros(t108.shape, bool)
    if vis06 is not None and day.any():
        refl = sun_corrected(vis06, sun)
        vthr, vhow = vis_threshold(refl, day & warm)
        bright = np.nan_to_num(refl, nan=0.0) > vthr
        snow = np.zeros(t108.shape, bool)
        if nir16 is not None:
            nir = sun_corrected(nir16, sun)
            ndsi = (refl - nir) / np.maximum(refl + nir, 1e-3)
            snow = np.nan_to_num(ndsi, nan=0.0) > C.SNOW_NDSI_MIN
        low = t108 >= C.T108_LOW_CLOUD_K
        smooth = local_std(refl) <= C.VIS_STD_MAX
        fog_d = clean_mask(bright & low & smooth & ~snow) & day
        # bright cloud that is not a fog sheet (cold or lumpy) hides the ground
        cloud_top = bright & ~snow & ~fog_d
        day_blind = day & (cloud_top | ~np.isfinite(refl))
        blind = (blind & ~day) | day_blind
        info.update(vis_threshold=round(vthr, 3), vis_method=vhow)

    out[fog_n | fog_d] = C.FOG
    blind = blind | ~np.isfinite(t108) | (t108 < C.T108_HIGH_CLOUD_K)
    out[blind] = C.UNKNOWN
    if return_info:
        return out, info
    return out


def pixel_km(grid=C.GRID):
    """(north-south, east-west) pixel size in km at the grid's mid latitude."""
    mid = math.radians((grid["lat_min"] + grid["lat_max"]) / 2)
    ns = grid["res"] * 111.32
    return ns, ns * math.cos(mid)


def distance_km(fogmap, grid=C.GRID):
    """km from every pixel to the nearest FOG pixel (0 on fog), capped."""
    fog = fogmap == C.FOG
    if not fog.any():
        return np.full(fogmap.shape, C.DIST_MAX_KM, dtype=np.float32)
    d = ndimage.distance_transform_edt(~fog, sampling=pixel_km(grid))
    return np.minimum(d, C.DIST_MAX_KM).astype(np.float32)


# ── Fog patches ─────────────────────────────────────────────────────────────
STATUS = {"born": 0, "tracked": 1, "merged": 2, "split": 3, "untracked": 4}
_EIGHT = np.ones((3, 3), dtype=bool)          # 8-neighbour connectivity
_KM_PER_DEG = 111.32


def label_patches(fogmap, lats, lons, grid=C.GRID):
    """
    Touching fog pixels -> patches. Returns (labels, props):
      labels : int32 map, 1..n for tracked-size patches, 0 elsewhere
      props  : dict of arrays (index k = label k+1): pixels, area_km2, lat, lon
    Specks smaller than PATCH_MIN_PIXELS get label 0 (they still count as fog
    in the fog map, they are just not tracked).
    """
    raw, n = ndimage.label(fogmap == C.FOG, structure=_EIGHT)
    empty = {k: np.zeros(0) for k in ("pixels", "area_km2", "lat", "lon")}
    if n == 0:
        return np.zeros(fogmap.shape, np.int32), empty
    idx = np.arange(1, n + 1)
    pixels = ndimage.sum_labels(np.ones_like(raw), raw, idx)
    keep = idx[pixels >= C.PATCH_MIN_PIXELS]
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    labels = remap[raw]
    if not len(keep):
        return labels, empty
    k = np.arange(1, len(keep) + 1)
    row_area = (grid["res"] * _KM_PER_DEG) ** 2 * np.cos(np.radians(lats))       # km2 per pixel, per row
    area_map = np.broadcast_to(row_area[:, None], fogmap.shape)
    lat_map = np.broadcast_to(np.asarray(lats)[:, None], fogmap.shape)
    lon_map = np.broadcast_to(np.asarray(lons)[None, :], fogmap.shape)
    props = {
        "pixels": ndimage.sum_labels(np.ones_like(labels), labels, k),
        "area_km2": ndimage.sum_labels(area_map, labels, k),
        "lat": ndimage.mean(lat_map, labels, k),
        "lon": ndimage.mean(lon_map, labels, k),
    }
    return labels, props


def _motion(area_new, lat_new, lon_new, area_old, lat_old, lon_old, gap_h):
    """Growth (km/h of equivalent-radius change) and drift (km/h east, north)."""
    growth = (math.sqrt(area_new / math.pi) - math.sqrt(area_old / math.pi)) / gap_h
    de = (lon_new - lon_old) * _KM_PER_DEG * math.cos(math.radians((lat_new + lat_old) / 2)) / gap_h
    dn = (lat_new - lat_old) * _KM_PER_DEG / gap_h
    if abs(growth) > C.PATCH_GROWTH_MAX_KMH:
        growth = math.nan
    if math.hypot(de, dn) > C.PATCH_DRIFT_MAX_KMH:
        de = dn = math.nan
    return growth, de, dn


def _smooth(new, old):
    if math.isfinite(new) and math.isfinite(old):
        return C.PATCH_SMOOTH * new + (1 - C.PATCH_SMOOTH) * old
    return new


def track_patches(labels, props, prev_ids=None, prev_table=None, gap_min=None, next_id=1):
    """
    Give each patch a persistent id and a motion by matching it with the
    previous scene's patches through pixel OVERLAP.

    prev_ids   : previous scene's patch-id map (persistent ids, 0 = none) or None
    prev_table : previous scene's patch table (dict of arrays) or None
    Returns (ids_map, table, next_id). Table columns: id, pixels, area_km2,
    lat, lon, status, age, growth_kmh, drift_e_kmh, drift_n_kmh.
    A NEW patch (born) never gets a speed: forming fog is not moving fog.
    """
    n = len(props["pixels"])
    table = {
        "id": np.zeros(n, np.int32), "pixels": np.asarray(props["pixels"], np.int32),
        "area_km2": np.asarray(props["area_km2"], np.float32),
        "lat": np.asarray(props["lat"], np.float32), "lon": np.asarray(props["lon"], np.float32),
        "status": np.full(n, STATUS["untracked"], np.int8), "age": np.zeros(n, np.int16),
        "growth_kmh": np.full(n, np.nan, np.float32),
        "drift_e_kmh": np.full(n, np.nan, np.float32), "drift_n_kmh": np.full(n, np.nan, np.float32),
    }
    usable = (prev_ids is not None and prev_table is not None and gap_min
              and 0 < gap_min <= C.PATCH_MAX_GAP_MIN)
    parents = {j: [] for j in range(n)}
    children = {}
    if usable:
        prow = {int(pid): i for i, pid in enumerate(prev_table["id"])}
        both = (labels > 0) & (prev_ids > 0)
        pairs, counts = np.unique(np.stack([labels[both], prev_ids[both]]), axis=1, return_counts=True)
        for (lab, pid), cnt in zip(pairs.T, counts):
            j, i = int(lab) - 1, prow.get(int(pid))
            if i is None:
                continue
            small = min(table["pixels"][j], int(prev_table["pixels"][i]))
            if cnt / max(small, 1) >= C.PATCH_OVERLAP_MIN:
                parents[j].append(i)
                children.setdefault(i, []).append(j)

    gap_h = (gap_min or 0) / 60.0
    used = set()
    # biggest patches first, so a split parent's id goes to its largest child
    for j in np.argsort(-table["pixels"]):
        par = parents[j]
        if not usable:
            status, pid = "untracked", None
        elif not par:
            status, pid = "born", None
        elif len(par) > 1:
            status = "merged"
            pid = int(prev_table["id"][max(par, key=lambda i: prev_table["pixels"][i])])
        elif len(children[par[0]]) > 1:
            status, pid = "split", int(prev_table["id"][par[0]])
        else:
            status, pid = "tracked", int(prev_table["id"][par[0]])
        if pid is None or pid in used:
            pid, next_id = next_id, next_id + 1
        used.add(pid)
        table["id"][j], table["status"][j] = pid, STATUS[status]
        if par:
            table["age"][j] = int(max(prev_table["age"][i] for i in par)) + 1
        ambiguous = status == "split" or any(len(children[i]) > 1 for i in par)
        if par and not ambiguous:
            a_old = float(sum(prev_table["area_km2"][i] for i in par))
            w = [float(prev_table["area_km2"][i]) / a_old for i in par]
            lat_old = sum(wi * float(prev_table["lat"][i]) for wi, i in zip(w, par))
            lon_old = sum(wi * float(prev_table["lon"][i]) for wi, i in zip(w, par))
            g, de, dn = _motion(float(table["area_km2"][j]), float(table["lat"][j]), float(table["lon"][j]),
                                a_old, lat_old, lon_old, gap_h)
            if status == "tracked":
                i = par[0]
                g = _smooth(g, float(prev_table["growth_kmh"][i]))
                de = _smooth(de, float(prev_table["drift_e_kmh"][i]))
                dn = _smooth(dn, float(prev_table["drift_n_kmh"][i]))
            table["growth_kmh"][j], table["drift_e_kmh"][j], table["drift_n_kmh"][j] = g, de, dn

    ids_map = np.zeros(labels.shape, np.int32)
    if n:
        lut = np.concatenate([[0], table["id"]]).astype(np.int32)
        ids_map = lut[labels]
    return ids_map, table, next_id


def scene_stats(fogmap):
    total = fogmap.size
    return {
        "fog_share": round(float((fogmap == C.FOG).sum() / total), 4),
        "unknown_share": round(float((fogmap == C.UNKNOWN).sum() / total), 4),
        "clear_share": round(float((fogmap == C.CLEAR).sum() / total), 4),
    }
