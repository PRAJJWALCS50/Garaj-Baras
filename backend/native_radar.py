"""Decode nonstandard IMD reflectivity products at the ingestion boundary.

Keep existing motion, decay, routes, scenes and alerts on their common palette
and physical pixel scale. Never interpret velocity or accumulated rainfall as
instantaneous reflectivity. No inferred timestamps for these new sources.
"""
import json
import os
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence
from fuzzy import COLOR_TABLE, dbz_to_label
from georef_native import KM_PER_PX
from radar import download_gif, gif_is_fresh, ocr_timestamp_from_image

ROOT = Path(__file__).parent
IST = timezone(timedelta(hours=5, minutes=30))

# Source legend interval midpoints (dBZ), bottom to top, measured from IMD.
SOHRA_DBZ = [10 + (i + .5) * 50 / 31 for i in range(31)]
MBL_DBZ = [4.5, 9.5, 15, 19.5, 22, 25.5, 31, 35.5,
           38, 41.5, 47, 51.5, 54, 57.5, 63, 69]


def _canonical_color(dbz):
    if dbz < 20:
        return (0, 0, 0)
    label = dbz_to_label(dbz)[0]
    # Preserve intensity-category boundaries when quantizing to the legacy
    # engine's ten dBZ levels (e.g. native white ~35 dBZ is NOT 60 dBZ).
    choices = [c for c in COLOR_TABLE if dbz_to_label(c[3])[0] == label]
    return min(choices, key=lambda c: abs(c[3] - dbz))[:3]


def _glyphs(image, box):
    ink = np.array(image.crop(box).convert('L')) < 80
    bounds = np.where(np.diff(np.r_[False, ink.any(axis=0), False]))[0]
    result = []
    for left, right in zip(bounds[::2], bounds[1::2]):
        glyph = ink[:, left:right]
        ys = np.where(glyph.any(axis=1))[0]
        if len(ys):
            glyph = glyph[ys[0]:ys[-1] + 1]
            result.append((np.array(Image.fromarray(glyph).resize((16, 24), Image.Resampling.NEAREST)) > 0))
    return result


def sohra_timestamp(image):
    """Read the explicit date and UTC time; reject ambiguous glyphs."""
    with open(ROOT / 'sohra_digits.json', encoding='utf-8') as f:
        templates = {k: np.array(v, dtype=bool) for k, v in json.load(f).items()}

    def read(box, separators):
        glyphs = _glyphs(image, box)
        chars = []
        for i, glyph in enumerate(glyphs):
            if i in separators:
                chars.append(separators[i])
                continue
            scores = sorted(((float((glyph == tpl).mean()), ch)
                             for ch, tpl in templates.items()), reverse=True)
            if scores[0][0] < .95 or scores[0][0] - scores[1][0] < .03:
                return ''
            chars.append(scores[0][1])
        return ''.join(chars)

    date = read((7, 17, 134, 36), {4: '/', 7: '/'})
    time = read((238, 17, 339, 36), {2: ':', 5: ':'})
    try:
        return datetime.strptime(date + ' ' + time, '%Y/%m/%d %H:%M:%S').replace(
            tzinfo=timezone.utc).astimezone(IST)
    except ValueError:
        return None


def decode_reflectivity(image, station):
    """Return (normalized rain-only image, measured timestamp) or reject."""
    arr = np.array(image.convert('RGB'))
    if station == 'sohra':
        if image.size != (1078, 770):
            raise ValueError('Sohra layout changed')
        colors = [arr[735 - i * 18, 810] for i in range(16)] + [
            arr[735 - i * 18, 950] for i in range(15)]
        # Distinct dark-violet low end + white top of left legend identifies Z.
        if np.linalg.norm(colors[0].astype(float) - (72, 61, 139)) > 8 or np.linalg.norm(
                colors[15].astype(float) - (254, 254, 254)) > 8:
            raise ValueError('Sohra product is not the supported dBZ scale')
        values = SOHRA_DBZ
        box, source_center, source_radius, range_km = (43, 193, 598, 748), 277., 277., 240.
        timestamp = sohra_timestamp(image)
    elif station == 'mahabaleshwar':
        if image.size != (720, 720):
            raise ValueError('Mahabaleshwar layout changed')
        colors = [arr[608 - i * 15, 630] for i in range(16)]
        # MAX_V has cyan at the bottom and crimson at the top. RN/SRI use
        # other scales. Only MAX_Z's blue/cyan/green/pink legend is accepted.
        anchors = {0: (0, 0, 254), 1: (4, 255, 239), 4: (24, 99, 56),
                   15: (255, 200, 255)}
        if any(np.linalg.norm(colors[i].astype(float) - c) > 8 for i, c in anchors.items()):
            raise ValueError('Mahabaleshwar feed is not MAX_Z reflectivity')
        borders = np.where((arr[300:639, 400:530] == 0).all(axis=2).mean(axis=0) > .98)[0]
        if not len(borders):
            raise ValueError('Mahabaleshwar map border not found')
        width = int(borders[0]) + 401
        box = (0, 640 - width, width, 640)
        source_center, source_radius, range_km = (width - 1) / 2, (width - 1) / 2, 170.
        values = MBL_DBZ
        timestamp = ocr_timestamp_from_image(image, (560, 107, 718, 268))
    else:
        raise ValueError('Unknown native station')
    if timestamp is None:
        raise ValueError('Unreadable native radar timestamp')
    if timestamp > datetime.now(IST) + timedelta(minutes=5):
        raise ValueError('Future radar frame')
    crop = arr[box[1]:box[3], box[0]:box[2]]
    # Exact GIF palette membership avoids turning terrain colours, rivers,
    # coastlines and grey no-data into rain. No broad RGB distance heuristic.
    # Sorted packed-RGB lookup avoids 31 full-image comparisons per frame on
    # Render's small CPU allocation, without a 48 MB dense RGB lookup table.
    palette = np.array(colors, dtype=np.uint32)
    keys = (palette[:, 0] << 16) | (palette[:, 1] << 8) | palette[:, 2]
    order = keys.argsort()
    keys = np.r_[keys[order], np.uint32(1 << 24)]
    mapped = np.array([_canonical_color(values[i]) for i in order] + [(0, 0, 0)], dtype=np.uint8)
    pixels = crop.astype(np.uint32)
    packed = (pixels[:, :, 0] << 16) | (pixels[:, :, 1] << 8) | pixels[:, :, 2]
    indices = np.searchsorted(keys, packed)
    normalized = mapped[indices]
    normalized[keys[indices] != packed] = 0
    radius = range_km / KM_PER_PX
    center = int(np.ceil(radius))
    yy, xx = np.mgrid[:2 * center + 1, :2 * center + 1]
    scale = source_radius / radius
    result = cv2.remap(normalized, ((xx - center) * scale + source_center).astype(np.float32),
                       ((yy - center) * scale + source_center).astype(np.float32),
                       cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    result[(xx - center) ** 2 + (yy - center) ** 2 > radius ** 2] = 0
    return Image.fromarray(result), timestamp


class NativeRadarFeed:
    def __init__(self, station, code, current_url):
        self.station = station
        self.gif_url = f'https://mausam.imd.gov.in/Radar/animation/Converted/{code}_MAXZ.gif'
        self.current_url = current_url
        self.folder = ROOT / f'frames_{station}'
        self.animation = ROOT / f'{station}_animation.gif'
        self.current = ROOT / f'{station}_radar.gif'
        self.index = self.folder / 'history.json'
        self.lock = threading.Lock()

    def history(self):
        try:
            records = json.loads(self.index.read_text(encoding='utf-8'))
            return [(str(self.folder / name), datetime.fromisoformat(ts))
                    for name, ts in records if (self.folder / name).exists()]
        except (OSError, ValueError):
            return []

    def extract(self, gif_path, output_folder=None):
        self.folder.mkdir(exist_ok=True)
        frames = {ts: path for path, ts in self.history()}
        accepted = 0
        previous = None
        with Image.open(gif_path) as gif:
            for frame in ImageSequence.Iterator(gif):
                full = frame.convert('RGB')
                content = full.tobytes()
                if content == previous:
                    continue
                previous = content
                try:
                    image, ts = decode_reflectivity(full, self.station)
                except ValueError:
                    continue
                name = f'frame_{ts.strftime("%Y%m%d_%H%M%S")}.png'
                path = self.folder / name
                image.save(path)
                frames[ts] = str(path)
                accepted += 1
        if not accepted:
            print(f'{self.station}: no valid reflectivity frames in {Path(gif_path).name}')
        latest = max(frames, default=None)
        # Six measured observations, <=6h old relative to newest; no fabricated
        # history from padded animation frames or the velocity product.
        keep = sorted((ts, p) for ts, p in frames.items()
                      if latest - ts <= timedelta(hours=6))[-6:]
        records = [(Path(p).name, ts.isoformat()) for ts, p in keep]
        temp = self.index.with_suffix('.tmp')
        temp.write_text(json.dumps(records), encoding='utf-8')
        os.replace(temp, self.index)
        names = {name for name, _ in records}
        for path in self.folder.glob('frame_*.png'):
            if path.name not in names:
                path.unlink()
        return [(p, ts) for ts, p in keep]

    def refresh(self, ttl_sec=600, force=False, clear_pngs=True):
        with self.lock:
            if not force and gif_is_fresh(ttl_sec, str(self.current)):
                return [], False
            ok, _ = download_gif(self.gif_url, str(self.animation))
            if ok:
                try:
                    self.extract(str(self.animation))
                except (OSError, ValueError) as exc:
                    print(f'{self.station}: animation rejected: {exc}')
            ok, _ = download_gif(self.current_url, str(self.current))
            if ok:
                try:
                    self.extract(str(self.current))
                except (OSError, ValueError) as exc:
                    print(f'{self.station}: current image rejected: {exc}')
            return self.history(), bool(self.history())

    def augment(self, frame_data):
        # refresh() has already merged animation + actual current Z product.
        return self.history() or frame_data
