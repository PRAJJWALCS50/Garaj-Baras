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
from functools import lru_cache

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
MANGALURU_DBZ = [20 + i * 40 / 15 for i in range(16)]
THIRUVANANTHAPURAM_DBZ = [2 + i * 4 for i in range(15)]


@lru_cache(maxsize=1)
def _thiruvananthapuram_templates():
    data = json.loads((ROOT / 'thiruvananthapuram_digits.json').read_text(encoding='utf-8'))
    return {label: [np.unpackbits(np.frombuffer(bytes.fromhex(value), dtype=np.uint8)).reshape(
        (24, 16)).astype(bool) for value in variants] for label, variants in data.items()}


def thiruvananthapuram_timestamp(image):
    """Read the explicit dated UTC header, without a runtime OCR binary."""
    templates = _thiruvananthapuram_templates()

    def read(box, count, separators):
        glyphs = _glyphs(image, box)
        if len(glyphs) != count:
            return ''
        result = []
        for index, glyph in enumerate(glyphs):
            expected = separators.get(index)
            if expected:
                if max(float((glyph == variant).mean()) for variant in templates[expected]) < .95:
                    return ''
                result.append(expected)
                continue
            scores = sorted((max(float((glyph == variant).mean()) for variant in variants), label)
                            for label, variants in templates.items() if label.isdigit())
            if scores[-1][0] < .95 or scores[-1][0] - scores[-2][0] < .03:
                return ''
            result.append(scores[-1][1])
        return ''.join(result)

    date = read((7, 17, 134, 36), 10, {4: '/', 7: '/'})
    clock = read((238, 17, 328, 36), 8, {2: ':', 5: ':'})
    try:
        return datetime.strptime(date + ' ' + clock, '%Y/%m/%d %H:%M:%S').replace(
            tzinfo=timezone.utc).astimezone(IST)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _mangaluru_templates():
    data = json.loads((ROOT / 'mangaluru_glyphs.json').read_text(encoding='utf-8'))
    return {kind: {label: [np.unpackbits(np.frombuffer(bytes.fromhex(v), dtype=np.uint8)).reshape(
        (24, 16 if kind == 'digits' else 48)).astype(bool) for v in variants]
        for label, variants in group.items()} for kind, group in data.items()}


def mangaluru_timestamp(image):
    """Read the explicit UTC date/time; never substitute today's date."""
    templates = _mangaluru_templates()

    def match(glyph, group, minimum, margin):
        scores = sorted((max(float((glyph == variant).mean()) for variant in variants), label)
                        for label, variants in group.items())
        if scores[-1][0] < minimum or scores[-1][0] - scores[-2][0] < margin:
            return None
        return scores[-1][1]

    def digits(box, count, separators=None):
        glyphs = _glyphs(image, box)
        if len(glyphs) != count:
            return None
        result = []
        for i, glyph in enumerate(glyphs):
            if separators and i in separators:
                # Reject damaged separators rather than shifting digit positions.
                if not .25 < float(glyph.mean()) < .8:
                    return None
                result.append(':')
            else:
                char = match(glyph, templates['digits'], .85, .035)
                if char is None:
                    return None
                result.append(char)
        return ''.join(result)

    clock = digits((1084, 94, 1141, 107), 8, {2, 5})
    # Month words have different widths; locate date separators instead of
    # assuming Oct's x-position for the year on future scans.
    date_ink = np.array(image.crop((1185, 94, 1305, 107)).convert('L')) < 80
    bounds = np.where(np.diff(np.r_[False, date_ink.any(axis=0), False]))[0]
    runs = list(zip(bounds[::2], bounds[1::2]))
    if len(runs) < 9:
        return None
    for index in (2, -5):
        left, right = runs[index]
        ys = np.where(date_ink[:, left:right].any(axis=1))[0]
        if not len(ys) or ys[-1] - ys[0] > 2:
            return None
    day = digits((1185, 94, 1185+int(runs[1][1]), 107), 2)
    year = digits((1185+int(runs[-4][0]), 94, 1185+int(runs[-1][1]), 107), 4)
    ink = date_ink[:, runs[3][0]:runs[-5][0]]
    ys, xs = np.where(ink)
    if not len(xs):
        return None
    glyph = np.array(Image.fromarray(ink[ys.min():ys.max()+1, xs.min():xs.max()+1]).resize(
        (48, 24), Image.Resampling.NEAREST)) > 0
    month = match(glyph, templates['months'], .82, .08)
    if not all((clock, day, year, month)):
        return None
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    try:
        h, m, s = map(int, clock.split(':'))
        return datetime(int(year), months.index(month)+1, int(day), h, m, s,
                        tzinfo=timezone.utc).astimezone(IST)
    except ValueError:
        return None


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
    elif station == 'thiruvananthapuram':
        if image.size != (1082, 720):
            raise ValueError('Thiruvananthapuram layout changed')
        colors = [arr[687 - i * 18, 765] for i in range(15)]
        # Fifteen four-dBZ intervals; white is 28..32 dBZ, not 60.
        # No-data is grey and is deliberately omitted from the palette.
        anchors = {0: (57, 0, 159), 5: (82, 208, 254),
                   7: (254, 254, 254), 14: (199, 0, 78)}
        if any(np.linalg.norm(colors[i].astype(float) - c) > 8 for i, c in anchors.items()):
            raise ValueError('Thiruvananthapuram feed is not supported MAX(Z) reflectivity')
        # North-up map, explicit site/range in header. Crosshair (300,438),
        # measured outer ring radius 257.7 px (240 km), excluding vertical panels.
        box, source_center, source_radius, range_km = (42, 180, 559, 698), 258., 257.7, 240.
        values = THIRUVANANTHAPURAM_DBZ
        timestamp = thiruvananthapuram_timestamp(image)
    elif station == 'mangaluru':
        if image.size != (1310, 1080):
            raise ValueError('Mangaluru layout changed')
        # Sixteen native legend levels, 20..60 dBZ. White is 41.3 dBZ,
        # not the legacy engine's extreme-rain white. Sample modal colours
        # away from labels/borders; exact membership excludes basemap cyan.
        colors = []
        for i in range(16):
            y = round(803 - i * 455 / 16)
            pixels = arr[y-2:y+2, 1179:1215].reshape(-1, 3)
            unique, counts = np.unique(pixels, axis=0, return_counts=True)
            colors.append(unique[counts.argmax()])
        anchors = {0: (0, 0, 147), 7: (134, 240, 255),
                   8: (255, 255, 255), 15: (163, 0, 0)}
        if any(np.linalg.norm(colors[i].astype(float) - c) > 12 for i, c in anchors.items()):
            raise ValueError('Mangaluru feed is not supported MAXDISPLAY(Z) reflectivity')
        box, source_center, source_radius, range_km = (0, 200, 880, 1080), 440., 440., 250.
        values = MANGALURU_DBZ
        timestamp = mangaluru_timestamp(image)
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
    if station == 'mangaluru':
        # IMD paints district boundaries in the same white as its 41.3-dBZ
        # bin. Thin isolated white cartography is not a storm: retain white
        # only inside a broad echo or alongside other measured rain colours.
        white = (crop == 255).all(axis=2)
        other_echo = normalized.any(axis=2) & ~white
        white_support = cv2.boxFilter(white.astype(np.float32), -1, (7, 7), normalize=False)
        echo_support = cv2.boxFilter(other_echo.astype(np.float32), -1, (7, 7), normalize=False)
        normalized[white & (white_support < 25) & (echo_support < 4)] = 0
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
