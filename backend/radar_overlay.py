"""Bounded observed radar playback, rendered through the existing georeferences."""
import base64
import math
from datetime import timezone

from india_mosaic import render_png
from PIL import Image


def build_overlay(name, georef, frame_data, now):
    # Use measured scan time, never fetch time, for freshness. Four frames keep
    # payloads small; no radar state or arrays are retained by this module.
    frames = {}
    for path, stamp in frame_data:
        if stamp.tzinfo is None:
            continue
        age = (now - stamp).total_seconds() / 60
        if 0 <= age <= 90:
            frames[stamp] = path
    if not frames:
        raise ValueError('No recent radar scans available (90-minute limit).')
    lat, lon = georef.CENTER_LAT, georef.CENTER_LON
    # Derive the extent from the source georeference, including curved edges.
    edge = []
    w, h = georef.IMAGE_WIDTH, georef.IMAGE_HEIGHT
    for i in range(21):
        x, y = (w - 1) * i / 20, (h - 1) * i / 20
        for px, py in ((x, 0), (x, h - 1), (0, y), (w - 1, y)):
            a, b = georef.pixel_to_latlon(px, py)
            if math.isfinite(a) and math.isfinite(b):
                edge.append((a, b))
    south, north = min(p[0] for p in edge), max(p[0] for p in edge)
    west, east = min(p[1] for p in edge), max(p[1] for p in edge)
    history = []
    for stamp, path in sorted(frames.items())[-4:]:
        # A missing/corrupt frame must not become a transparent clear-weather map.
        with Image.open(path) as image:
            image.verify()
        png = render_png([(georef, path)], bounds=(south, west, north, east), width=640, height=640)
        history.append({'timestamp': stamp.astimezone(timezone.utc).isoformat(),
                        'image': 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')})
    return {'station': name, 'center': [lat, lon],
            'bounds': [[south, west], [north, east]], 'frames': history,
            'kind': 'observed', 'latest_timestamp': history[-1]['timestamp']}
