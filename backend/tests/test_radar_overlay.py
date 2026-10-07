"""Geographic alignment and freshness boundary for the route overlay."""
import sys
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from io import BytesIO
import base64
import math
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_overlay import build_overlay
from india_mosaic import render_png

class Geo:
    CENTER_LAT, CENTER_LON = 28, 77
    IMAGE_WIDTH = IMAGE_HEIGHT = 100
    @staticmethod
    def pixel_to_latlon(x, y): return 29 - y / 50, 76 + x / 50
    @staticmethod
    def latlon_to_pixel(lat, lon): return (lon - 76) * 50, (29 - lat) * 50

class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / 'frame.png')
        a = np.zeros((100, 100, 3), np.uint8)
        a[48:54, 48:54] = [255, 255, 0]
        Image.fromarray(a).save(self.path)
        self.now = datetime(2026, 10, 7, 10, tzinfo=timezone.utc)

    def test_measured_times_deduplicated_bounded_and_sorted(self):
        frames = [(self.path, self.now - timedelta(minutes=m)) for m in (100, 60, 50, 40, 30, 20, 10, 10, -2)]
        result = build_overlay('test', Geo, frames, self.now)
        self.assertEqual(len(result['frames']), 4)
        self.assertEqual(result['latest_timestamp'], (self.now - timedelta(minutes=10)).isoformat())
        self.assertEqual(result['kind'], 'observed')
        self.assertEqual(result['bounds'], [[27.02, 76], [29, 77.98]])
        image = Image.open(BytesIO(base64.b64decode(result['frames'][-1]['image'].split(',')[1])))
        self.assertEqual(image.size, (640, 640))
        self.assertEqual(image.mode, 'RGBA')
        self.assertEqual(image.getpixel((0, 0))[3], 0)

    def test_stale_future_and_naive_frames_fail_closed(self):
        for stamp in (self.now - timedelta(minutes=91), self.now + timedelta(minutes=1), self.now.replace(tzinfo=None)):
            with self.assertRaises(ValueError): build_overlay('test', Geo, [(self.path, stamp)], self.now)

    def test_missing_image_fails_closed(self):
        with self.assertRaises(OSError): build_overlay('test', Geo, [('missing.png', self.now)], self.now)

    def test_rain_lands_at_expected_web_mercator_position(self):
        png = render_png([(Geo, self.path)], bounds=(27, 76, 29, 78), width=640, height=640)
        arr = np.array(Image.open(BytesIO(png)))
        ys, xs = np.nonzero(arr[:, :, 3])
        merc = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        expected_y = (merc(29) - merc(28)) / (merc(29) - merc(27)) * 639
        self.assertLess(abs(xs.mean() - 319.5), 15)
        self.assertLess(abs(ys.mean() - expected_y), 15)
        self.assertTrue((arr[ys, xs, :3] == [255, 255, 0]).all())

if __name__ == '__main__': unittest.main()
