"""Offline checks against real IMD CDWRTERLS MAX(Z) pixels."""
import math
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import georef_thiruvananthapuram as geo
from native_radar import (thiruvananthapuram_timestamp, decode_reflectivity,
                          NativeRadarFeed, THIRUVANANTHAPURAM_DBZ, _canonical_color)
from fuzzy import rgb_to_dbz, dbz_to_label
from PIL import Image
import numpy as np

FIXTURES = Path(__file__).parent / 'fixtures'
SOURCE = FIXTURES / 'thiruvananthapuram.png'
PREVIOUS = FIXTURES / 'thiruvananthapuram_previous.png'
TIME = datetime(2026, 10, 5, 1, 7, 31, tzinfo=timezone.utc)


class ThiruvananthapuramTests(unittest.TestCase):
    def image(self, path=SOURCE):
        with Image.open(path) as image:
            return image.convert('RGB')

    def test_explicit_date_handles_midnight(self):
        self.assertEqual(thiruvananthapuram_timestamp(self.image()), TIME)
        self.assertEqual(thiruvananthapuram_timestamp(self.image(PREVIOUS)),
                         datetime(2026, 10, 4, 23, 35, 13, tzinfo=timezone.utc))
        image = self.image()
        image.paste('white', (239, 17, 251, 36))
        self.assertIsNone(thiruvananthapuram_timestamp(image))

    def test_white_is_light_rain_and_low_bins_are_below_detection_floor(self):
        self.assertEqual(rgb_to_dbz(*_canonical_color(THIRUVANANTHAPURAM_DBZ[7])), 30)
        self.assertEqual(dbz_to_label(30)[0], 'Light Rain')
        for dbz in THIRUVANANTHAPURAM_DBZ:
            converted = rgb_to_dbz(*_canonical_color(dbz))
            if dbz < 20:
                self.assertEqual(converted, 0)
            else:
                self.assertEqual(dbz_to_label(converted)[0], dbz_to_label(dbz)[0])

    def test_native_source_normalizes_and_excludes_no_data(self):
        image = self.image()
        # Grey no-data and an actual native white echo, within circular bounds.
        image.paste((204, 204, 204), (180, 450, 205, 475))
        image.paste((254, 254, 254), (220, 450, 245, 475))
        normalized, ts = decode_reflectivity(image, 'thiruvananthapuram')
        self.assertEqual(normalized.size, (geo.IMAGE_WIDTH, geo.IMAGE_HEIGHT))
        self.assertEqual(ts, TIME)
        arr = np.array(normalized)
        self.assertGreater(np.count_nonzero(arr), 50)
        self.assertFalse(arr[0, 0].any())
        scale = 257.7 / (240 / .877)
        def sample(x, y):
            return arr[round(geo.CENTER_PY+(y-438)/scale), round(geo.CENTER_PX+(x-300)/scale)]
        self.assertFalse(sample(192, 462).any())
        self.assertEqual(rgb_to_dbz(*sample(232, 462)), 30)

    def test_wrong_product_or_damaged_time_is_rejected(self):
        image = self.image()
        image.paste('green', (747, 586, 784, 605))
        with self.assertRaisesRegex(ValueError, 'reflectivity'):
            decode_reflectivity(image, 'thiruvananthapuram')
        with self.assertRaisesRegex(ValueError, 'layout'):
            decode_reflectivity(image.resize((572, 572)), 'thiruvananthapuram')
        image = self.image()
        image.paste('white', (7, 17, 134, 36))
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            decode_reflectivity(image, 'thiruvananthapuram')
        with patch('native_radar.thiruvananthapuram_timestamp',
                   return_value=datetime.now(timezone.utc)+timedelta(hours=1)):
            with self.assertRaisesRegex(ValueError, 'Future'):
                decode_reflectivity(self.image(), 'thiruvananthapuram')

    def test_physical_scale_inverse_and_circular_coverage(self):
        self.assertEqual(geo.latlon_to_pixel(geo.CENTER_LAT, geo.CENTER_LON),
                         (geo.CENTER_PX, geo.CENTER_PY))
        for bearing in range(0, 360, 45):
            la, lo = map(math.radians, (geo.CENTER_LAT, geo.CENTER_LON))
            b, d = math.radians(bearing), 100 / 6371
            lat = math.asin(math.sin(la)*math.cos(d)+math.cos(la)*math.sin(d)*math.cos(b))
            lon = lo + math.atan2(math.sin(b)*math.sin(d)*math.cos(la), math.cos(d)-math.sin(la)*math.sin(lat))
            x, y = geo.latlon_to_pixel(math.degrees(lat), math.degrees(lon))
            self.assertAlmostEqual(math.hypot(x-geo.CENTER_PX, y-geo.CENTER_PY)*.877, 100, delta=.65)
            inverse = geo.pixel_to_latlon(x, y)
            self.assertLess(math.hypot(inverse[0]-math.degrees(lat), inverse[1]-math.degrees(lon))*111.32, .8)
        self.assertFalse(geo.is_within_radar(*geo.pixel_to_latlon(0, 0)))
        self.assertFalse(geo.is_within_radar(13.0827, 80.2707))

    def test_south_indian_cities_select_new_radar(self):
        from main import _detect_radar
        for point in [(8.5241, 76.9366), (8.8932, 76.6141), (8.0883, 77.5385),
                      (9.9312, 76.2673), (9.4981, 76.3388), (8.7139, 77.7567)]:
            self.assertEqual(_detect_radar(*point), 'thiruvananthapuram')
        self.assertEqual(_detect_radar(12.9141, 74.8560), 'mangaluru')

    def test_history_deduplicates_and_orders_real_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            feed = NativeRadarFeed('thiruvananthapuram', 'TVM', '')
            feed.folder = Path(temp)
            feed.index = feed.folder / 'history.json'
            self.assertEqual(len(feed.extract(PREVIOUS)), 1)
            self.assertEqual(len(feed.extract(PREVIOUS)), 1)
            history = feed.extract(SOURCE)
            self.assertEqual(len(history), 2)
            self.assertEqual(history[-1][1], TIME)
            self.assertLess(history[0][1], history[1][1])

    def test_stale_and_single_scan_do_not_claim_forecasts(self):
        from main import _forecast_ready, _require_native_fresh, _require_forecast_history
        from fastapi import HTTPException
        ts = datetime.now(timezone.utc)
        state = {'latest_ts': ts, 'frame_data': [('one.png', ts)]}
        self.assertFalse(_forecast_ready('thiruvananthapuram', state))
        with self.assertRaises(HTTPException):
            _require_forecast_history('thiruvananthapuram', state)
        state['frame_data'].append(('two.png', ts-timedelta(minutes=15)))
        self.assertTrue(_forecast_ready('thiruvananthapuram', state))
        state['latest_ts'] = ts-timedelta(hours=2)
        with self.assertRaises(HTTPException) as exc:
            _require_native_fresh('thiruvananthapuram', state)
        self.assertEqual(exc.exception.status_code, 503)
        self.assertIn('Thiruvananthapuram radar feed is stale', exc.exception.detail)


if __name__ == '__main__':
    unittest.main()
