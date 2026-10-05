"""Offline regression coverage for Mangaluru's actual IMD MAXDISPLAY(Z)."""
import math
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import georef_mangaluru as geo
from native_radar import (mangaluru_timestamp, decode_reflectivity,
                          NativeRadarFeed, MANGALURU_DBZ, _canonical_color)
from fuzzy import rgb_to_dbz, dbz_to_label
from PIL import Image
import numpy as np

SOURCE = Path(__file__).parent / 'fixtures' / 'mangaluru.png'
TIME = datetime(2026, 10, 4, 23, 30, 1, tzinfo=timezone.utc)


class MangaluruTests(unittest.TestCase):
    def image(self):
        with Image.open(SOURCE) as image:
            return image.convert('RGB')

    def test_real_date_is_not_replaced_with_today(self):
        image = self.image()
        self.assertEqual(mangaluru_timestamp(image), TIME)
        image.paste('white', (1084, 94, 1091, 107))
        self.assertIsNone(mangaluru_timestamp(image))

    def test_native_palette_and_physical_geometry(self):
        image, ts = decode_reflectivity(self.image(), 'mangaluru')
        self.assertEqual(ts, TIME)
        self.assertEqual(image.size, (573, 573))
        arr = np.array(image)
        self.assertGreater(np.count_nonzero(arr), 50)
        self.assertLess(arr.any(axis=2).mean(), .1)
        self.assertFalse(arr[0, 0].any())
        white_dbz = rgb_to_dbz(*_canonical_color(MANGALURU_DBZ[8]))
        self.assertEqual(dbz_to_label(white_dbz)[0], 'Heavy Rain')
        self.assertLess(white_dbz, 60)

    def test_invalid_product_and_layout_fail_closed(self):
        image = self.image()
        image.paste('green', (1179, 795, 1215, 808))
        with self.assertRaisesRegex(ValueError, 'reflectivity'):
            decode_reflectivity(image, 'mangaluru')
        with self.assertRaisesRegex(ValueError, 'layout'):
            decode_reflectivity(image.resize((720, 720)), 'mangaluru')

    def test_white_cartography_does_not_become_extreme_rain(self):
        image = self.image()
        # Isolated thin white line versus a broad measured white echo.
        image.paste('white', (250, 600, 251, 680))
        image.paste('white', (300, 600, 325, 625))
        normalized, _ = decode_reflectivity(image, 'mangaluru')
        arr = np.array(normalized)
        scale = 440 / (250 / .877)
        def sample(x, y):
            return arr[round(286 + (y-640)/scale), round(286 + (x-440)/scale)]
        self.assertFalse(sample(250, 640).any())
        self.assertTrue(sample(312, 612).any())

    def test_circular_range_and_inverse(self):
        self.assertEqual(geo.latlon_to_pixel(geo.CENTER_LAT, geo.CENTER_LON), (286, 286))
        for bearing in range(0, 360, 45):
            la, lo = map(math.radians, (geo.CENTER_LAT, geo.CENTER_LON))
            b, d = math.radians(bearing), 100 / 6371
            lat = math.asin(math.sin(la)*math.cos(d)+math.cos(la)*math.sin(d)*math.cos(b))
            lon = lo + math.atan2(math.sin(b)*math.sin(d)*math.cos(la), math.cos(d)-math.sin(la)*math.sin(lat))
            x, y = geo.latlon_to_pixel(math.degrees(lat), math.degrees(lon))
            self.assertAlmostEqual(math.hypot(x-286, y-286)*.877, 100, delta=.65)
            inverse = geo.pixel_to_latlon(x, y)
            self.assertLess(math.hypot(inverse[0]-math.degrees(lat), inverse[1]-math.degrees(lon))*111.32, .8)
        self.assertFalse(geo.is_within_radar(*geo.pixel_to_latlon(0, 0)))

    def test_coastal_cities_choose_mangaluru(self):
        from main import _detect_radar
        for point in [(12.9141, 74.8560), (13.3409, 74.7421), (11.8745, 75.3704)]:
            self.assertEqual(_detect_radar(*point), 'mangaluru')
        self.assertFalse(geo.is_within_radar(12.9716, 77.5946))

    def test_duplicate_observations_do_not_create_motion(self):
        with tempfile.TemporaryDirectory() as temp:
            feed = NativeRadarFeed('mangaluru', 'MLR', '')
            feed.folder = Path(temp)
            feed.index = feed.folder / 'history.json'
            self.assertEqual(len(feed.extract(SOURCE)), 1)
            self.assertEqual(len(feed.extract(SOURCE)), 1)
            with patch('native_radar.mangaluru_timestamp', return_value=TIME + timedelta(minutes=10)):
                self.assertEqual(len(feed.extract(SOURCE)), 2)

    def test_stale_and_single_observation_are_not_forecasts(self):
        from main import _forecast_ready, _require_forecast_history, _require_native_fresh
        from fastapi import HTTPException
        fresh = datetime.now(timezone.utc)
        state = {'latest_ts': fresh, 'frame_data': [('one.png', fresh)]}
        self.assertFalse(_forecast_ready('mangaluru', state))
        with self.assertRaises(HTTPException):
            _require_forecast_history('mangaluru', state)
        state['frame_data'].append(('two.png', fresh-timedelta(minutes=10)))
        self.assertTrue(_forecast_ready('mangaluru', state))
        state['latest_ts'] = fresh-timedelta(hours=2)
        self.assertFalse(_forecast_ready('mangaluru', state))
        with self.assertRaises(HTTPException) as exc:
            _require_native_fresh('mangaluru', state)
        self.assertEqual(exc.exception.status_code, 503)
        self.assertIn('stale', exc.exception.detail)

    def test_assistant_routes_use_multi_radar_engine(self):
        from main import _tool_get_route_rain
        points = [(12.9141, 74.8560, 0), (13.3409, 74.7421, 40)]
        with patch('main.generate_waypoints', return_value=points), patch(
                'main.predict_waypoints', return_value={'rain_waypoints': 1}) as predict:
            self.assertEqual(_tool_get_route_rain(12.9141,74.8560,13.3409,74.7421), {'rain_waypoints': 1})
        self.assertEqual([(p.lat,p.lon,p.eta_mins) for p in predict.call_args.args[0].waypoints], points)


if __name__ == '__main__':
    unittest.main()
