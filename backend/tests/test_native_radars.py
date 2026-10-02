"""Regression checks against real IMD products captured 2026-10-02."""
import math
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import georef_sohra
import georef_mahabaleshwar
from fuzzy import rgb_to_dbz, dbz_to_label
from native_radar import (decode_reflectivity, sohra_timestamp, NativeRadarFeed,
                          _canonical_color, SOHRA_DBZ, MBL_DBZ)
from PIL import Image
import numpy as np

FIXTURES = Path(__file__).parent / 'fixtures'
TIME = datetime(2026, 10, 2, 11, 30, 9, tzinfo=timezone.utc)


class NativeRadarTests(unittest.TestCase):
    def decode(self, filename, station):
        with Image.open(FIXTURES / filename) as image, patch(
                'native_radar.ocr_timestamp_from_image', return_value=TIME):
            return decode_reflectivity(image.convert('RGB'), station)

    def test_sohra_explicit_date_and_time(self):
        with Image.open(FIXTURES / 'sohra.png') as image:
            image = image.convert('RGB')
            ts = sohra_timestamp(image)
            self.assertEqual(ts.astimezone(timezone.utc).isoformat(), '2026-10-02T10:48:38+00:00')
            # A damaged digit is rejected rather than receiving today's time.
            image.paste('white', (239, 17, 251, 36))
            self.assertIsNone(sohra_timestamp(image))

    def test_native_white_is_moderate_not_extreme_rain(self):
        # Sohra's white interval is 34.19..35.81 dBZ, unlike Delhi's 60 dBZ white.
        self.assertEqual(rgb_to_dbz(*_canonical_color(SOHRA_DBZ[15])), 35)
        self.assertEqual(dbz_to_label(rgb_to_dbz(*_canonical_color(MBL_DBZ[12])))[0], 'Heavy Rain')
        for dbz in SOHRA_DBZ + MBL_DBZ:
            converted = rgb_to_dbz(*_canonical_color(dbz))
            if dbz >= 20:
                self.assertEqual(dbz_to_label(converted)[0], dbz_to_label(dbz)[0])
            else:
                self.assertEqual(converted, 0)

    def test_both_sources_normalize_to_engine_geometry(self):
        for station, module in [('sohra', georef_sohra), ('mahabaleshwar', georef_mahabaleshwar)]:
            image, ts = self.decode(station + '.png', station)
            self.assertEqual(image.size, (module.IMAGE_WIDTH, module.IMAGE_HEIGHT))
            arr = np.array(image)
            self.assertGreater(np.count_nonzero(arr), 50)
            self.assertTrue((arr[0, 0] == 0).all())
            self.assertLess(np.count_nonzero(arr.any(axis=2)) / arr.shape[0] ** 2, .2)
            self.assertIsNotNone(ts)

    def test_velocity_feed_cannot_be_used_for_rain(self):
        with self.assertRaisesRegex(ValueError, 'not MAX_Z'):
            self.decode('mahabaleshwar_velocity.png', 'mahabaleshwar')

    def test_physical_scale_range_and_inverse(self):
        for geo in [georef_sohra, georef_mahabaleshwar]:
            self.assertEqual(geo.latlon_to_pixel(geo.CENTER_LAT, geo.CENTER_LON),
                             (geo.CENTER_PX, geo.CENTER_PY))
            for bearing in range(0, 360, 30):
                # Independently calculated destination 100 km from the site.
                lat1, lon1 = map(math.radians, (geo.CENTER_LAT, geo.CENTER_LON))
                b, d = math.radians(bearing), 100 / 6371
                lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(b))
                lon2 = lon1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(lat1), math.cos(d) - math.sin(lat1) * math.sin(lat2))
                x, y = geo.latlon_to_pixel(math.degrees(lat2), math.degrees(lon2))
                self.assertAlmostEqual(math.hypot(x - geo.CENTER_PX, y - geo.CENTER_PY) * .877, 100, delta=.65)
                la, lo = geo.pixel_to_latlon(x, y)
                self.assertLess(math.hypot(la - math.degrees(lat2), lo - math.degrees(lon2)) * 111.32, .8)
            # Square crop corners must not extend circular radar coverage.
            corner = geo.pixel_to_latlon(0, 0)
            self.assertFalse(geo.is_within_radar(*corner))
            beyond = geo.pixel_to_latlon(geo.CENTER_PX, geo.CENTER_PY - (geo.RANGE_KM + 1) / .877)
            self.assertFalse(geo.is_within_radar(*beyond))

    def test_known_cities_select_new_radars(self):
        from main import _detect_radar
        for lat, lon in [(25.5788, 91.8933), (26.1445, 91.7362)]:
            self.assertEqual(_detect_radar(lat, lon), 'sohra')
        for lat, lon in [(18.5204, 73.8567), (17.6805, 74.0183), (16.9902, 73.312)]:
            self.assertEqual(_detect_radar(lat, lon), 'mahabaleshwar')
        self.assertEqual(_detect_radar(28.6139, 77.2090), 'delhi')

    def test_history_deduplicates_real_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            feed = NativeRadarFeed('mahabaleshwar', 'MBL', '')
            feed.folder = Path(temp)
            feed.index = feed.folder / 'history.json'
            source = FIXTURES / 'mahabaleshwar.png'
            with patch('native_radar.ocr_timestamp_from_image', return_value=TIME):
                first = feed.extract(source)
                repeated = feed.extract(source)
            self.assertEqual(len(first), 1)
            self.assertEqual(len(repeated), 1)
            # A rejected MAX_V frame cannot add fake motion/history.
            self.assertEqual(feed.extract(FIXTURES / 'mahabaleshwar_velocity.png'), repeated)
            with patch('native_radar.ocr_timestamp_from_image', return_value=TIME + timedelta(minutes=10)):
                self.assertEqual(len(feed.extract(source)), 2)

    def test_initial_frame_does_not_claim_a_motion_forecast(self):
        from main import _forecast_ready, _require_forecast_history
        from fastapi import HTTPException
        one = {'frame_data': [('frame.png', TIME)]}
        self.assertFalse(_forecast_ready('mahabaleshwar', one))
        with self.assertRaises(HTTPException) as exc:
            _require_forecast_history('mahabaleshwar', one)
        self.assertEqual(exc.exception.status_code, 503)
        one['frame_data'].append(('next.png', TIME + timedelta(minutes=10)))
        self.assertTrue(_forecast_ready('mahabaleshwar', one))

    def test_slow_cold_start_returns_retryable_response(self):
        from main import nowcast_location, NowcastRequest
        from fastapi import HTTPException
        with patch('main._load_sohra_radar_state', return_value={'clutter_mask': None}), patch(
                'main._touch_radar_and_evict'):
            with self.assertRaises(HTTPException) as exc:
                nowcast_location(NowcastRequest(lat=25.5788, lon=91.8933))
        self.assertEqual(exc.exception.status_code, 503)
        self.assertIn('warming up', exc.exception.detail)


if __name__ == '__main__':
    unittest.main()
