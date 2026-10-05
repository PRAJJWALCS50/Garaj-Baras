"""No live pushes: exercise delivery outcomes and persisted retry state."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import alerts
from pywebpush import WebPushException

SUB = {'endpoint': 'https://push.example.test/device', 'keys': {}}


class DeliveryTests(unittest.TestCase):
    def send(self, side_effect=None):
        with patch('alerts.load_vapid', return_value={'private_key_pem': 'fake'}), \
             patch('py_vapid.Vapid.from_pem'), \
             patch('pywebpush.webpush', return_value=Mock(status_code=201), side_effect=side_effect) as push:
            result = alerts.send_test_verbose(json.dumps(SUB), 'Test', 'Test')
            self.assertEqual(push.call_args.kwargs['timeout'], 15)
            return result

    def test_accepted(self):
        self.assertTrue(self.send()['ok'])

    def test_rejections_are_not_success(self):
        for status in (403, 429, 500, 404, 410):
            with self.subTest(status=status):
                result = self.send(WebPushException('rejected', response=Mock(status_code=status)))
                self.assertFalse(result['ok'])
                self.assertEqual(result['dead'], status in (404, 410))
                self.assertEqual(result['status'], status)

    def test_timeout_and_missing_keys_are_retryable(self):
        self.assertFalse(self.send(TimeoutError())['dead'])
        with patch('alerts.load_vapid', return_value={}):
            self.assertFalse(alerts._send_push(json.dumps(SUB), 'Test', 'Test')['ok'])


class SavedAlertTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for target, value in [('alerts.DB_PATH', str(Path(self.temp.name)/'alerts.db')),
                              ('alerts._db_ready', False), ('db.IS_POSTGRES', False),
                              ('db.AUTOINC_PK', 'INTEGER PRIMARY KEY AUTOINCREMENT')]:
            p = patch(target, value); p.start(); self.addCleanup(p.stop)
        alerts.subscribe(SUB, 28.6, 77.2, 'Delhi')
        frame = Path(self.temp.name)/'frame.png'
        Image.new('RGB', (8, 8)).save(frame)
        self.state = {'latest_frame': str(frame), 'movement': (0, 0)}

    def row(self):
        with alerts._conn() as c:
            return c.execute('SELECT state,last_notified_at,last_rain_alert_at FROM subscriptions').fetchone()

    def process(self, delivery):
        slots = [{'slot_mins': 0, 'has_rain': True, 'probability': 97, 'intensity': 'Heavy Rain'}]
        with patch('alerts._send_push', return_value=delivery) as push, \
             patch('optical_flow.isolate_rain', return_value=np.zeros((8, 8))), \
             patch('nowcast.compute_nowcast_slots', return_value=slots):
            alerts.process_alerts('delhi', self.state, lambda *_: True, lambda *_: (4, 4))
            return push.call_count

    def test_failed_rain_push_retries_without_cooldown(self):
        self.assertEqual(self.process({'ok': False, 'dead': False}), 1)
        self.assertEqual(self.row(), ('clear', None, None))
        self.assertEqual(self.process({'ok': True, 'dead': False}), 1)
        state, head, arrival = self.row()
        self.assertEqual(state, 'raining'); self.assertIsNotNone(head); self.assertIsNotNone(arrival)
        self.assertEqual(self.process({'ok': True, 'dead': False}), 0)

    def test_expired_endpoint_removed(self):
        self.process({'ok': False, 'dead': True})
        self.assertIsNone(self.row())

    def test_server_status_and_location_change(self):
        with patch('alerts.public_key', return_value='public'):
            status = alerts.subscription_status(SUB['endpoint'])
            self.assertEqual(status['label'], 'Delhi'); self.assertEqual(status['lat'], 28.6)
            self.assertNotIn('sub_json', status)
            self.assertFalse(alerts.subscription_status('missing')['enabled'])
        self.process({'ok': True, 'dead': False})
        alerts.subscribe(SUB, 28.6, 77.2, 'Renamed')
        self.assertEqual(self.row()[0], 'raining')
        alerts.subscribe(SUB, 12.9, 74.85, 'Mangaluru')
        self.assertEqual(self.row(), ('clear', None, None))

    def test_incomplete_native_history_never_pushes(self):
        with patch('alerts._send_push') as push:
            alerts.process_alerts('mangaluru', self.state, lambda *_: True, lambda *_: (4, 4))
            push.assert_not_called()

    def test_test_endpoint_reports_failure_and_prunes_only_expired(self):
        from main import alerts_test, AlertEndpointRequest
        req = AlertEndpointRequest(endpoint=SUB['endpoint'])
        for dead in (False, True):
            with patch('alerts.send_test_verbose', return_value={'ok': False, 'dead': dead}) as send:
                self.assertFalse(alerts_test(req)['ok'])
                send.assert_called_once()
                self.assertEqual(alerts.get_subscription(SUB['endpoint']) is None, dead)

    def test_stale_scan_skipped_by_instant_and_fresh_cache_sweep(self):
        import main
        from datetime import datetime, timezone, timedelta
        stale = {**self.state, 'frame_data': [1, 2],
                 'latest_ts': datetime.now(timezone.utc)-timedelta(hours=2)}
        ready = Mock(); ready.is_set.return_value = True
        reg = {'ready': ready, 'cache': stale, 'georef': Mock()}
        with patch.dict(main._RADAR_REGISTRY, {'mangaluru': reg}), \
             patch('main._detect_radar', return_value='mangaluru'), \
             patch('main._ensure_radar_fresh_blocking', return_value=stale), \
             patch('main._is_fresh', return_value=True), \
             patch('alerts.process_alerts') as process, \
             patch('alerts.all_subscription_coords', return_value=[(12.9, 74.85)]), \
             patch('journeys.estimated_positions', return_value=[]), \
             patch('journeys.process_journeys', return_value={}):
            main._instant_alert_check(SUB['endpoint'], 12.9, 74.85)
            main._sweep_alerts()
            process.assert_not_called()

    def test_journey_transient_failure_preserves_watch_and_retries(self):
        import journeys
        with patch('journeys.DB_PATH', alerts.DB_PATH), patch('journeys._db_ready', False):
            jid = journeys.start_journey(SUB, [
                {'lat': 28.6, 'lon': 77.2, 'cum_km': 0},
                {'lat': 28.7, 'lon': 77.2, 'cum_km': 10}], 30)
            geo = Mock(); geo.is_within_radar.return_value = True
            geo.latlon_to_pixel.return_value = (4, 4)
            bundle = {'state': self.state, 'georef': geo, 'lag_info': {'lag_mins': 5}}
            with patch('journeys.check_route_rain', return_value=[]), \
                 patch('journeys.enrich_results', return_value=[{'rain_expected': True, 'eta_mins': 10, 'label': 'Heavy Rain'}]), \
                 patch('optical_flow.isolate_rain', return_value=np.zeros((8, 8))), \
                 patch('journeys._send_push', side_effect=[{'ok': False, 'dead': False}, {'ok': True, 'dead': False}]) as send:
                run = lambda: journeys.process_journeys(lambda *_: 'delhi', lambda *_: bundle)
                self.assertEqual(run()['notified'], 0)
                with journeys._conn() as c:
                    self.assertEqual(c.execute('SELECT active,last_notified_at FROM journeys WHERE id=?', (jid,)).fetchone(), (1, None))
                self.assertEqual(run()['notified'], 1)
                self.assertEqual(send.call_count, 2)


if __name__ == '__main__':
    unittest.main()
