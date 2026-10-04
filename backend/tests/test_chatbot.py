"""Offline regressions for AI geocoding and provider failover."""
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import httpx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chatbot


def response(status=200, payload=None, headers=None):
    return httpx.Response(status, json=payload, headers=headers,
                          request=httpx.Request('GET', 'https://example.test/search'))


NOMINATIM = [{'display_name': 'Connaught Place, Delhi', 'lat': '28.6317', 'lon': '77.2194'}]
PHOTON = {'features': [{'properties': {'name': 'Connaught Place', 'city': 'Delhi', 'countrycode': 'IN'},
                        'geometry': {'coordinates': [77.2194, 28.6317]}}]}


class GeocodeTests(unittest.TestCase):
    def setUp(self):
        chatbot._GEOCODE_CACHE.clear()
        chatbot._GEOCODE_LAST_REQUEST.clear()
        chatbot._GEOCODE_COOLDOWN.clear()
        self.clock = patch.object(chatbot.time, 'monotonic', return_value=1000.0)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def test_repeated_normalized_place_uses_cache_and_returns_copy(self):
        with patch.object(chatbot.httpx, 'get', return_value=response(payload=NOMINATIM)) as get:
            first = chatbot._geocode_place(' Connaught  Place Delhi ')
            first['matches'][0]['lat'] = 0
            again = chatbot._geocode_place('connaught place delhi')
        self.assertEqual(get.call_count, 1)
        self.assertEqual(again['matches'][0]['lat'], 28.6317)

    def test_rate_limit_falls_back_and_cools_down_primary(self):
        with patch.object(chatbot.httpx, 'get', side_effect=[
            response(429, {}, {'Retry-After': '120'}), response(payload=PHOTON),
            response(payload=PHOTON)]) as get, patch.object(chatbot.time, 'sleep'):
            result = chatbot._geocode_place('Connaught Place Delhi')
            chatbot._geocode_place('Another Delhi place')
        self.assertEqual(result['matches'][0]['lat'], 28.6317)
        self.assertEqual(chatbot._GEOCODE_COOLDOWN['nominatim'], 1120)
        self.assertIn('photon', get.call_args_list[1].args[0])
        self.assertIn('photon', get.call_args_list[2].args[0])

    def test_primary_timeout_uses_fallback(self):
        with patch.object(chatbot.httpx, 'get', side_effect=[httpx.ReadTimeout('timeout'), response(payload=PHOTON)]):
            self.assertTrue(chatbot._geocode_place('Delhi')['matches'])

    def test_requests_are_spaced(self):
        chatbot._GEOCODE_LAST_REQUEST['nominatim'] = 999.8
        with patch.object(chatbot.httpx, 'get', return_value=response(payload=NOMINATIM)), patch.object(chatbot.time, 'sleep') as sleep:
            chatbot._geocode_place('Delhi')
        self.assertAlmostEqual(sleep.call_args.args[0], 0.9)

    def test_no_match_is_not_provider_failure(self):
        with patch.object(chatbot.httpx, 'get', side_effect=[response(payload=[]), response(payload={'features': []})]):
            result = chatbot._geocode_place('nonexistent')
        self.assertEqual(result['matches'], [])
        self.assertIn('note', result)
        self.assertNotIn('error', result)

    def test_both_failures_are_retryable_and_not_cached(self):
        with patch.object(chatbot.httpx, 'get', side_effect=[response(429, {}), response(503, {})]):
            result = chatbot._geocode_place('Delhi')
        self.assertTrue(result['retryable'])
        self.assertEqual(result['matches'], [])
        self.assertFalse(chatbot._GEOCODE_CACHE)

    def test_country_and_coordinate_filters(self):
        foreign = {'features': [{'properties': {'countrycode': 'US', 'name': 'Delhi'},
                                'geometry': {'coordinates': [-74, 42]}}]}
        self.assertEqual(chatbot._geocode_matches('photon', foreign), [])
        self.assertEqual(chatbot._geocode_matches('nominatim', [{'lat': 'nan', 'lon': '77'}]), [])

    def test_expired_cache_refetches(self):
        chatbot._GEOCODE_CACHE['delhi'] = (999, {'matches': NOMINATIM})
        with patch.object(chatbot.httpx, 'get', return_value=response(payload=NOMINATIM)) as get:
            chatbot._geocode_place('Delhi')
        get.assert_called_once()

    def test_cache_is_bounded(self):
        for i in range(chatbot._GEOCODE_CACHE_LIMIT):
            chatbot._GEOCODE_CACHE[str(i)] = (2000, {'matches': []})
        with patch.object(chatbot.httpx, 'get', return_value=response(payload=NOMINATIM)):
            chatbot._geocode_place('Delhi')
        self.assertEqual(len(chatbot._GEOCODE_CACHE), chatbot._GEOCODE_CACHE_LIMIT)
        self.assertNotIn('0', chatbot._GEOCODE_CACHE)

    def test_retry_after_date(self):
        with patch.object(chatbot.time, 'time', return_value=0):
            self.assertEqual(chatbot._retry_after_seconds('Thu, 01 Jan 1970 00:02:00 GMT'), 120)
        self.assertEqual(chatbot._retry_after_seconds('bad'), 60)


class ProviderTests(unittest.TestCase):
    def fake_google(self):
        class APIError(Exception):
            def __init__(self, code):
                self.code = code
        genai = ModuleType('google.genai')
        genai.types = SimpleNamespace()
        genai.errors = SimpleNamespace(APIError=APIError)
        genai.Client = Mock(return_value=Mock())
        google = ModuleType('google')
        google.genai = genai
        return {'google': google, 'google.genai': genai}, APIError

    def test_non_quota_failure_does_not_claim_quota(self):
        modules, APIError = self.fake_google()
        with patch.dict(sys.modules, modules), patch.object(chatbot, 'get_api_keys', return_value=['key']), patch.object(chatbot, '_groq_key', return_value=''), patch.object(chatbot, '_build_contents', return_value=[]), patch.object(chatbot, '_run_once', side_effect=APIError(403)):
            output = ''.join(chatbot.chat_stream([]))
        self.assertIn('HTTP 403', output)
        self.assertNotIn('request limit', output)

    def test_quota_failure_has_correct_message(self):
        modules, APIError = self.fake_google()
        with patch.dict(sys.modules, modules), patch.object(chatbot, 'get_api_keys', return_value=['key']), patch.object(chatbot, '_groq_key', return_value=''), patch.object(chatbot, '_build_contents', return_value=[]), patch.object(chatbot, '_run_once', side_effect=APIError(429)):
            output = ''.join(chatbot.chat_stream([]))
        self.assertIn('request limit', output)

    def test_bad_first_key_tries_second(self):
        modules, APIError = self.fake_google()
        with patch.dict(sys.modules, modules), patch.object(chatbot, 'get_api_keys', return_value=['bad', 'good']), patch.object(chatbot, '_groq_key', return_value=''), patch.object(chatbot, '_build_contents', return_value=[]), patch.object(chatbot, '_run_once', side_effect=[APIError(403), iter([chatbot._sse({'type': 'done'})])]) as run:
            output = ''.join(chatbot.chat_stream([]))
        self.assertIn('done', output)
        self.assertEqual(run.call_count, 2)

    def test_groq_only_needs_no_google_import(self):
        with patch.dict(sys.modules, {'google': None}), patch.object(chatbot, 'get_api_keys', return_value=[]), patch.object(chatbot, '_groq_key', return_value='groq'), patch.object(chatbot, '_run_groq', return_value=iter([chatbot._sse({'type': 'done'})])):
            self.assertIn('done', ''.join(chatbot.chat_stream([])))

    def test_no_fallback_after_partial_text(self):
        modules, APIError = self.fake_google()
        def partial(*args):
            yield chatbot._sse({'type': 'text', 'delta': 'Hello'})
            raise APIError(429)
        with patch.dict(sys.modules, modules), patch.object(chatbot, 'get_api_keys', return_value=['key']), patch.object(chatbot, '_groq_key', return_value='groq'), patch.object(chatbot, '_build_contents', return_value=[]), patch.object(chatbot, '_run_once', side_effect=partial), patch.object(chatbot, '_run_groq') as groq:
            output = ''.join(chatbot.chat_stream([]))
        self.assertIn('interrupted', output)
        groq.assert_not_called()


if __name__ == '__main__':
    unittest.main()
