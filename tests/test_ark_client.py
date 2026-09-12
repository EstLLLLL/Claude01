import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ark_client

ENV = {"ARK_API_KEY": "test-key-do-not-send", "ARK_MODEL": "deepseek-v4-pro-ga-260813",
       "ARK_BASE_URL": ark_client.DEFAULT_BASE_URL}

def response(content, finish="stop"):
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}, "finish_reason": finish}]}).encode())

class ArkClientTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, ENV)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_ark_model_and_non_thinking_payload(self):
        with patch("ark_client.urllib.request.urlopen", return_value=response('{"ok": true}')) as call:
            self.assertEqual(ark_client.chat_json([]), {"ok": True})
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, ark_client.DEFAULT_BASE_URL + "/chat/completions")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], ENV["ARK_MODEL"])
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("response_format", payload)

    def test_billing_and_auth_fail_immediately_without_exposing_secret(self):
        for status in [400, 401, 402, 403, 404]:
            with self.subTest(status=status):
                error = urllib.error.HTTPError("url", status, "failure", {}, io.BytesIO(b'{"error":{"code":"InsufficientBalance"}}'))
                with patch("ark_client.urllib.request.urlopen", side_effect=error) as call, patch("ark_client.time.sleep") as sleep:
                    with self.assertRaises(ark_client.ArkError) as caught:
                        ark_client.chat_json([])
                self.assertIn(str(status), str(caught.exception))
                self.assertNotIn(ENV["ARK_API_KEY"], str(caught.exception))
                self.assertEqual(call.call_count, 1)
                sleep.assert_not_called()

    def test_transient_rate_limit_retries_and_recovers(self):
        error = urllib.error.HTTPError("url", 429, "limited", {}, io.BytesIO(b'{}'))
        with patch("ark_client.urllib.request.urlopen", side_effect=[error, response('{"ok": true}')]) as call, patch("ark_client.time.sleep"):
            self.assertEqual(ark_client.chat_json([]), {"ok": True})
        self.assertEqual(call.call_count, 2)

    def test_exhausted_network_retry_fails(self):
        with patch("ark_client.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")) as call, patch("ark_client.time.sleep"):
            with self.assertRaises(ark_client.ArkError):
                ark_client.chat_json([])
        self.assertEqual(call.call_count, 3)

    def test_malformed_and_truncated_responses_fail(self):
        for content, finish in [("not json", "stop"), ("[]", "stop"), ('{"ok": true}', "length")]:
            with patch("ark_client.urllib.request.urlopen", side_effect=lambda *a, **k: response(content, finish)), patch("ark_client.time.sleep"):
                with self.assertRaises(ark_client.ArkError):
                    ark_client.chat_json([])

    def test_no_silent_fallback_to_old_deepseek_key(self):
        with patch.dict(os.environ, {"ARK_API_KEY": "", "DEEPSEEK_API_KEY": "old-key"}):
            with self.assertRaises(ark_client.ArkError):
                ark_client.settings()

    def test_invalid_json_can_recover_without_accepting_partial_content(self):
        with patch("ark_client.urllib.request.urlopen", side_effect=[response('not json'), response('{"ok": true}')]) as call, patch("ark_client.time.sleep"):
            self.assertEqual(ark_client.chat_json([]), {"ok": True})
        self.assertEqual(call.call_count, 2)

    def test_content_filter_is_distinct_and_never_retried(self):
        with patch("ark_client.urllib.request.urlopen", return_value=response('Cannot answer', 'content_filter')) as call, patch("ark_client.time.sleep") as sleep:
            with self.assertRaises(ark_client.ArkContentFiltered):
                ark_client.chat_json([])
        self.assertEqual(call.call_count, 1)
        sleep.assert_not_called()
