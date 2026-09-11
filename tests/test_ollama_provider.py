import unittest
from unittest.mock import patch, MagicMock
import requests

from src.provider.models import LLMRequest, TokenUsage
from src.provider.ollama import OllamaProvider
from src.provider.factory import get_provider
from src.reasoning.direct import DirectReasoner


class TestOllamaProvider(unittest.TestCase):
    def setUp(self):
        self.config = {
            "provider": "ollama",
            "model": "llama3.2:latest",
            "temperature": 0.2,
            "max_tokens": 100,
            "timeout": 30,
            "api_base": "http://localhost:11434",
        }
        self.provider = OllamaProvider(self.config)

    def test_provider_factory(self):
        provider = get_provider(self.config)
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.default_model, "llama3.2:latest")

    def test_provider_factory_invalid(self):
        with self.assertRaises(ValueError):
            get_provider({"provider": "unsupported_cloud_xyz"})

    @patch("requests.get")
    def test_health_check_healthy(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        self.assertTrue(self.provider.health_check())

    @patch("requests.get")
    def test_health_check_unhealthy(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        self.assertFalse(self.provider.health_check())

    @patch("requests.post")
    def test_generate_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "25 multiplied by 4 is 100.\nFINAL ANSWER: 100",
            "done": True,
            "prompt_eval_count": 18,
            "eval_count": 14,
            "total_duration": 500000000,
        }
        mock_post.return_value = mock_resp

        request = LLMRequest(prompt="What is 25 * 4?")
        response = self.provider.generate(request)

        self.assertEqual(response.model, "llama3.2:latest")
        self.assertIn("FINAL ANSWER: 100", response.text)
        self.assertEqual(response.token_usage.prompt_tokens, 18)
        self.assertEqual(response.token_usage.completion_tokens, 14)
        self.assertEqual(response.token_usage.total_tokens, 32)
        self.assertGreaterEqual(response.latency_seconds, 0.0)
        self.assertEqual(response.metadata["provider"], "ollama")
        self.assertEqual(response.metadata["external_api_cost"], 0.0)

    @patch("requests.post")
    def test_generate_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Refused")
        request = LLMRequest(prompt="Test prompt")

        with self.assertRaises(ConnectionError):
            self.provider.generate(request)

    @patch("requests.post")
    def test_generate_timeout_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Timed out")
        request = LLMRequest(prompt="Test prompt", timeout=5)

        with self.assertRaises(TimeoutError):
            self.provider.generate(request)


class TestDirectReasoner(unittest.TestCase):
    def test_parse_response_with_marker(self):
        provider_mock = MagicMock()
        reasoner = DirectReasoner(provider_mock)
        text = "Step 1: Compute 25 * 4.\nStep 2: 25 * 4 = 100.\nFINAL ANSWER: 100"
        answer, explanation = reasoner._parse_response(text)
        self.assertEqual(answer, "100")
        self.assertIn("Step 1", explanation)

    def test_parse_response_fallback(self):
        provider_mock = MagicMock()
        reasoner = DirectReasoner(provider_mock)
        text = "The capital of France is Paris."
        answer, explanation = reasoner._parse_response(text)
        self.assertEqual(answer, "The capital of France is Paris.")
        self.assertEqual(explanation, "The capital of France is Paris.")

    def test_direct_reasoner_full_flow(self):
        provider_mock = MagicMock()
        provider_mock.provider_name = "ollama"
        provider_mock.default_temperature = 0.1
        provider_mock.default_max_tokens = 500
        provider_mock.default_timeout = 60
        provider_mock.calculate_cost.return_value = 0.0

        mock_resp = MagicMock()
        mock_resp.text = "Because 2 + 2 equals 4.\nFINAL ANSWER: 4"
        mock_resp.model = "llama3.2:latest"
        mock_resp.token_usage = TokenUsage(prompt_tokens=10, completion_tokens=8, total_tokens=18)
        mock_resp.latency_seconds = 0.25
        mock_resp.metadata = {"provider": "ollama"}
        provider_mock.generate.return_value = mock_resp

        reasoner = DirectReasoner(provider_mock)
        result = reasoner.answer("What is 2 + 2?")

        self.assertEqual(result.answer, "4")
        self.assertEqual(result.model, "llama3.2:latest")
        self.assertEqual(result.token_usage.total_tokens, 18)
        self.assertEqual(result.external_api_cost, 0.0)


if __name__ == "__main__":
    unittest.main()
