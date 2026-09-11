import unittest
from unittest.mock import MagicMock, patch
import requests

from src.provider.exceptions import (
    LLMProviderError,
    MalformedOutputError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from src.provider.models import LLMRequest, LLMResponse, TokenUsage
from src.provider.ollama import OllamaProvider
from src.router.models import DifficultyLevel, ReasoningStrategy
from src.router.router import AdaptiveRouter
from src.utils.retry import retry_with_exponential_backoff


class TestRobustnessEngineering(unittest.TestCase):
    def test_retry_success_after_transient_failure(self):
        mock_func = MagicMock()
        mock_func.side_effect = [
            requests.exceptions.ConnectionError("Temporary blip"),
            requests.exceptions.Timeout("Temporary lag"),
            {"response": "success", "done": True},
        ]

        decorated = retry_with_exponential_backoff(
            max_retries=3,
            initial_delay=0.01,
            jitter=False,
        )(mock_func)

        result = decorated()
        self.assertEqual(result["response"], "success")
        self.assertEqual(mock_func.call_count, 3)

    def test_retry_non_retryable_abort_immediately(self):
        mock_func = MagicMock()
        mock_func.side_effect = OllamaModelNotFoundError("Model xyz does not exist")

        decorated = retry_with_exponential_backoff(
            max_retries=3,
            initial_delay=0.01,
            jitter=False,
        )(mock_func)

        with self.assertRaises(OllamaModelNotFoundError):
            decorated()

        # Must abort immediately without wasting time retrying
        self.assertEqual(mock_func.call_count, 1)

    def test_retry_exhaustion_raises(self):
        mock_func = MagicMock()
        mock_func.side_effect = requests.exceptions.ConnectionError("Permanent outage")

        decorated = retry_with_exponential_backoff(
            max_retries=2,
            initial_delay=0.01,
            jitter=False,
        )(mock_func)

        with self.assertRaises(requests.exceptions.ConnectionError):
            decorated()

        self.assertEqual(mock_func.call_count, 3)  # 1 initial + 2 retries

    @patch("requests.post")
    def test_ollama_provider_model_not_found_404(self, mock_post):
        provider = OllamaProvider({
            "provider": "ollama",
            "model": "non_existent_model:latest",
            "max_retries": 0,
        })
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "model 'non_existent_model:latest' not found, try pulling it first"
        mock_post.return_value = mock_resp

        req = LLMRequest(prompt="Test prompt")
        with self.assertRaises(OllamaModelNotFoundError):
            provider.generate(req)

    @patch("requests.post")
    def test_ollama_provider_malformed_json_response(self, mock_post):
        provider = OllamaProvider({
            "provider": "ollama",
            "model": "llama3.2:latest",
            "max_retries": 0,
        })
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = ["not_a_dictionary"]
        mock_post.return_value = mock_resp

        req = LLMRequest(prompt="Test prompt")
        with self.assertRaises(MalformedOutputError):
            provider.generate(req)

    def test_router_graceful_fallback_on_debate_failure(self):
        mock_provider = MagicMock()
        mock_provider.default_model = "llama3.2:latest"
        mock_provider.provider_name = "ollama"
        mock_provider.calculate_cost.return_value = 0.0

        router = AdaptiveRouter(mock_provider)

        # Initial assessment triggers HARD
        initial_resp = LLMResponse(
            text="EXPLANATION: Hard paradox\nFINAL ANSWER: Fallback answer\nDIFFICULTY: HARD\nCONFIDENCE: 0.20\nCONFIDENCE REASON: Hard",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            latency_seconds=0.1,
        )
        mock_provider.generate.return_value = initial_resp

        # Mock debate pipeline to raise an unhandled provider error
        router.debate_pipeline.run = MagicMock(side_effect=OllamaTimeoutError("Debate agent timed out"))

        # Router must gracefully fall back to DIRECT answer without crashing
        response = router.route_and_solve("Intractable problem", request_id="req_test_123")

        self.assertEqual(response.strategy, ReasoningStrategy.DIRECT)
        self.assertEqual(response.answer, "Fallback answer")
        self.assertTrue(response.metadata["fallback_triggered"])
        self.assertEqual(response.metadata["original_strategy"], "MULTI_AGENT_DEBATE")
        self.assertEqual(response.metadata["request_id"], "req_test_123")
        self.assertIn("Debate agent timed out", response.metadata["fallback_error"])

    def test_router_graceful_fallback_on_self_consistency_failure(self):
        mock_provider = MagicMock()
        mock_provider.default_model = "llama3.2:latest"
        mock_provider.provider_name = "ollama"
        mock_provider.calculate_cost.return_value = 0.0

        router = AdaptiveRouter(mock_provider)

        # Initial assessment triggers UNCERTAIN
        initial_resp = LLMResponse(
            text="EXPLANATION: Trick question\nFINAL ANSWER: 5 cents\nDIFFICULTY: UNCERTAIN\nCONFIDENCE: 0.65\nCONFIDENCE REASON: Tricky",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            latency_seconds=0.1,
        )
        mock_provider.generate.return_value = initial_resp

        # Mock consistency reasoner to raise error
        router.consistency_reasoner.sample_and_vote = MagicMock(
            side_effect=OllamaConnectionError("Ollama crashed during sampling")
        )

        response = router.route_and_solve("Bat and ball query", request_id="req_test_456")

        self.assertEqual(response.strategy, ReasoningStrategy.DIRECT)
        self.assertEqual(response.answer, "5 cents")
        self.assertTrue(response.metadata["fallback_triggered"])
        self.assertEqual(response.metadata["original_strategy"], "SELF_CONSISTENCY")
        self.assertEqual(response.metadata["request_id"], "req_test_456")


if __name__ == "__main__":
    unittest.main()
