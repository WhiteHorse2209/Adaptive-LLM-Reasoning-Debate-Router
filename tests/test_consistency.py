import unittest
from unittest.mock import MagicMock

from src.provider.models import LLMResponse, TokenUsage
from src.reasoning.consistency import SelfConsistencyReasoner
from src.router.models import DifficultyLevel, ReasoningStrategy
from src.router.router import AdaptiveRouter


class TestSelfConsistencyReasoner(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.mock_provider.default_model = "llama3.2:latest"
        self.mock_provider.provider_name = "ollama"
        self.mock_provider.default_temperature = 0.7
        self.mock_provider.default_max_tokens = 500
        self.mock_provider.default_timeout = 60
        self.mock_provider.calculate_cost.return_value = 0.0
        self.reasoner = SelfConsistencyReasoner(self.mock_provider)

    def test_normalize_answer(self):
        self.assertEqual(self.reasoner.normalize_answer("  42  "), "42")
        self.assertEqual(self.reasoner.normalize_answer("100."), "100")
        self.assertEqual(self.reasoner.normalize_answer("$5.00"), "5")
        self.assertEqual(self.reasoner.normalize_answer("Paris!"), "paris")
        self.assertEqual(self.reasoner.normalize_answer("42.0"), "42")
        self.assertEqual(self.reasoner.normalize_answer("3.14"), "3.14")

    def test_sample_and_vote_unanimous(self):
        responses = [
            LLMResponse(
                text="Reasoning 1.\nFINAL ANSWER: 42",
                model="llama3.2:latest",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
                latency_seconds=0.2,
            ),
            LLMResponse(
                text="Reasoning 2.\nFINAL ANSWER: 42",
                model="llama3.2:latest",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=12, total_tokens=22),
                latency_seconds=0.2,
            ),
            LLMResponse(
                text="Reasoning 3.\nFINAL ANSWER: 42",
                model="llama3.2:latest",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=14, total_tokens=24),
                latency_seconds=0.2,
            ),
        ]
        self.mock_provider.generate.side_effect = responses

        result = self.reasoner.sample_and_vote("What is the answer to life?", num_samples=3)

        self.assertEqual(result.final_answer, "42")
        self.assertEqual(result.agreement_score, 1.0)
        self.assertEqual(result.agreement_distribution, {"42": 3})
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(result.token_usage.total_tokens, 71)
        self.assertAlmostEqual(result.latency_seconds, 0.6, places=2)
        self.assertEqual(result.external_api_cost, 0.0)

    def test_sample_and_vote_majority(self):
        responses = [
            LLMResponse(
                text="Method A.\nFINAL ANSWER: 42",
                model="llama3.2:latest",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                latency_seconds=0.1,
            ),
            LLMResponse(
                text="Method B.\nFINAL ANSWER: 40",
                model="llama3.2:latest",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                latency_seconds=0.1,
            ),
            LLMResponse(
                text="Method C.\nFINAL ANSWER: 42",
                model="llama3.2:latest",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                latency_seconds=0.1,
            ),
        ]
        self.mock_provider.generate.side_effect = responses

        result = self.reasoner.sample_and_vote("Estimate value", num_samples=3)

        self.assertEqual(result.final_answer, "42")
        self.assertAlmostEqual(result.agreement_score, 0.667, places=3)
        self.assertEqual(result.agreement_distribution, {"42": 2, "40": 1})
        self.assertEqual(result.token_usage.total_tokens, 60)

    def test_sample_and_vote_invalid_samples(self):
        with self.assertRaises(ValueError):
            self.reasoner.sample_and_vote("Question", num_samples=0)


class TestRouterWithSelfConsistency(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.mock_provider.default_model = "llama3.2:latest"
        self.mock_provider.provider_name = "ollama"
        self.mock_provider.default_temperature = 0.1
        self.mock_provider.default_max_tokens = 500
        self.mock_provider.default_timeout = 60
        self.mock_provider.calculate_cost.return_value = 0.0

        self.router_config = {
            "confidence_threshold_high": 0.80,
            "confidence_threshold_low": 0.50,
            "consistency_samples": 3,
            "sample_temperature": 0.7,
        }
        self.router = AdaptiveRouter(self.mock_provider, self.router_config)

    def test_router_activates_self_consistency_for_uncertain(self):
        # 1. Estimator initial pass returns UNCERTAIN
        initial_resp = LLMResponse(
            text=(
                "EXPLANATION: Multiple plausible answers depending on interpretation.\n"
                "FINAL ANSWER: 5 cents\n"
                "DIFFICULTY: UNCERTAIN\n"
                "CONFIDENCE: 0.65\n"
                "CONFIDENCE REASON: Intuition can mislead."
            ),
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=40, completion_tokens=30, total_tokens=70),
            latency_seconds=0.5,
            metadata={"provider": "ollama"},
        )

        # 2. Self-consistency samples (3 samples)
        sample1 = LLMResponse(
            text="Bat + ball = 1.10. Bat = ball + 1.00. 2*ball = 0.10. Ball = 0.05.\nFINAL ANSWER: 5 cents",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=30, completion_tokens=25, total_tokens=55),
            latency_seconds=0.3,
            metadata={"provider": "ollama"},
        )
        sample2 = LLMResponse(
            text="Bat costs 1 dollar more, so ball is 10 cents.\nFINAL ANSWER: 10 cents",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=30, completion_tokens=20, total_tokens=50),
            latency_seconds=0.25,
            metadata={"provider": "ollama"},
        )
        sample3 = LLMResponse(
            text="Let x be ball price. x + (x + 1) = 1.10 -> 2x = 0.10 -> x = 0.05.\nFINAL ANSWER: 5 cents",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=30, completion_tokens=25, total_tokens=55),
            latency_seconds=0.3,
            metadata={"provider": "ollama"},
        )

        self.mock_provider.generate.side_effect = [initial_resp, sample1, sample2, sample3]

        result = self.router.route_and_solve(
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
        )

        self.assertEqual(result.strategy, ReasoningStrategy.SELF_CONSISTENCY)
        self.assertEqual(result.difficulty, DifficultyLevel.UNCERTAIN)
        self.assertEqual(result.answer, "5 cents")
        self.assertEqual(result.call_count, 4)  # 1 initial + 3 samples
        self.assertAlmostEqual(result.metadata["agreement_score"], 0.667, places=3)
        self.assertEqual(result.metadata["agreement_distribution"], {"5 cents": 2, "10 cents": 1})
        self.assertEqual(result.token_usage.total_tokens, 70 + 55 + 50 + 55)  # 230
        self.assertAlmostEqual(result.latency_seconds, 0.5 + 0.3 + 0.25 + 0.3, places=2)

    def test_router_avoids_self_consistency_for_easy(self):
        # 1. Estimator initial pass returns EASY
        initial_resp = LLMResponse(
            text=(
                "EXPLANATION: 25 * 4 = 100\n"
                "FINAL ANSWER: 100\n"
                "DIFFICULTY: EASY\n"
                "CONFIDENCE: 0.95\n"
                "CONFIDENCE REASON: Elementary multiplication."
            ),
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=30, completion_tokens=20, total_tokens=50),
            latency_seconds=0.3,
            metadata={"provider": "ollama"},
        )
        self.mock_provider.generate.return_value = initial_resp

        result = self.router.route_and_solve("What is 25 * 4?")

        self.assertEqual(result.strategy, ReasoningStrategy.DIRECT)
        self.assertEqual(result.difficulty, DifficultyLevel.EASY)
        self.assertEqual(result.call_count, 1)  # Only 1 call! 0 samples!
        self.assertNotIn("agreement_score", result.metadata)


if __name__ == "__main__":
    unittest.main()
