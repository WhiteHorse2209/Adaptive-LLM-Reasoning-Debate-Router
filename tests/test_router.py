import unittest
from unittest.mock import MagicMock

from src.provider.models import LLMResponse, TokenUsage
from src.router.estimator import ConfidenceEstimator
from src.router.models import DifficultyLevel, ReasoningStrategy
from src.router.router import AdaptiveRouter


class TestConfidenceEstimator(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.mock_provider.default_temperature = 0.1
        self.mock_provider.default_max_tokens = 500
        self.mock_provider.default_timeout = 60
        self.mock_provider.calculate_cost.return_value = 0.0
        self.estimator = ConfidenceEstimator(self.mock_provider)

    def test_parse_assessment_clean_output(self):
        raw_text = (
            "EXPLANATION: Multiplying 25 by 4 gives 100.\n"
            "FINAL ANSWER: 100\n"
            "DIFFICULTY: EASY\n"
            "CONFIDENCE: 0.98\n"
            "CONFIDENCE REASON: Standard basic arithmetic."
        )
        answer, explanation, assessment = self.estimator.parse_assessment(raw_text)

        self.assertEqual(answer, "100")
        self.assertIn("Multiplying 25 by 4", explanation)
        self.assertEqual(assessment.difficulty, DifficultyLevel.EASY)
        self.assertEqual(assessment.score, 0.98)
        self.assertEqual(assessment.level, "HIGH")
        self.assertEqual(assessment.justification, "Standard basic arithmetic.")

    def test_parse_assessment_percentage_confidence(self):
        raw_text = (
            "EXPLANATION: 10 + 10 = 20\n"
            "FINAL ANSWER: 20\n"
            "DIFFICULTY: EASY\n"
            "CONFIDENCE: 95%\n"
            "CONFIDENCE REASON: Elementary addition."
        )
        answer, explanation, assessment = self.estimator.parse_assessment(raw_text)
        self.assertEqual(assessment.score, 0.95)
        self.assertEqual(assessment.level, "HIGH")

    def test_parse_assessment_uncertain(self):
        raw_text = (
            "EXPLANATION: The outcome depends on unspecified boundary conditions.\n"
            "FINAL ANSWER: Unclear\n"
            "DIFFICULTY: UNCERTAIN\n"
            "CONFIDENCE: 0.65\n"
            "CONFIDENCE REASON: Multiple interpretations exist."
        )
        answer, explanation, assessment = self.estimator.parse_assessment(raw_text)
        self.assertEqual(assessment.difficulty, DifficultyLevel.UNCERTAIN)
        self.assertEqual(assessment.score, 0.65)
        self.assertEqual(assessment.level, "MEDIUM")

    def test_parse_assessment_hard(self):
        raw_text = (
            "EXPLANATION: Intractable optimization problem with exponential complexity.\n"
            "FINAL ANSWER: NP-hard\n"
            "DIFFICULTY: HARD\n"
            "CONFIDENCE: 0.35\n"
            "CONFIDENCE REASON: High uncertainty in approximation bound."
        )
        answer, explanation, assessment = self.estimator.parse_assessment(raw_text)
        self.assertEqual(assessment.difficulty, DifficultyLevel.HARD)
        self.assertEqual(assessment.score, 0.35)
        self.assertEqual(assessment.level, "LOW")

    def test_parse_assessment_fallback_missing_tags(self):
        raw_text = "The capital of Australia is Canberra."
        answer, explanation, assessment = self.estimator.parse_assessment(raw_text)
        self.assertEqual(answer, "The capital of Australia is Canberra.")
        self.assertGreaterEqual(assessment.score, 0.0)
        self.assertLessEqual(assessment.score, 1.0)


class TestAdaptiveRouter(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.mock_provider.provider_name = "ollama"
        self.mock_provider.default_model = "llama3.2:latest"
        self.mock_provider.default_temperature = 0.1
        self.mock_provider.default_max_tokens = 500
        self.mock_provider.default_timeout = 60
        self.mock_provider.calculate_cost.return_value = 0.0

        self.router_config = {
            "confidence_threshold_high": 0.80,
            "confidence_threshold_low": 0.50,
        }
        self.router = AdaptiveRouter(self.mock_provider, self.router_config)

    def test_route_easy_to_direct(self):
        mock_resp = LLMResponse(
            text=(
                "EXPLANATION: 25 * 4 = 100\n"
                "FINAL ANSWER: 100\n"
                "DIFFICULTY: EASY\n"
                "CONFIDENCE: 0.95\n"
                "CONFIDENCE REASON: Trivial math"
            ),
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=40, completion_tokens=25, total_tokens=65),
            latency_seconds=0.45,
            metadata={"provider": "ollama"},
        )
        self.mock_provider.generate.return_value = mock_resp

        result = self.router.route_and_solve("What is 25 * 4?")

        self.assertEqual(result.strategy, ReasoningStrategy.DIRECT)
        self.assertEqual(result.difficulty, DifficultyLevel.EASY)
        self.assertEqual(result.confidence.score, 0.95)
        self.assertEqual(result.call_count, 1)
        self.assertEqual(result.external_api_cost, 0.0)
        self.assertEqual(result.metadata["route_decision"], "DIRECT")
        self.assertEqual(result.metadata["additional_agents_called"], 0)

    def test_route_uncertain_to_self_consistency(self):
        mock_resp = LLMResponse(
            text=(
                "EXPLANATION: Several plausible answers depending on context.\n"
                "FINAL ANSWER: Ambiguous\n"
                "DIFFICULTY: UNCERTAIN\n"
                "CONFIDENCE: 0.65\n"
                "CONFIDENCE REASON: Need sampling"
            ),
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
            latency_seconds=0.55,
            metadata={"provider": "ollama"},
        )
        self.mock_provider.generate.return_value = mock_resp

        result = self.router.route_and_solve("Is light a wave or a particle?")

        self.assertEqual(result.strategy, ReasoningStrategy.SELF_CONSISTENCY)
        self.assertEqual(result.difficulty, DifficultyLevel.UNCERTAIN)
        self.assertEqual(result.confidence.score, 0.65)
        self.assertEqual(result.call_count, 4)  # 1 estimation + 3 samples
        self.assertIn("agreement_score", result.metadata)

    def test_route_hard_to_debate(self):
        mock_resp = LLMResponse(
            text=(
                "EXPLANATION: High dimensional paradox with conflicting axioms.\n"
                "FINAL ANSWER: Undecidable\n"
                "DIFFICULTY: HARD\n"
                "CONFIDENCE: 0.30\n"
                "CONFIDENCE REASON: Requires cross-agent argumentation"
            ),
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=60, completion_tokens=40, total_tokens=100),
            latency_seconds=0.75,
            metadata={"provider": "ollama"},
        )
        self.mock_provider.generate.return_value = mock_resp

        result = self.router.route_and_solve("Resolve Newcomb's Paradox.")

        self.assertEqual(result.strategy, ReasoningStrategy.MULTI_AGENT_DEBATE)
        self.assertEqual(result.difficulty, DifficultyLevel.HARD)
        self.assertEqual(result.confidence.score, 0.30)
        self.assertEqual(result.call_count, 6)  # 1 estimation + 4 debate + 1 judge
        self.assertIsNotNone(result.transcript)
        self.assertIsNotNone(result.verdict)
        self.assertEqual(result.metadata["route_decision"], "MULTI_AGENT_DEBATE")


if __name__ == "__main__":
    unittest.main()
