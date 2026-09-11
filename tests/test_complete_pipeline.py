import unittest
from unittest.mock import MagicMock

from src.provider.models import LLMResponse, TokenUsage
from src.router.models import DifficultyLevel, ReasoningStrategy
from src.router.router import AdaptiveRouter


class TestCompleteAdaptivePipeline(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.mock_provider.default_model = "llama3.2:latest"
        self.mock_provider.provider_name = "ollama"
        self.mock_provider.default_timeout = 60
        self.mock_provider.calculate_cost.return_value = 0.0

        self.router = AdaptiveRouter(
            self.mock_provider,
            router_config={
                "confidence_threshold_high": 0.80,
                "confidence_threshold_low": 0.50,
                "consistency_samples": 3,
                "sample_temperature": 0.7,
            },
            debate_config={"num_rounds": 2},
        )

    def test_mode_1_direct_execution(self):
        """Mode 1: Easy question should execute in exactly 1 call without extra compute."""
        easy_resp = LLMResponse(
            text="EXPLANATION: 2 + 2 = 4\nFINAL ANSWER: 4\nDIFFICULTY: EASY\nCONFIDENCE: 0.99\nCONFIDENCE REASON: Basic math.",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35),
            latency_seconds=0.2,
        )
        self.mock_provider.generate.return_value = easy_resp

        result = self.router.route_and_solve("What is 2 + 2?")

        self.assertEqual(result.strategy, ReasoningStrategy.DIRECT)
        self.assertEqual(result.difficulty, DifficultyLevel.EASY)
        self.assertEqual(result.call_count, 1)
        self.assertEqual(result.answer, "4")
        self.assertIsNone(result.transcript)
        self.assertIsNone(result.verdict)
        self.assertIsNone(result.self_consistency_result)
        self.assertEqual(result.external_api_cost, 0.0)

    def test_mode_2_self_consistency_execution(self):
        """Mode 2: Uncertain question should trigger self-consistency sampling and majority voting."""
        initial_resp = LLMResponse(
            text="EXPLANATION: Subtle problem.\nFINAL ANSWER: 5\nDIFFICULTY: UNCERTAIN\nCONFIDENCE: 0.60\nCONFIDENCE REASON: Trick question.",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=25, completion_tokens=15, total_tokens=40),
            latency_seconds=0.2,
        )
        sample1 = LLMResponse(text="Work 1\nFINAL ANSWER: 5 cents", model="llama3.2:latest", token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30), latency_seconds=0.1)
        sample2 = LLMResponse(text="Work 2\nFINAL ANSWER: 5 cents", model="llama3.2:latest", token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30), latency_seconds=0.1)
        sample3 = LLMResponse(text="Work 3\nFINAL ANSWER: 10 cents", model="llama3.2:latest", token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30), latency_seconds=0.1)

        self.mock_provider.generate.side_effect = [initial_resp, sample1, sample2, sample3]

        result = self.router.route_and_solve("Bat and ball problem")

        self.assertEqual(result.strategy, ReasoningStrategy.SELF_CONSISTENCY)
        self.assertEqual(result.difficulty, DifficultyLevel.UNCERTAIN)
        self.assertEqual(result.call_count, 4)  # 1 initial + 3 samples
        self.assertEqual(result.answer, "5 cents")
        self.assertIsNotNone(result.self_consistency_result)
        self.assertAlmostEqual(result.metadata["agreement_score"], 0.667, places=3)
        self.assertEqual(result.metadata["agreement_distribution"], {"5 cents": 2, "10 cents": 1})
        self.assertEqual(result.external_api_cost, 0.0)

    def test_mode_3_debate_and_judge_execution(self):
        """Mode 3: Hard question should execute full multi-agent debate and judge adjudication."""
        initial_resp = LLMResponse(
            text="EXPLANATION: Intractable dilemma.\nFINAL ANSWER: Controversial\nDIFFICULTY: HARD\nCONFIDENCE: 0.35\nCONFIDENCE REASON: Conflicting paradigms.",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=30, completion_tokens=15, total_tokens=45),
            latency_seconds=0.25,
        )
        # Debate Round 1 (2 calls)
        r1_a = LLMResponse(text="ARGUMENT: Argument A1\nFINAL ANSWER: Option A", model="llama3.2:latest", token_usage=TokenUsage(prompt_tokens=25, completion_tokens=15, total_tokens=40), latency_seconds=0.2)
        r1_b = LLMResponse(text="ARGUMENT: Argument B1\nFINAL ANSWER: Option B", model="llama3.2:latest", token_usage=TokenUsage(prompt_tokens=25, completion_tokens=15, total_tokens=40), latency_seconds=0.2)
        # Debate Round 2 (2 calls)
        r2_a = LLMResponse(text="CRITIQUE & REBUTTAL: Rebuttal A\nANSWER REVISED: NO\nFINAL ANSWER: Option A", model="llama3.2:latest", token_usage=TokenUsage(prompt_tokens=35, completion_tokens=20, total_tokens=55), latency_seconds=0.2)
        r2_b = LLMResponse(text="CRITIQUE & REBUTTAL: Conceding to A\nANSWER REVISED: YES\nFINAL ANSWER: Option A", model="llama3.2:latest", token_usage=TokenUsage(prompt_tokens=35, completion_tokens=20, total_tokens=55), latency_seconds=0.2)
        # Judge (1 call)
        judge_resp = LLMResponse(
            text=(
                "EVALUATION: Agent A's first-principles derivation was vindicated by Agent B's concession.\n"
                "IDENTIFIED FLAWS: Agent B initially assumed static boundary conditions\n"
                "WINNING AGENT: Agent_A\n"
                "VERDICT CONFIDENCE: 0.98\n"
                "FINAL ANSWER: Option A"
            ),
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=60, completion_tokens=30, total_tokens=90),
            latency_seconds=0.3,
        )

        self.mock_provider.generate.side_effect = [initial_resp, r1_a, r1_b, r2_a, r2_b, judge_resp]

        result = self.router.route_and_solve("Foundational physics controversy")

        self.assertEqual(result.strategy, ReasoningStrategy.MULTI_AGENT_DEBATE)
        self.assertEqual(result.difficulty, DifficultyLevel.HARD)
        self.assertEqual(result.call_count, 6)  # 1 initial + 4 debate + 1 judge
        self.assertEqual(result.answer, "Option A")
        self.assertIsNotNone(result.transcript)
        self.assertIsNotNone(result.verdict)
        self.assertEqual(result.verdict.winning_agent, "Agent_A")
        self.assertEqual(result.verdict.confidence_in_verdict, 0.98)
        self.assertTrue(result.transcript.consensus_reached)
        self.assertEqual(result.external_api_cost, 0.0)


if __name__ == "__main__":
    unittest.main()
