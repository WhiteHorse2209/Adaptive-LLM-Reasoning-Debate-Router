import unittest
from unittest.mock import MagicMock

from src.evaluation.dataset import (
    BenchmarkItem,
    extract_numerical_answer,
    get_benchmark_dataset,
    is_answer_correct,
)
from src.evaluation.evaluator import BenchmarkEvaluator
from src.provider.models import LLMResponse, TokenUsage
from src.reasoning.debate_models import DebateTranscript
from src.reasoning.judge_models import JudgeVerdict
from src.reasoning.models import CandidateSolution, SelfConsistencyResult
from src.router.models import ConfidenceAssessment, DifficultyLevel, ReasoningStrategy, RoutedResponse


class TestEvaluationFramework(unittest.TestCase):
    def test_extract_numerical_answer(self):
        self.assertEqual(extract_numerical_answer("FINAL ANSWER: 42"), 42.0)
        self.assertEqual(extract_numerical_answer("The result is $21.50."), 21.50)
        self.assertEqual(extract_numerical_answer("Discount is 20%."), 20.0)
        self.assertIsNone(extract_numerical_answer("No numbers here"))

    def test_is_answer_correct(self):
        self.assertTrue(is_answer_correct("FINAL ANSWER: 21", "21"))
        self.assertTrue(is_answer_correct("$5.00", "5"))
        self.assertTrue(is_answer_correct("43.20", "43.2"))
        self.assertTrue(is_answer_correct("pedestrian safety", "Pedestrian Safety"))
        self.assertFalse(is_answer_correct("15", "25"))

    def test_get_benchmark_dataset(self):
        all_items = get_benchmark_dataset()
        self.assertGreaterEqual(len(all_items), 10)

        easy_items = get_benchmark_dataset(difficulty="EASY")
        self.assertTrue(all(item.difficulty_label == "EASY" for item in easy_items))

        arithmetic_items = get_benchmark_dataset(category="arithmetic")
        self.assertTrue(all(item.category == "arithmetic" for item in arithmetic_items))

        limited = get_benchmark_dataset(limit=2)
        self.assertEqual(len(limited), 2)

    def test_evaluator_direct_flow(self):
        mock_provider = MagicMock()
        mock_provider.calculate_cost.return_value = 0.0
        evaluator = BenchmarkEvaluator(mock_provider)

        mock_resp = MagicMock()
        mock_resp.answer = "21"
        mock_resp.token_usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_resp.external_api_cost = 0.0
        evaluator.direct_reasoner.answer = MagicMock(return_value=mock_resp)

        items = [
            BenchmarkItem(id="q1", question="3 * 7?", ground_truth="21", difficulty_label="EASY", category="math"),
            BenchmarkItem(id="q2", question="3 * 7 again?", ground_truth="21", difficulty_label="EASY", category="math"),
        ]

        summary = evaluator.evaluate_direct(items)
        self.assertEqual(summary.total_queries, 2)
        self.assertEqual(summary.correct_count, 2)
        self.assertEqual(summary.accuracy, 1.0)
        self.assertEqual(summary.mean_calls_per_query, 1.0)
        self.assertEqual(summary.mean_tokens_per_query, 15.0)

    def test_evaluator_self_consistency_flow(self):
        mock_provider = MagicMock()
        mock_provider.calculate_cost.return_value = 0.0
        evaluator = BenchmarkEvaluator(mock_provider)

        sc_res = SelfConsistencyResult(
            final_answer="5",
            final_explanation="Consensus",
            agreement_score=1.0,
            agreement_distribution={"5": 3},
            candidates=[],
            num_samples=3,
            model="llama3.2:latest",
            provider="ollama",
            token_usage=TokenUsage(prompt_tokens=30, completion_tokens=15, total_tokens=45),
            latency_seconds=0.3,
        )
        evaluator.consistency_reasoner.sample_and_vote = MagicMock(return_value=sc_res)

        items = [
            BenchmarkItem(id="q1", question="Ball cost?", ground_truth="5", difficulty_label="MEDIUM", category="math"),
        ]

        summary = evaluator.evaluate_self_consistency(items, num_samples=3)
        self.assertEqual(summary.total_queries, 1)
        self.assertEqual(summary.correct_count, 1)
        self.assertEqual(summary.accuracy, 1.0)
        self.assertEqual(summary.mean_calls_per_query, 3.0)

    def test_evaluator_adaptive_flow(self):
        mock_provider = MagicMock()
        mock_provider.calculate_cost.return_value = 0.0
        evaluator = BenchmarkEvaluator(mock_provider)

        resp_direct = RoutedResponse(
            answer="21",
            explanation="Exp 1",
            strategy=ReasoningStrategy.DIRECT,
            difficulty=DifficultyLevel.EASY,
            confidence=ConfidenceAssessment(score=0.95, level="HIGH", difficulty=DifficultyLevel.EASY, justification="Easy"),
            call_count=1,
            model="llama3.2:latest",
            provider="ollama",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_seconds=0.1,
            external_api_cost=0.0,
        )
        resp_debate = RoutedResponse(
            answer="47",
            explanation="Exp 2",
            strategy=ReasoningStrategy.MULTI_AGENT_DEBATE,
            difficulty=DifficultyLevel.HARD,
            confidence=ConfidenceAssessment(score=0.30, level="LOW", difficulty=DifficultyLevel.HARD, justification="Hard"),
            call_count=5,
            model="llama3.2:latest",
            provider="ollama",
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=50, total_tokens=100),
            latency_seconds=0.5,
            external_api_cost=0.0,
        )

        evaluator.router.route_and_solve = MagicMock(side_effect=[resp_direct, resp_debate])

        items = [
            BenchmarkItem(id="q1", question="Easy math", ground_truth="21", difficulty_label="EASY", category="math"),
            BenchmarkItem(id="q2", question="Lily pad", ground_truth="47", difficulty_label="HARD", category="logic"),
        ]

        summary = evaluator.evaluate_adaptive(items)
        self.assertEqual(summary.total_queries, 2)
        self.assertEqual(summary.correct_count, 2)
        self.assertEqual(summary.accuracy, 1.0)
        self.assertEqual(summary.mean_calls_per_query, 3.0)  # (1 + 5) / 2 = 3.0
        self.assertEqual(summary.routing_breakdown["DIRECT"], 50.0)
        self.assertEqual(summary.routing_breakdown["MULTI_AGENT_DEBATE"], 50.0)
        self.assertEqual(summary.routing_breakdown["SELF_CONSISTENCY"], 0.0)


if __name__ == "__main__":
    unittest.main()
