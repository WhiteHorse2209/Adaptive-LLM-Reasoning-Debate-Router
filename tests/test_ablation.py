import csv
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.evaluation.ablation import (
    AblationConfig,
    AblationEngine,
    get_standard_ablation_matrix,
)
from src.evaluation.dataset import BenchmarkItem
from src.provider.models import TokenUsage
from src.reasoning.debate_models import DebateTranscript
from src.router.models import (
    ConfidenceAssessment,
    DifficultyLevel,
    ReasoningStrategy,
    RoutedResponse,
)


class TestAblationStudies(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_standard_ablation_matrix(self):
        matrix = get_standard_ablation_matrix()
        self.assertGreaterEqual(len(matrix), 8)

        experiment_ids = [c.experiment_id for c in matrix]
        self.assertEqual(len(experiment_ids), len(set(experiment_ids)), "Experiment IDs must be unique.")

        # Check that specific ablations exist
        self.assertTrue(any(c.adjudication_type == "majority_voting" for c in matrix))
        self.assertTrue(any(c.debate_num_agents == 3 for c in matrix))
        self.assertTrue(any(c.debate_num_rounds == 1 for c in matrix))
        self.assertTrue(any(c.confidence_threshold_high == 0.90 for c in matrix))

    @patch("src.evaluation.ablation.AdaptiveRouter")
    def test_ablation_single_run(self, mock_router_cls):
        mock_provider = MagicMock()
        mock_provider.calculate_cost.return_value = 0.0

        mock_router_instance = MagicMock()
        routed_resp = RoutedResponse(
            answer="21",
            explanation="7 * 3 = 21",
            strategy=ReasoningStrategy.DIRECT,
            difficulty=DifficultyLevel.EASY,
            confidence=ConfidenceAssessment(score=0.95, level="HIGH", difficulty=DifficultyLevel.EASY, justification="Simple math"),
            call_count=1,
            model="llama3.2:latest",
            provider="ollama",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_seconds=0.2,
            external_api_cost=0.0,
        )
        mock_router_instance.route_and_solve.return_value = routed_resp
        mock_router_cls.return_value = mock_router_instance

        engine = AblationEngine(mock_provider)
        cfg = AblationConfig(
            experiment_id="test_exp",
            name="Test Exp",
            description="Testing ablation",
            confidence_threshold_high=0.80,
            confidence_threshold_low=0.50,
        )
        items = [
            BenchmarkItem(id="q1", question="What is 3*7?", ground_truth="21", difficulty_label="EASY", category="math"),
        ]

        result = engine.run_single_ablation(cfg, items)
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.mean_calls_per_query, 1.0)
        self.assertEqual(result.mean_tokens_per_query, 15.0)
        self.assertGreater(result.efficiency_index, 0.0)
        self.assertEqual(result.routing_breakdown["DIRECT"], 100.0)

    @patch("src.evaluation.ablation.AdaptiveRouter")
    def test_ablation_majority_voting_adjudication(self, mock_router_cls):
        mock_provider = MagicMock()
        mock_provider.calculate_cost.return_value = 0.0

        mock_router_instance = MagicMock()
        mock_transcript = DebateTranscript(
            question="Lily pad query",
            final_agent_answers={"Agent_A": "47", "Agent_B": "47"},
            consensus_reached=True,
            total_calls=4,
        )
        routed_resp = RoutedResponse(
            answer="Wrong Judge Answer",
            explanation="Judge erred",
            strategy=ReasoningStrategy.MULTI_AGENT_DEBATE,
            difficulty=DifficultyLevel.HARD,
            confidence=ConfidenceAssessment(score=0.25, level="LOW", difficulty=DifficultyLevel.HARD, justification="Debate"),
            call_count=5,  # 1 router + 4 debate + 1 judge
            model="llama3.2:latest",
            provider="ollama",
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=50, total_tokens=100),
            latency_seconds=0.5,
            external_api_cost=0.0,
            transcript=mock_transcript,
        )
        mock_router_instance.route_and_solve.return_value = routed_resp
        mock_router_cls.return_value = mock_router_instance

        engine = AblationEngine(mock_provider)
        cfg = AblationConfig(
            experiment_id="exp_no_judge",
            name="Majority Voting Test",
            description="Eliminates judge",
            adjudication_type="majority_voting",
        )
        items = [
            BenchmarkItem(id="q2", question="Lily pad question", ground_truth="47", difficulty_label="HARD", category="logic"),
        ]

        result = engine.run_single_ablation(cfg, items)
        # Agent answers were "47", which matches ground truth "47", despite judge answer being wrong
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.mean_calls_per_query, 4.0)  # 5 - 1 judge call = 4

    @patch("src.evaluation.ablation.AdaptiveRouter")
    def test_ablation_export_files(self, mock_router_cls):
        mock_router_instance = MagicMock()
        routed_resp = RoutedResponse(
            answer="2",
            explanation="1 + 1 = 2",
            strategy=ReasoningStrategy.DIRECT,
            difficulty=DifficultyLevel.EASY,
            confidence=ConfidenceAssessment(score=0.99, level="HIGH", difficulty=DifficultyLevel.EASY, justification="Basic"),
            call_count=1,
            model="llama3.2:latest",
            provider="ollama",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_seconds=0.1,
            external_api_cost=0.0,
        )
        mock_router_instance.route_and_solve.return_value = routed_resp
        mock_router_cls.return_value = mock_router_instance

        cfg = AblationConfig(
            experiment_id="exp_test",
            name="Export Test",
            description="Testing export",
        )
        mock_provider = MagicMock()
        mock_provider.calculate_cost.return_value = 0.0

        run_res = AblationEngine(mock_provider).run_single_ablation(
            cfg,
            [BenchmarkItem(id="q1", question="1+1?", ground_truth="2", difficulty_label="EASY", category="math")],
        )

        json_file = os.path.join(self.test_dir, "test_results.json")
        csv_file = os.path.join(self.test_dir, "test_summary.csv")

        AblationEngine.export_results([run_res], json_path=json_file, csv_path=csv_file)

        self.assertTrue(os.path.exists(json_file))
        self.assertTrue(os.path.exists(csv_file))

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["experiment_id"], "exp_test")

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 2)  # Header + 1 row
            self.assertIn("Efficiency_Index", reader[0])
            self.assertEqual(reader[1][0], "exp_test")


if __name__ == "__main__":
    unittest.main()
