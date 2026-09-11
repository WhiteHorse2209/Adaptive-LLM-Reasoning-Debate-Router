import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.app import app
from src.provider.exceptions import OllamaConnectionError
from src.provider.models import LLMResponse, TokenUsage
from src.reasoning.debate_models import DebateRound, DebateTranscript, DebateTurn
from src.reasoning.judge_models import JudgeVerdict
from src.reasoning.models import CandidateSolution, SelfConsistencyResult
from src.router.models import ConfidenceAssessment, DifficultyLevel, ReasoningStrategy, RoutedResponse


class TestFastAPIRoutes(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("src.api.app.get_pipeline_components")
    def test_health_endpoint_healthy(self, mock_get_components):
        mock_provider = MagicMock()
        mock_provider.health_check.return_value = True
        mock_provider.provider_name = "ollama"
        mock_provider.default_model = "llama3.2:latest"
        mock_config = MagicMock()
        mock_config.active_profile = "local_fast"

        mock_get_components.return_value = (mock_config, {}, mock_provider, None, {})

        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["provider_healthy"])
        self.assertEqual(data["provider"], "ollama")
        self.assertEqual(data["default_model"], "llama3.2:latest")

    @patch("src.api.app.get_pipeline_components")
    def test_health_endpoint_degraded(self, mock_get_components):
        mock_provider = MagicMock()
        mock_provider.health_check.return_value = False
        mock_provider.provider_name = "ollama"
        mock_provider.default_model = "llama3.2:latest"
        mock_config = MagicMock()
        mock_config.active_profile = "local_fast"

        mock_get_components.return_value = (mock_config, {}, mock_provider, None, {})

        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "degraded")
        self.assertFalse(data["provider_healthy"])

    @patch("src.api.app.DirectReasoner")
    @patch("src.api.app.get_pipeline_components")
    def test_reason_mode_direct_override(self, mock_get_components, mock_reasoner_cls):
        mock_provider = MagicMock()
        mock_provider.provider_name = "ollama"
        mock_provider.default_model = "llama3.2:latest"

        mock_reasoner = MagicMock()
        mock_resp = MagicMock()
        mock_resp.answer = "100"
        mock_resp.explanation = "25 * 4 = 100"
        mock_resp.model = "llama3.2:latest"
        mock_resp.token_usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_resp.latency_seconds = 0.15
        mock_resp.external_api_cost = 0.0
        mock_reasoner.answer.return_value = mock_resp
        mock_reasoner_cls.return_value = mock_reasoner

        mock_config = MagicMock()
        mock_config.raw_config = {}
        mock_get_components.return_value = (mock_config, {}, mock_provider, None, {})

        resp = self.client.post("/reason", json={"question": "What is 25 * 4?", "mode": "direct"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["answer"], "100")
        self.assertEqual(data["strategy"], "DIRECT")
        self.assertEqual(data["call_count"], 1)
        self.assertEqual(data["token_usage"]["total_tokens"], 15)

    @patch("src.api.app.SelfConsistencyReasoner")
    @patch("src.api.app.get_pipeline_components")
    def test_reason_mode_self_consistency_override(self, mock_get_components, mock_sc_cls):
        mock_provider = MagicMock()
        mock_provider.provider_name = "ollama"
        mock_provider.default_model = "llama3.2:latest"
        mock_provider.calculate_cost.return_value = 0.0

        mock_sc_instance = MagicMock()
        mock_sc_res = SelfConsistencyResult(
            final_answer="5 cents",
            final_explanation="Consensus solution",
            agreement_score=0.67,
            agreement_distribution={"5 cents": 2, "10 cents": 1},
            candidates=[
                CandidateSolution(sample_id=1, reasoning_chain="Chain 1", answer="5 cents", explanation="Exp 1", token_usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10), latency_seconds=0.1),
            ],
            num_samples=3,
            model="llama3.2:latest",
            provider="ollama",
            token_usage=TokenUsage(prompt_tokens=15, completion_tokens=15, total_tokens=30),
            latency_seconds=0.3,
        )
        mock_sc_instance.sample_and_vote.return_value = mock_sc_res
        mock_sc_cls.return_value = mock_sc_instance

        mock_config = MagicMock()
        mock_config.raw_config = {"router": {"consistency_samples": 3, "sample_temperature": 0.7}}
        mock_get_components.return_value = (mock_config, {}, mock_provider, None, {})

        resp = self.client.post("/reason", json={"question": "Bat and ball", "mode": "self_consistency"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["answer"], "5 cents")
        self.assertEqual(data["strategy"], "SELF_CONSISTENCY")
        self.assertIsNotNone(data["self_consistency_info"])
        self.assertEqual(data["self_consistency_info"]["agreement_distribution"], {"5 cents": 2, "10 cents": 1})

    @patch("src.api.app.DebateWithJudgePipeline")
    @patch("src.api.app.get_pipeline_components")
    def test_reason_mode_debate_override(self, mock_get_components, mock_pipeline_cls):
        mock_provider = MagicMock()
        mock_provider.provider_name = "ollama"
        mock_provider.default_model = "llama3.2:latest"

        mock_pipeline = MagicMock()
        mock_pipeline_res = MagicMock()
        mock_pipeline_res.final_answer = "Option A"
        mock_pipeline_res.explanation = "Debate and judge explanation"
        mock_pipeline_res.total_calls = 5
        mock_pipeline_res.total_token_usage = TokenUsage(prompt_tokens=50, completion_tokens=50, total_tokens=100)
        mock_pipeline_res.total_latency_seconds = 1.2
        mock_pipeline_res.external_api_cost = 0.0

        t1 = DebateTurn(round_number=1, agent_name="Agent_A", argument="Arg A", current_answer="Option A", token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20), latency_seconds=0.1)
        r1 = DebateRound(round_number=1, turns=[t1])
        mock_pipeline_res.transcript = DebateTranscript(question="Q", rounds=[r1], final_agent_answers={"Agent_A": "Option A"}, consensus_reached=True, total_calls=1, total_tokens=20)
        mock_pipeline_res.verdict = JudgeVerdict(
            verdict_answer="Option A",
            confidence_in_verdict=0.95,
            winning_agent="Agent_A",
            evaluation_summary="Rigorous logic",
            identified_flaws=[],
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            latency_seconds=0.2,
        )

        mock_pipeline.run.return_value = mock_pipeline_res
        mock_pipeline_cls.return_value = mock_pipeline

        mock_config = MagicMock()
        mock_config.raw_config = {"debate": {"num_rounds": 2}}
        mock_get_components.return_value = (mock_config, {}, mock_provider, None, {"num_rounds": 2})

        resp = self.client.post("/reason", json={"question": "Hard dilemma", "mode": "debate"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["answer"], "Option A")
        self.assertEqual(data["strategy"], "MULTI_AGENT_DEBATE")
        self.assertIsNotNone(data["debate_info"])
        self.assertEqual(data["debate_info"]["verdict"]["winning_agent"], "Agent_A")

    @patch("src.api.app.get_pipeline_components")
    def test_reason_mode_adaptive(self, mock_get_components):
        mock_provider = MagicMock()
        mock_provider.provider_name = "ollama"
        mock_provider.default_model = "llama3.2:latest"

        mock_router = MagicMock()
        routed_resp = RoutedResponse(
            answer="Paris",
            explanation="Capital of France",
            strategy=ReasoningStrategy.DIRECT,
            difficulty=DifficultyLevel.EASY,
            confidence=ConfidenceAssessment(score=0.98, level="HIGH", difficulty=DifficultyLevel.EASY, justification="Common knowledge"),
            call_count=1,
            model="llama3.2:latest",
            provider="ollama",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_seconds=0.1,
            external_api_cost=0.0,
            metadata={},
        )
        mock_router.route_and_solve.return_value = routed_resp

        mock_config = MagicMock()
        mock_config.raw_config = {}
        mock_get_components.return_value = (mock_config, {}, mock_provider, mock_router, {})

        resp = self.client.post("/reason", json={"question": "What is the capital of France?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["answer"], "Paris")
        self.assertEqual(data["strategy"], "DIRECT")
        self.assertEqual(data["difficulty"], "EASY")

    @patch("src.api.app.get_pipeline_components")
    def test_reason_ollama_unavailable_raises_503(self, mock_get_components):
        mock_provider = MagicMock()
        mock_provider.provider_name = "ollama"
        mock_router = MagicMock()
        mock_router.route_and_solve.side_effect = OllamaConnectionError("Connection refused")

        mock_config = MagicMock()
        mock_config.raw_config = {}
        mock_get_components.return_value = (mock_config, {}, mock_provider, mock_router, {})

        resp = self.client.post("/reason", json={"question": "Test query"})
        self.assertEqual(resp.status_code, 503)
        self.assertIn("unavailable", resp.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
