import unittest
from unittest.mock import MagicMock

from src.provider.models import LLMResponse, TokenUsage
from src.reasoning.debate_models import DebateRound, DebateTranscript, DebateTurn
from src.reasoning.judge import DebateJudge, DebateWithJudgePipeline


class TestDebateJudge(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.mock_provider.default_model = "llama3.2:latest"
        self.mock_provider.provider_name = "ollama"
        self.mock_provider.default_timeout = 60
        self.mock_provider.calculate_cost.return_value = 0.0
        self.judge = DebateJudge(self.mock_provider)

    def test_parse_verdict_complete(self):
        transcript = DebateTranscript(
            question="What is the capital of Australia?",
            rounds=[],
            final_agent_answers={"Agent_A": "Sydney", "Agent_B": "Canberra"},
            consensus_reached=False,
        )

        raw_verdict_text = (
            "EVALUATION: Agent A erroneously picked the largest city (Sydney). "
            "Agent B correctly cited the federal capital (Canberra) based on the 1908 compromise.\n"
            "IDENTIFIED FLAWS:\n"
            "- Agent A confused economic prominence with political capital status\n"
            "WINNING AGENT: Agent_B\n"
            "VERDICT CONFIDENCE: 0.98\n"
            "FINAL ANSWER: Canberra"
        )

        answer, eval_summary, winner, conf, flaws = self.judge._parse_verdict(raw_verdict_text, transcript)

        self.assertEqual(answer, "Canberra")
        self.assertIn("Agent A erroneously picked", eval_summary)
        self.assertEqual(winner, "Agent_B")
        self.assertEqual(conf, 0.98)
        self.assertEqual(len(flaws), 1)
        self.assertIn("Agent A confused economic prominence", flaws[0])

    def test_parse_verdict_percentage_and_no_flaws(self):
        transcript = DebateTranscript(
            question="What is 2 + 2?",
            rounds=[],
            final_agent_answers={"Agent_A": "4", "Agent_B": "4"},
            consensus_reached=True,
            consensus_answer="4",
        )

        raw_verdict_text = (
            "EVALUATION: Both agents derived 4 rigorously.\n"
            "IDENTIFIED FLAWS: None\n"
            "WINNING AGENT: SYNTHESIS\n"
            "VERDICT CONFIDENCE: 100%\n"
            "FINAL ANSWER: 4"
        )

        answer, eval_summary, winner, conf, flaws = self.judge._parse_verdict(raw_verdict_text, transcript)
        self.assertEqual(answer, "4")
        self.assertEqual(conf, 1.0)
        self.assertEqual(flaws, [])
        self.assertEqual(winner, "SYNTHESIS")

    def test_adjudicate_call(self):
        transcript = DebateTranscript(
            question="Which is greater: 9.11 or 9.9?",
            rounds=[
                DebateRound(
                    round_number=1,
                    turns=[
                        DebateTurn(
                            round_number=1,
                            agent_name="Agent_A",
                            argument="9.11 > 9.9 because 11 > 9.",
                            current_answer="9.11",
                            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                        ),
                        DebateTurn(
                            round_number=1,
                            agent_name="Agent_B",
                            argument="9.90 > 9.11 because 9 tenths > 1 tenth.",
                            current_answer="9.9",
                            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                        ),
                    ],
                )
            ],
            final_agent_answers={"Agent_A": "9.11", "Agent_B": "9.9"},
            consensus_reached=False,
            total_calls=2,
        )

        mock_resp = LLMResponse(
            text=(
                "EVALUATION: Agent A fell into the integer trap comparing 11 to 9 without place value. "
                "Agent B correctly evaluated place value (0.9 vs 0.11).\n"
                "IDENTIFIED FLAWS:\n"
                "- Failure to align decimal place values\n"
                "WINNING AGENT: Agent_B\n"
                "VERDICT CONFIDENCE: 0.99\n"
                "FINAL ANSWER: 9.9"
            ),
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=40, total_tokens=90),
            latency_seconds=0.45,
            metadata={"provider": "ollama"},
        )
        self.mock_provider.generate.return_value = mock_resp

        verdict = self.judge.adjudicate(transcript)

        self.assertEqual(verdict.verdict_answer, "9.9")
        self.assertEqual(verdict.winning_agent, "Agent_B")
        self.assertEqual(verdict.confidence_in_verdict, 0.99)
        self.assertEqual(verdict.token_usage.total_tokens, 90)
        self.assertEqual(verdict.latency_seconds, 0.45)


class TestDebateWithJudgePipeline(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.mock_provider.default_model = "llama3.2:latest"
        self.mock_provider.provider_name = "ollama"
        self.mock_provider.default_timeout = 60
        self.mock_provider.calculate_cost.return_value = 0.0

    def test_pipeline_execution(self):
        # 4 responses for 2 agents * 2 rounds debate + 1 response for Judge
        resp_a1 = LLMResponse(text="ARGUMENT: Arg A1\nFINAL ANSWER: Option A", model="llama3.2:latest")
        resp_b1 = LLMResponse(text="ARGUMENT: Arg B1\nFINAL ANSWER: Option B", model="llama3.2:latest")
        resp_a2 = LLMResponse(text="CRITIQUE & REBUTTAL: Rebuttal A\nANSWER REVISED: NO\nFINAL ANSWER: Option A", model="llama3.2:latest")
        resp_b2 = LLMResponse(text="CRITIQUE & REBUTTAL: Rebuttal B\nANSWER REVISED: NO\nFINAL ANSWER: Option B", model="llama3.2:latest")
        resp_judge = LLMResponse(
            text=(
                "EVALUATION: Agent B provided superior empirical proof.\n"
                "IDENTIFIED FLAWS: Agent A lacked counter-evidence\n"
                "WINNING AGENT: Agent_B\n"
                "VERDICT CONFIDENCE: 0.95\n"
                "FINAL ANSWER: Option B"
            ),
            model="llama3.2:latest",
        )

        self.mock_provider.generate.side_effect = [resp_a1, resp_b1, resp_a2, resp_b2, resp_judge]

        pipeline = DebateWithJudgePipeline(
            self.mock_provider,
            debate_config={"num_rounds": 2},
        )
        result = pipeline.run("Complex philosophical dilemma")

        self.assertEqual(result.total_calls, 5)  # 4 debate calls + 1 judge call
        self.assertEqual(result.final_answer, "Option B")
        self.assertEqual(result.verdict.winning_agent, "Agent_B")
        self.assertEqual(len(result.transcript.rounds), 2)


if __name__ == "__main__":
    unittest.main()
