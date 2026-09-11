import unittest
from unittest.mock import MagicMock

from src.provider.models import LLMResponse, TokenUsage
from src.reasoning.debate import DebateAgent, MultiAgentDebateEngine
from src.reasoning.debate_models import AgentConfig, DebateTurn


class TestDebateAgent(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.mock_provider.default_timeout = 60
        self.config = AgentConfig(
            name="Agent_A",
            persona="Analytical logician.",
            temperature=0.7,
        )
        self.agent = DebateAgent(self.config, self.mock_provider)

    def test_generate_initial_turn(self):
        mock_resp = LLMResponse(
            text="ARGUMENT: 25 * 4 is calculated as 25 + 25 + 25 + 25 = 100.\nFINAL ANSWER: 100",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
            latency_seconds=0.3,
        )
        self.mock_provider.generate.return_value = mock_resp

        turn = self.agent.generate_initial_turn("What is 25 * 4?")

        self.assertEqual(turn.round_number, 1)
        self.assertEqual(turn.agent_name, "Agent_A")
        self.assertEqual(turn.current_answer, "100")
        self.assertIn("25 + 25", turn.argument)
        self.assertFalse(turn.revised)
        self.assertEqual(len(self.agent.history), 1)

    def test_generate_critique_turn_with_revision(self):
        # Establish initial turn
        self.agent.history.append(
            DebateTurn(
                round_number=1,
                agent_name="Agent_A",
                argument="Initial guess.",
                current_answer="10 cents",
            )
        )

        peer_turn = DebateTurn(
            round_number=1,
            agent_name="Agent_B",
            argument="If the ball is 10 cents, the bat is 1.10, total 1.20. It must be 5 cents.",
            current_answer="5 cents",
        )

        mock_resp = LLMResponse(
            text=(
                "CRITIQUE & REBUTTAL: Agent B is correct. My algebraic setup was flawed.\n"
                "ANSWER REVISED: YES\n"
                "FINAL ANSWER: 5 cents"
            ),
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=80, completion_tokens=30, total_tokens=110),
            latency_seconds=0.4,
        )
        self.mock_provider.generate.return_value = mock_resp

        critique_turn = self.agent.generate_critique_turn(
            question="Bat and ball problem",
            round_number=2,
            peer_turns=[peer_turn],
        )

        self.assertEqual(critique_turn.round_number, 2)
        self.assertTrue(critique_turn.revised)
        self.assertEqual(critique_turn.current_answer, "5 cents")
        self.assertIn("Agent B is correct", critique_turn.argument)

        # Verify that the peer's argument was passed into the prompt
        call_args = self.mock_provider.generate.call_args[0][0]
        self.assertIn("Agent_B", call_args.system_prompt)
        self.assertIn("If the ball is 10 cents", call_args.system_prompt)


class TestMultiAgentDebateEngine(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.mock_provider.default_model = "llama3.2:latest"
        self.mock_provider.provider_name = "ollama"
        self.mock_provider.default_timeout = 60
        self.mock_provider.calculate_cost.return_value = 0.0

    def test_run_debate_consensus_reached(self):
        # Round 1: Agent A says 10, Agent B says 5
        r1_a = LLMResponse(
            text="ARGUMENT: Bat = 1.00 so ball = 0.10\nFINAL ANSWER: 10 cents",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35),
            latency_seconds=0.2,
        )
        r1_b = LLMResponse(
            text="ARGUMENT: 2x = 0.10, x = 0.05\nFINAL ANSWER: 5 cents",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35),
            latency_seconds=0.2,
        )
        # Round 2: Agent A concedes to 5, Agent B maintains 5
        r2_a = LLMResponse(
            text="CRITIQUE & REBUTTAL: Conceding to Agent B.\nANSWER REVISED: YES\nFINAL ANSWER: 5 cents",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
            latency_seconds=0.25,
        )
        r2_b = LLMResponse(
            text="CRITIQUE & REBUTTAL: Defending my formula.\nANSWER REVISED: NO\nFINAL ANSWER: 5 cents",
            model="llama3.2:latest",
            token_usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
            latency_seconds=0.25,
        )

        self.mock_provider.generate.side_effect = [r1_a, r1_b, r2_a, r2_b]

        engine = MultiAgentDebateEngine(self.mock_provider, debate_config={"num_rounds": 2})
        transcript = engine.run_debate("Bat and ball problem")

        self.assertEqual(transcript.total_calls, 4)  # 2 agents * 2 rounds
        self.assertEqual(len(transcript.rounds), 2)
        self.assertTrue(transcript.consensus_reached)
        self.assertEqual(transcript.consensus_answer, "5 cents")
        self.assertEqual(transcript.final_agent_answers, {"Agent_A": "5 cents", "Agent_B": "5 cents"})
        self.assertEqual(transcript.total_token_usage.total_tokens, 190)
        self.assertAlmostEqual(transcript.total_latency_seconds, 0.9, places=2)

    def test_run_debate_no_consensus(self):
        # Round 1: Agent A says Option 1, Agent B says Option 2
        r1_a = LLMResponse(text="FINAL ANSWER: Option 1", model="llama3.2:latest")
        r1_b = LLMResponse(text="FINAL ANSWER: Option 2", model="llama3.2:latest")
        # Round 2: Both maintain their positions
        r2_a = LLMResponse(text="CRITIQUE & REBUTTAL: Disagree.\nANSWER REVISED: NO\nFINAL ANSWER: Option 1", model="llama3.2:latest")
        r2_b = LLMResponse(text="CRITIQUE & REBUTTAL: Disagree.\nANSWER REVISED: NO\nFINAL ANSWER: Option 2", model="llama3.2:latest")

        self.mock_provider.generate.side_effect = [r1_a, r1_b, r2_a, r2_b]

        engine = MultiAgentDebateEngine(self.mock_provider, debate_config={"num_rounds": 2})
        transcript = engine.run_debate("Controversial ethical query")

        self.assertFalse(transcript.consensus_reached)
        self.assertIsNone(transcript.consensus_answer)
        self.assertEqual(transcript.final_agent_answers, {"Agent_A": "Option 1", "Agent_B": "Option 2"})


if __name__ == "__main__":
    unittest.main()
