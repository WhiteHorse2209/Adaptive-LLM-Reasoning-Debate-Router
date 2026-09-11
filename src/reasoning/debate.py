import re
from typing import Any, Dict, List, Optional, Tuple

from src.provider.base import LLMProvider
from src.provider.models import LLMRequest, TokenUsage
from src.reasoning.debate_models import (
    AgentConfig,
    DebateRound,
    DebateTranscript,
    DebateTurn,
)


class DebateAgent:
    """An individual debating agent with distinct persona and reasoning capabilities."""

    def __init__(self, config: AgentConfig, provider: LLMProvider):
        self.config = config
        self.name = config.name
        self.persona = config.persona
        self.provider = provider
        self.temperature = config.temperature
        self.history: List[DebateTurn] = []

    def generate_initial_turn(self, question: str) -> DebateTurn:
        """Solves the question independently in Round 1 with no peer exposure."""
        system_prompt = (
            f"You are {self.name}.\n"
            f"Persona: {self.persona}\n"
            "You are participating in a multi-agent debate to find the absolute truth and correct answer to a question. "
            "In this first round, solve the question completely independently and thoroughly.\n"
            "Format your response strictly as:\n"
            "ARGUMENT: <your step-by-step reasoning and logical deductions>\n"
            "FINAL ANSWER: <concise answer>"
        )

        request = LLMRequest(
            prompt=f"Question: {question}",
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.provider.default_timeout,
        )

        resp = self.provider.generate(request)
        argument, answer = self._parse_initial_turn(resp.text)

        turn = DebateTurn(
            round_number=1,
            agent_name=self.name,
            argument=argument,
            current_answer=answer,
            revised=False,
            token_usage=resp.token_usage,
            latency_seconds=resp.latency_seconds,
        )
        self.history.append(turn)
        return turn

    def generate_critique_turn(
        self,
        question: str,
        round_number: int,
        peer_turns: List[DebateTurn],
    ) -> DebateTurn:
        """Examines peer arguments, challenges invalid assumptions, defends sound reasoning, or revises."""
        own_prev_turn = self.history[-1]
        own_prev_answer = own_prev_turn.current_answer

        # Construct peer transcript
        peer_blocks = []
        for p in peer_turns:
            peer_blocks.append(
                f"--- Peer: {p.agent_name} (Round {p.round_number}) ---\n"
                f"Proposed Answer: {p.current_answer}\n"
                f"Argument:\n{p.argument}"
            )
        peers_text = "\n\n".join(peer_blocks)

        system_prompt = (
            f"You are {self.name}.\n"
            f"Persona: {self.persona}\n"
            f"You are in Round {round_number} of a multi-agent debate.\n"
            f"Your previous answer was: '{own_prev_answer}'.\n\n"
            f"Peer debaters have presented the following arguments:\n"
            f"{peers_text}\n\n"
            "Instructions:\n"
            "1. Scrutinize your peer debaters' arguments for logical fallacies, unsupported assumptions, or mathematical errors.\n"
            "2. If a peer makes a valid counterargument that exposes a genuine flaw in your own previous reasoning, be intellectually honest, concede the error, and revise your answer.\n"
            "3. If your previous reasoning was sound, defend it against their objections with clear, robust counter-evidence.\n"
            "4. Conclude with whether you revised your answer (YES or NO) and your current final answer.\n\n"
            "Format your response strictly as:\n"
            "CRITIQUE & REBUTTAL: <your detailed critique of peers and defense or revision>\n"
            "ANSWER REVISED: <YES or NO>\n"
            "FINAL ANSWER: <your current final answer>"
        )

        request = LLMRequest(
            prompt=f"Original Question: {question}",
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.provider.default_timeout,
        )

        resp = self.provider.generate(request)
        argument, revised, answer = self._parse_critique_turn(resp.text, own_prev_answer)

        turn = DebateTurn(
            round_number=round_number,
            agent_name=self.name,
            argument=argument,
            current_answer=answer,
            revised=revised,
            token_usage=resp.token_usage,
            latency_seconds=resp.latency_seconds,
        )
        self.history.append(turn)
        return turn

    def _parse_initial_turn(self, text: str) -> Tuple[str, str]:
        """Extracts argument and answer from Round 1 response."""
        ans_match = re.search(r"FINAL ANSWER:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if ans_match:
            answer = ans_match.group(1).strip()
            arg_match = re.search(r"ARGUMENT:\s*(.*?)(?=FINAL ANSWER:|$)", text, re.IGNORECASE | re.DOTALL)
            argument = arg_match.group(1).strip() if arg_match else text[:ans_match.start()].strip()
            return argument or text.strip(), answer

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return text.strip(), lines[-1] if lines else text.strip()

    def _parse_critique_turn(self, text: str, own_prev_answer: str) -> Tuple[str, bool, str]:
        """Extracts critique argument, revision status, and current answer from Round 2+ response."""
        ans_match = re.search(r"FINAL ANSWER:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        answer = ans_match.group(1).strip() if ans_match else ""

        revised_match = re.search(r"ANSWER REVISED:\s*(YES|NO)", text, re.IGNORECASE)
        if revised_match:
            explicit_revised = revised_match.group(1).upper() == "YES"
        else:
            explicit_revised = False

        critique_match = re.search(
            r"CRITIQUE & REBUTTAL:\s*(.*?)(?=\n(?:ANSWER REVISED|FINAL ANSWER):|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        argument = critique_match.group(1).strip() if critique_match else text.strip()

        if not answer:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            answer = lines[-1] if lines else own_prev_answer

        # Double check revision by comparing canonicalized answers
        norm_prev = self._normalize(own_prev_answer)
        norm_curr = self._normalize(answer)
        actually_changed = norm_prev != norm_curr
        revised = explicit_revised or actually_changed

        return argument, revised, answer

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text.lower()).strip()


class MultiAgentDebateEngine:
    """Orchestrates genuine multi-round debates between multiple agents."""

    def __init__(self, provider: LLMProvider, debate_config: Optional[Dict[str, Any]] = None):
        self.provider = provider
        self.config = debate_config or {}
        self.num_rounds = int(self.config.get("num_rounds", 2))
        self.agent_temperature = float(self.config.get("agent_temperature", 0.7))

        raw_agents = self.config.get("agents")
        if raw_agents:
            self.agent_configs = [
                AgentConfig(
                    name=a.get("name", f"Agent_{idx}"),
                    persona=a.get("persona", "Analytical reasoning agent."),
                    temperature=a.get("temperature", self.agent_temperature),
                )
                for idx, a in enumerate(raw_agents, 1)
            ]
        else:
            self.agent_configs = [
                AgentConfig(
                    name="Agent_A",
                    persona="Analytical logician focused on rigorous first-principles derivation.",
                    temperature=self.agent_temperature,
                ),
                AgentConfig(
                    name="Agent_B",
                    persona="Critical empirical thinker focused on edge cases and counterexamples.",
                    temperature=self.agent_temperature,
                ),
            ]

    def run_debate(self, question: str, num_rounds: Optional[int] = None) -> DebateTranscript:
        """Executes the complete multi-round debate."""
        rounds_to_run = num_rounds or self.num_rounds
        if rounds_to_run < 1:
            raise ValueError("num_rounds must be at least 1")

        agents = [DebateAgent(cfg, self.provider) for cfg in self.agent_configs]
        rounds: List[DebateRound] = []

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency = 0.0

        # --- Round 1: Independent Discovery ---
        r1_turns: List[DebateTurn] = []
        for agent in agents:
            turn = agent.generate_initial_turn(question)
            r1_turns.append(turn)
            total_prompt_tokens += turn.token_usage.prompt_tokens
            total_completion_tokens += turn.token_usage.completion_tokens
            total_latency += turn.latency_seconds

        rounds.append(DebateRound(round_number=1, turns=r1_turns))

        # --- Round 2+: Cross-Examination and Revision ---
        for r in range(2, rounds_to_run + 1):
            prev_round = rounds[-1]
            r_turns: List[DebateTurn] = []

            for agent in agents:
                # Get all peer turns from the previous round
                peer_turns = [t for t in prev_round.turns if t.agent_name != agent.name]
                turn = agent.generate_critique_turn(question, r, peer_turns)
                r_turns.append(turn)

                total_prompt_tokens += turn.token_usage.prompt_tokens
                total_completion_tokens += turn.token_usage.completion_tokens
                total_latency += turn.latency_seconds

            rounds.append(DebateRound(round_number=r, turns=r_turns))

        # Final Round Analysis
        final_round = rounds[-1]
        final_agent_answers = {t.agent_name: t.current_answer for t in final_round.turns}

        # Check Consensus
        norm_answers = [DebateAgent._normalize(t.current_answer) for t in final_round.turns]
        consensus_reached = len(set(norm_answers)) == 1 if norm_answers else False
        consensus_answer = final_round.turns[0].current_answer if consensus_reached else None

        total_tokens = TokenUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        )

        cost = self.provider.calculate_cost(total_tokens)
        total_calls = len(agents) * rounds_to_run

        return DebateTranscript(
            question=question,
            rounds=rounds,
            final_agent_answers=final_agent_answers,
            consensus_reached=consensus_reached,
            consensus_answer=consensus_answer,
            total_calls=total_calls,
            total_token_usage=total_tokens,
            total_latency_seconds=round(total_latency, 4),
            external_api_cost=cost,
            metadata={
                "num_agents": len(agents),
                "num_rounds": rounds_to_run,
                "agent_names": [a.name for a in agents],
            },
        )
