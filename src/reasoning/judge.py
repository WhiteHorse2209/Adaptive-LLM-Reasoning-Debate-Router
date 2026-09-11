import re
from typing import Any, Dict, List, Optional, Tuple

from src.provider.base import LLMProvider
from src.provider.models import LLMRequest, TokenUsage
from src.reasoning.debate import MultiAgentDebateEngine
from src.reasoning.debate_models import DebateTranscript
from src.reasoning.judge_models import DebatePipelineResult, JudgeVerdict


class DebateJudge:
    """Impartial evaluator that reviews debate history and determines the true answer."""

    SYSTEM_PROMPT = (
        "You are an impartial, highly rigorous Supreme Debate Judge. "
        "Your task is to adjudicate a multi-agent debate and issue an authoritative verdict on the original question.\n\n"
        "Rules for Adjudication:\n"
        "1. Do NOT rely on naive majority voting. Multiple agents can share common cognitive biases or fallacies.\n"
        "2. Evaluate the factual, mathematical, and logical correctness of each deduction.\n"
        "3. Scrutinize how agents handled peer critiques in later rounds. Did an agent present valid counterarguments, "
        "or did they double down on an exposed mistake?\n"
        "4. Determine whether one agent's argument is decisively correct, or if a synthesized perspective represents the truth.\n\n"
        "Format your verdict strictly with these sections:\n"
        "EVALUATION: <thorough analytical breakdown evaluating each agent's deductions and rebuttals>\n"
        "IDENTIFIED FLAWS: <comma-separated list of specific errors or fallacies detected in the debate, or 'None'>\n"
        "WINNING AGENT: <Agent_A | Agent_B | SYNTHESIS>\n"
        "VERDICT CONFIDENCE: <number between 0.0 and 1.0>\n"
        "FINAL ANSWER: <concise definitive answer>"
    )

    def __init__(self, provider: LLMProvider, judge_config: Optional[Dict[str, Any]] = None):
        self.provider = provider
        self.config = judge_config or {}
        self.temperature = float(self.config.get("temperature", 0.1))

    def adjudicate(self, transcript: DebateTranscript) -> JudgeVerdict:
        """Evaluates the full debate transcript and renders a verdict."""
        formatted_transcript = self._format_transcript_for_judge(transcript)

        prompt = (
            f"Original Question:\n{transcript.question}\n\n"
            f"Full Debate History:\n{formatted_transcript}\n\n"
            "Issue your authoritative verdict following the required format."
        )

        request = LLMRequest(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=600,
            timeout=self.provider.default_timeout,
        )

        resp = self.provider.generate(request)
        answer, eval_summary, winning_agent, confidence, flaws = self._parse_verdict(resp.text, transcript)

        return JudgeVerdict(
            verdict_answer=answer,
            evaluation_summary=eval_summary,
            winning_agent=winning_agent,
            confidence_in_verdict=confidence,
            identified_flaws=flaws,
            token_usage=resp.token_usage,
            latency_seconds=resp.latency_seconds,
            metadata={"judge_model": resp.model, "provider": resp.metadata.get("provider", "ollama")},
        )

    def _format_transcript_for_judge(self, transcript: DebateTranscript) -> str:
        """Formats the transcript into a clean chronological audit log."""
        lines = []
        for r in transcript.rounds:
            lines.append(f"=== ROUND {r.round_number} ===")
            for t in r.turns:
                rev = " [REVISED FROM PREVIOUS]" if t.revised else ""
                lines.append(f"[{t.agent_name}]{rev}:")
                lines.append(f"Argument: {t.argument}")
                lines.append(f"Proposed Answer: {t.current_answer}\n")
        return "\n".join(lines)

    def _parse_verdict(
        self, text: str, transcript: DebateTranscript
    ) -> Tuple[str, str, Optional[str], float, List[str]]:
        """Extracts verdict fields with robust regex fallbacks."""
        # 1. Final Answer
        ans_match = re.search(r"FINAL ANSWER:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if ans_match:
            answer = ans_match.group(1).strip()
        else:
            # Fallback to consensus answer or first agent answer
            answer = transcript.consensus_answer or list(transcript.final_agent_answers.values())[0]

        # 2. Evaluation Summary
        eval_match = re.search(
            r"EVALUATION:\s*(.*?)(?=\n(?:IDENTIFIED FLAWS|WINNING AGENT|VERDICT CONFIDENCE|FINAL ANSWER):|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        eval_summary = eval_match.group(1).strip() if eval_match else text.strip()

        # 3. Winning Agent
        win_match = re.search(r"WINNING AGENT:\s*([A-Za-z0-9_]+)", text, re.IGNORECASE)
        winning_agent = win_match.group(1).strip() if win_match else "SYNTHESIS"

        # 4. Confidence
        conf_match = re.search(r"VERDICT CONFIDENCE:\s*(\d+(?:\.\d+)?%?)", text, re.IGNORECASE)
        if conf_match:
            raw_c = conf_match.group(1).strip()
            try:
                if raw_c.endswith("%"):
                    confidence = float(raw_c[:-1]) / 100.0
                else:
                    val = float(raw_c)
                    confidence = val / 100.0 if val > 1.0 else val
            except ValueError:
                confidence = 0.90
        else:
            confidence = 0.90
        confidence = max(0.0, min(1.0, round(confidence, 3)))

        # 5. Identified Flaws
        flaws_match = re.search(
            r"IDENTIFIED FLAWS:\s*(.*?)(?=\n(?:WINNING AGENT|VERDICT CONFIDENCE|FINAL ANSWER):|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if flaws_match:
            raw_flaws = flaws_match.group(1).strip()
            if raw_flaws.lower() in ["none", "n/a", "no flaws identified"]:
                flaws = []
            else:
                flaws = [f.strip("- *").strip() for f in raw_flaws.split("\n") if f.strip("- *").strip()]
        else:
            flaws = []

        return answer, eval_summary, winning_agent, confidence, flaws


class DebateWithJudgePipeline:
    """Full end-to-end pipeline: Question -> Agents -> Debate -> Judge -> Final Answer."""

    def __init__(
        self,
        provider: LLMProvider,
        debate_config: Optional[Dict[str, Any]] = None,
        judge_config: Optional[Dict[str, Any]] = None,
    ):
        self.provider = provider
        self.debate_engine = MultiAgentDebateEngine(provider, debate_config=debate_config)
        self.judge = DebateJudge(provider, judge_config=judge_config)

    def run(self, question: str, num_rounds: Optional[int] = None) -> DebatePipelineResult:
        """Executes multi-agent debate followed by judge adjudication."""
        # 1. Run multi-agent debate
        transcript = self.debate_engine.run_debate(question, num_rounds=num_rounds)

        # 2. Adjudicate debate transcript
        verdict = self.judge.adjudicate(transcript)

        # 3. Consolidate telemetry
        total_calls = transcript.total_calls + 1
        total_usage = TokenUsage(
            prompt_tokens=transcript.total_token_usage.prompt_tokens + verdict.token_usage.prompt_tokens,
            completion_tokens=transcript.total_token_usage.completion_tokens + verdict.token_usage.completion_tokens,
            total_tokens=transcript.total_token_usage.total_tokens + verdict.token_usage.total_tokens,
        )
        total_latency = round(transcript.total_latency_seconds + verdict.latency_seconds, 4)
        cost = self.provider.calculate_cost(total_usage)

        return DebatePipelineResult(
            question=question,
            final_answer=verdict.verdict_answer,
            explanation=verdict.evaluation_summary,
            transcript=transcript,
            verdict=verdict,
            total_calls=total_calls,
            total_token_usage=total_usage,
            total_latency_seconds=total_latency,
            external_api_cost=cost,
            metadata={
                "winning_agent": verdict.winning_agent,
                "confidence_in_verdict": verdict.confidence_in_verdict,
                "consensus_reached_in_debate": transcript.consensus_reached,
            },
        )
