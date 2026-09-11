"""Reasoning engines for the Adaptive LLM Router."""
from src.reasoning.direct import DirectReasoner
from src.reasoning.consistency import SelfConsistencyReasoner
from src.reasoning.models import CandidateSolution, SelfConsistencyResult
from src.reasoning.debate_models import (
    AgentConfig,
    DebateTurn,
    DebateRound,
    DebateTranscript,
)
from src.reasoning.debate import DebateAgent, MultiAgentDebateEngine
from src.reasoning.judge_models import JudgeVerdict, DebatePipelineResult
from src.reasoning.judge import DebateJudge, DebateWithJudgePipeline

__all__ = [
    "DirectReasoner",
    "SelfConsistencyReasoner",
    "CandidateSolution",
    "SelfConsistencyResult",
    "AgentConfig",
    "DebateTurn",
    "DebateRound",
    "DebateTranscript",
    "DebateAgent",
    "MultiAgentDebateEngine",
    "JudgeVerdict",
    "DebatePipelineResult",
    "DebateJudge",
    "DebateWithJudgePipeline",
]
