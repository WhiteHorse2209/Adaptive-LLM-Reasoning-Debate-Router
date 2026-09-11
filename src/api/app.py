"""FastAPI application providing REST endpoints for adaptive reasoning."""

import uuid
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import HealthApiResponse, ReasonApiRequest, ReasonApiResponse
from src.config.config import load_config
from src.provider.exceptions import (
    LLMProviderError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from src.provider.factory import get_provider
from src.reasoning.consistency import SelfConsistencyReasoner
from src.reasoning.direct import DirectReasoner
from src.reasoning.judge import DebateWithJudgePipeline
from src.router.router import AdaptiveRouter
from src.utils.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Adaptive LLM Reasoning & Debate Router API",
    description="Production-grade API exposing dynamic adaptive reasoning, self-consistency consensus, and multi-agent debate adjudication.",
    version="1.0.0",
)

# CORS middleware for open accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline_components(profile_name: Optional[str] = None):
    """Initializes provider and router based on current configuration and optional profile override."""
    config = load_config()
    profile = config.get_active_profile(profile_name)
    provider = get_provider(profile)
    router_config = config.raw_config.get("router", {})
    debate_config = config.raw_config.get("debate", {})
    judge_config = config.raw_config.get("judge", {})

    router = AdaptiveRouter(
        provider=provider,
        router_config=router_config,
        debate_config=debate_config,
        judge_config=judge_config,
    )
    return config, profile, provider, router, debate_config


@app.get("/health", response_model=HealthApiResponse)
def health():
    """Health check endpoint evaluating provider connectivity."""
    try:
        config, profile, provider, _, _ = get_pipeline_components()
        is_healthy = provider.health_check()
        return HealthApiResponse(
            status="healthy" if is_healthy else "degraded",
            provider=provider.provider_name,
            default_model=provider.default_model,
            provider_healthy=is_healthy,
            active_profile=config.active_profile,
        )
    except Exception as e:
        logger.error(f"Health check failed with error: {e}")
        return HealthApiResponse(
            status="unhealthy",
            provider="unknown",
            default_model="unknown",
            provider_healthy=False,
            active_profile="unknown",
        )


@app.post("/reason", response_model=ReasonApiResponse)
def reason(request: ReasonApiRequest):
    """Executes reasoning over a user query via adaptive routing or explicit mode override."""
    req_id = uuid.uuid4().hex[:8]
    logger.info(f"[{req_id}] Incoming request: mode='{request.mode}', question='{request.question[:60]}...'")

    try:
        config, profile, provider, router, debate_config = get_pipeline_components(request.profile)
    except Exception as e:
        logger.error(f"[{req_id}] Failed to initialize provider components: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration/Provider initialization failed: {e}",
        )

    mode = (request.mode or "adaptive").lower()

    try:
        if mode == "direct":
            reasoner = DirectReasoner(provider)
            res = reasoner.answer(
                request.question,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            return ReasonApiResponse(
                request_id=req_id,
                question=request.question,
                answer=res.answer,
                explanation=res.explanation,
                strategy="DIRECT",
                difficulty="MANUAL_OVERRIDE",
                confidence_score=1.0,
                model=res.model,
                provider=provider.provider_name,
                call_count=1,
                token_usage={
                    "prompt_tokens": res.token_usage.prompt_tokens,
                    "completion_tokens": res.token_usage.completion_tokens,
                    "total_tokens": res.token_usage.total_tokens,
                },
                latency_seconds=res.latency_seconds,
                external_api_cost=res.external_api_cost,
                metadata={"mode": "direct_override"},
            )

        elif mode == "self_consistency":
            reasoner = SelfConsistencyReasoner(provider)
            samples = int(config.raw_config.get("router", {}).get("consistency_samples", 3))
            sample_temp = request.temperature if request.temperature is not None else float(config.raw_config.get("router", {}).get("sample_temperature", 0.7))
            sc_res = reasoner.sample_and_vote(
                question=request.question,
                num_samples=samples,
                temperature=sample_temp,
                max_tokens=request.max_tokens,
            )
            return ReasonApiResponse(
                request_id=req_id,
                question=request.question,
                answer=sc_res.final_answer,
                explanation=sc_res.final_explanation,
                strategy="SELF_CONSISTENCY",
                difficulty="MANUAL_OVERRIDE",
                confidence_score=sc_res.agreement_score,
                model=provider.default_model,
                provider=provider.provider_name,
                call_count=sc_res.num_samples,
                token_usage={
                    "prompt_tokens": sc_res.token_usage.prompt_tokens,
                    "completion_tokens": sc_res.token_usage.completion_tokens,
                    "total_tokens": sc_res.token_usage.total_tokens,
                },
                latency_seconds=sc_res.latency_seconds,
                external_api_cost=provider.calculate_cost(sc_res.token_usage),
                self_consistency_info={
                    "agreement_score": sc_res.agreement_score,
                    "agreement_distribution": sc_res.agreement_distribution,
                    "candidates": [
                        {"sample_id": c.sample_id, "answer": c.answer, "explanation": c.explanation}
                        for c in sc_res.candidates
                    ],
                },
                metadata={"mode": "self_consistency_override"},
            )

        elif mode == "debate":
            deb_cfg = dict(debate_config)
            if request.rounds is not None:
                deb_cfg["num_rounds"] = request.rounds
            pipeline = DebateWithJudgePipeline(provider, debate_config=deb_cfg)
            deb_res = pipeline.run(request.question)
            return ReasonApiResponse(
                request_id=req_id,
                question=request.question,
                answer=deb_res.final_answer,
                explanation=deb_res.explanation,
                strategy="MULTI_AGENT_DEBATE",
                difficulty="MANUAL_OVERRIDE",
                confidence_score=deb_res.verdict.confidence_in_verdict,
                model=provider.default_model,
                provider=provider.provider_name,
                call_count=deb_res.total_calls,
                token_usage={
                    "prompt_tokens": deb_res.total_token_usage.prompt_tokens,
                    "completion_tokens": deb_res.total_token_usage.completion_tokens,
                    "total_tokens": deb_res.total_token_usage.total_tokens,
                },
                latency_seconds=deb_res.total_latency_seconds,
                external_api_cost=deb_res.external_api_cost,
                debate_info={
                    "rounds": [
                        {
                            "round_number": r.round_number,
                            "turns": [
                                {
                                    "agent_name": t.agent_name,
                                    "argument": t.argument,
                                    "answer": t.current_answer,
                                    "revised": t.revised,
                                }
                                for t in r.turns
                            ],
                        }
                        for r in deb_res.transcript.rounds
                    ],
                    "consensus_reached": deb_res.transcript.consensus_reached,
                    "verdict": {
                        "winning_agent": deb_res.verdict.winning_agent,
                        "verdict_answer": deb_res.verdict.verdict_answer,
                        "evaluation_summary": deb_res.verdict.evaluation_summary,
                        "confidence_in_verdict": deb_res.verdict.confidence_in_verdict,
                        "identified_flaws": deb_res.verdict.identified_flaws,
                    },
                },
                metadata={"mode": "debate_override"},
            )

        else:
            # Adaptive Mode (Default)
            routed = router.route_and_solve(
                question=request.question,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                request_id=req_id,
            )

            debate_info = None
            if routed.transcript and routed.verdict:
                debate_info = {
                    "rounds": [
                        {
                            "round_number": r.round_number,
                            "turns": [
                                {
                                    "agent_name": t.agent_name,
                                    "argument": t.argument,
                                    "answer": t.current_answer,
                                    "revised": t.revised,
                                }
                                for t in r.turns
                            ],
                        }
                        for r in routed.transcript.rounds
                    ],
                    "consensus_reached": routed.transcript.consensus_reached,
                    "verdict": {
                        "winning_agent": routed.verdict.winning_agent,
                        "verdict_answer": routed.verdict.verdict_answer,
                        "evaluation_summary": routed.verdict.evaluation_summary,
                        "confidence_in_verdict": routed.verdict.confidence_in_verdict,
                        "identified_flaws": routed.verdict.identified_flaws,
                    },
                }

            sc_info = None
            if routed.self_consistency_result:
                sc = routed.self_consistency_result
                sc_info = {
                    "agreement_score": sc.agreement_score,
                    "agreement_distribution": sc.agreement_distribution,
                    "candidates": [
                        {"sample_id": c.sample_id, "answer": c.answer, "explanation": c.explanation}
                        for c in sc.candidates
                    ],
                }

            return ReasonApiResponse(
                request_id=req_id,
                question=request.question,
                answer=routed.answer,
                explanation=routed.explanation,
                strategy=routed.strategy.value,
                difficulty=routed.difficulty.value,
                confidence_score=routed.confidence.score,
                model=routed.model,
                provider=routed.provider,
                call_count=routed.call_count,
                token_usage={
                    "prompt_tokens": routed.token_usage.prompt_tokens,
                    "completion_tokens": routed.token_usage.completion_tokens,
                    "total_tokens": routed.token_usage.total_tokens,
                },
                latency_seconds=routed.latency_seconds,
                external_api_cost=routed.external_api_cost,
                debate_info=debate_info,
                self_consistency_info=sc_info,
                metadata=routed.metadata,
            )

    except OllamaConnectionError as e:
        logger.error(f"[{req_id}] Ollama connection failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Local Ollama server is unavailable. Ensure 'ollama serve' is running: {e}",
        )
    except OllamaModelNotFoundError as e:
        logger.error(f"[{req_id}] Ollama model not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configured model is not found in local Ollama: {e}",
        )
    except LLMProviderError as e:
        logger.error(f"[{req_id}] LLM provider error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM Provider error: {e}",
        )
    except Exception as e:
        logger.exception(f"[{req_id}] Unexpected server error during reasoning: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal reasoning engine error: {e}",
        )
