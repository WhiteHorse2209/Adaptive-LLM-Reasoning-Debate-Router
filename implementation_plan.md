# Dual-Phase Implementation Plan: Phase 6 (Debate Judge) & Phase 7 (Complete Adaptive Reasoning Engine)

## Goal Description
Implement the two culminating reasoning phases of the project:
1. **Phase 6 — Debate Judge**: A separate impartial arbiter that reviews the full debate history, analyzes argument strength, logical consistency, and peer rebuttals, and issues an authoritative final answer with explanation (avoiding naive majority voting).
2. **Phase 7 — Complete Adaptive Reasoning Engine**: The full synthesis of the system where the `AdaptiveRouter` dynamically routes across all three reasoning modes:
   - **Mode 1 (EASY)** -> Direct (1 call)
   - **Mode 2 (UNCERTAIN)** -> Self-Consistency ($1 + N$ samples, consensus voting)
   - **Mode 3 (HARD)** -> Multi-Agent Debate + Judge ($1 + N_{\text{agents}} \times N_{\text{rounds}} + 1$ calls)

As requested, both phases will be implemented in this session, with **individual git commits per phase**.

---

## Phase 6: Debate Judge

### 1. Data Models
#### [NEW] `src/reasoning/judge_models.py`
- `JudgeVerdict`:
  - `verdict_answer: str`
  - `confidence_in_verdict: float` (0.0 - 1.0)
  - `winning_agent: Optional[str]` ("Agent_A", "Agent_B", or "SYNTHESIS")
  - `evaluation_summary: str`
  - `identified_flaws: List[str]`
  - `token_usage: TokenUsage`
  - `latency_seconds: float`
- `DebatePipelineResult`:
  - `question: str`
  - `final_answer: str`
  - `explanation: str`
  - `transcript: DebateTranscript`
  - `verdict: JudgeVerdict`
  - `total_calls: int` ($N_{\text{debate}} + 1$)
  - `total_token_usage: TokenUsage`
  - `total_latency_seconds: float`
  - `external_api_cost: float = 0.0`

### 2. Debate Judge Engine
#### [NEW] `src/reasoning/judge.py`
- `DebateJudge`:
  - Receives complete `DebateTranscript`.
  - Prompts model with structured adjudication rubric (logical consistency, evidence, responsiveness to counterarguments, avoidance of dogmatic doubling-down).
  - Evaluates argument merits and determines the winning argument or synthesized truth.
- `DebateWithJudgePipeline`:
  - Chains `MultiAgentDebateEngine` into `DebateJudge`.

### 3. Unit Tests & Commit for Phase 6
#### [NEW] `tests/test_judge.py`
- Tests judge parsing, verdict selection, flaw detection, and full debate-to-judge pipeline.
- Git commit message: `feat(phase-6): implement debate judge and adjudication pipeline`.

---

## Phase 7: Complete Adaptive Reasoning Engine

### 1. Adaptive Router Full Integration
#### [MODIFY] `src/router/router.py`
- Full routing dispatch:
  - When query is `HARD` or low confidence:
    - Runs `DebateWithJudgePipeline`.
    - Returns verdict answer, debate transcript, and judge critique.
  - When query is `UNCERTAIN`:
    - Runs `SelfConsistencyReasoner`.
  - When query is `EASY`:
    - Runs `DirectReasoner` (1 call).

### 2. Router Models Update
#### [MODIFY] `src/router/models.py`
- Enhance `RoutedResponse` to optionally embed `DebateTranscript`, `JudgeVerdict`, and `SelfConsistencyResult`.

### 3. CLI & Application Integration
#### [MODIFY] `src/main.py`
- Unified CLI displaying rich telemetry for all three active paths.

### 4. Unit Tests & Commit for Phase 7
#### [NEW] `tests/test_complete_pipeline.py`
- Verifies that all 3 modes trigger properly, accumulate tokens, track call counts, and return structured telemetry.
- Git commit message: `feat(phase-7): implement complete adaptive reasoning engine integrating direct, self-consistency, and debate+judge`.

---

## Verification Plan

### Automated Tests
- Run `pytest tests/ -v` to ensure 100% test pass rate across all suites (30+ tests).

### Manual Verification
1. **Easy Question -> DIRECT**:
   `python -m src.main -q "What is 15 + 25?" --profile local_fast` (Expect 1 call, DIRECT)
2. **Uncertain Question -> SELF_CONSISTENCY**:
   `python -m src.main -q "Which number is larger: 9.11 or 9.9?" --profile local_fast` (Expect 4 calls, consensus voting)
3. **Hard Question -> MULTI_AGENT_DEBATE + JUDGE**:
   `python -m src.main -q "Resolve the paradox of Schrödinger's cat in the context of the Many-Worlds interpretation versus the Copenhagen interpretation." --profile local_fast` (Expect full debate + judge verdict)
