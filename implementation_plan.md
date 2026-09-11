# Phase 4 Implementation Plan: Self-Consistency Reasoning

## Goal Description
Implement **Mode 2 — Self-Consistency** using the local Ollama provider. For queries identified by the router as `UNCERTAIN`, the system generates multiple independent candidate reasoning paths at non-zero temperature, compares candidate answers, performs majority voting to select the most consistent answer, and calculates the consensus/agreement metric. The router will be updated so that:
- **EASY** -> DIRECT (1 call)
- **UNCERTAIN** -> SELF-CONSISTENCY ($N$ samples, consensus voting)
- **HARD** -> MULTI-AGENT DEBATE (Flagged for Phase 5)

## Key Concepts & Mathematical Foundation
1. **Self-Consistency Sampling (Wang et al., 2022)**:
   - For an uncertain problem $q$, sample $N$ independent generation paths $G = \{g_1, g_2, \dots, g_N\}$ using sampling temperature $T > 0$ (default $0.7$).
   - Extract candidate answers $A = \{a_1, a_2, \dots, a_N\}$.
   - Normalize answers (casing, numeric formatting, whitespace).
   - Select the consensus answer by marginalizing over paths:
     $$a^* = \arg\max_{a} \sum_{i=1}^N \mathbb{I}(a_i = a)$$
2. **Agreement Score**:
   $$\text{Agreement} = \frac{\max_a \text{count}(a)}{N}$$
   Measures confidence in the selected candidate answer (e.g. 3/3 = 1.0, 2/3 = 0.67, 4/5 = 0.80).
3. **Telemetry & Compute Tracking**:
   - `call_count`: $1 \text{ (initial evaluation)} + N \text{ (sampling calls)} = 1 + N$.
   - Cumulative token usage: $\sum \text{tokens}$.
   - Cumulative latency: $\sum \text{latency}$.
   - Zero external API cost ($0.00) on local hardware.

---

## Proposed Changes

### 1. Self-Consistency Models
#### [NEW] `src/reasoning/models.py`
- `CandidateSolution`:
  - `sample_id: int`
  - `answer: str`
  - `explanation: str`
  - `token_usage: TokenUsage`
  - `latency_seconds: float`
- `SelfConsistencyResult`:
  - `final_answer: str`
  - `final_explanation: str`
  - `agreement_score: float`
  - `agreement_distribution: Dict[str, int]`
  - `candidates: List[CandidateSolution]`
  - `num_samples: int`
  - `model: str`
  - `provider: str`
  - `token_usage: TokenUsage`
  - `latency_seconds: float`
  - `external_api_cost: float = 0.0`

### 2. Self-Consistency Reasoner
#### [NEW] `src/reasoning/consistency.py`
- `SelfConsistencyReasoner`:
  - `sample_and_vote(question: str, num_samples: int = 3, temperature: float = 0.7) -> SelfConsistencyResult`
  - `_normalize_answer(answer: str) -> str`: Normalizes punctuation, numerical representations (e.g. "100" vs "100.0"), and casing.
  - Generates $N$ diverse solutions in parallel/sequence.
  - Tallies votes and resolves ties systematically.
  - Aggregates tokens, latency, and candidate solutions.

### 3. Adaptive Router Integration
#### [MODIFY] `src/router/router.py`
- Inject `SelfConsistencyReasoner` into `AdaptiveRouter`.
- When decision is `UNCERTAIN`:
  - Automatically executes `self_consistency_reasoner.sample_and_vote(question, num_samples)`.
  - Sets `strategy = ReasoningStrategy.SELF_CONSISTENCY`.
  - Aggregates call count: $1 + N$.
  - Aggregates initial and sampling tokens and latency.
  - Records candidate distribution and agreement score into response telemetry.

### 4. Configuration Updates
#### [MODIFY] `config.json`
- Add router parameters:
  ```json
  "router": {
    "confidence_threshold_high": 0.80,
    "confidence_threshold_low": 0.50,
    "estimation_mode": "single_pass",
    "consistency_samples": 3,
    "sample_temperature": 0.7
  }
  ```

### 5. CLI & Display
#### [MODIFY] `src/main.py`
- Display candidate solutions, agreement percentage, and sample breakdown when `SELF_CONSISTENCY` is executed.

### 6. Automated Testing
#### [NEW] `tests/test_consistency.py`
- Unit tests for:
  - Majority voting with unanimous candidates (100% agreement).
  - Majority voting with divided candidates (e.g. 2 vs 1 -> 66.7% agreement).
  - Tie-breaking behavior.
  - Answer normalization (e.g. whitespace, trailing periods, case insensitivity).
  - Token accumulation and call counting.
  - Router integration test for `UNCERTAIN` queries triggering self-consistency.

### 7. Documentation & Repository Sync
#### [MODIFY] `implementation_plan.md`
- Attach Phase 4 plan directly in repository root.
#### [MODIFY] `README.md`
- Update with Phase 4 Self-Consistency architecture, consensus mechanism, configuration, and sample outputs.
- Git commit and push to remote: `feat(phase-4): implement self-consistency reasoning and router integration`.

---

## Verification Plan

### Automated Tests
- Run `.\venv\Scripts\python.exe -m pytest tests/ -v` to ensure all existing (20) and new unit tests pass (100% pass rate).

### Manual Verification
1. **Easy Question (Direct Route)**:
   - Run: `python -m src.main -q "What is 25 * 4?" --profile local_fast`
   - Verify: `Strategy = DIRECT`, `Calls Made = 1`.
2. **Uncertain Question (Self-Consistency Route)**:
   - Run: `python -m src.main -q "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?" --profile local_fast`
   - Verify: `Strategy = SELF_CONSISTENCY`, Candidate solutions generated, agreement score calculated, `Calls Made = 1 + N`, final consensus answer = `5 cents` (or 0.05).
