# Phase 3 Implementation Plan: Confidence & Difficulty Estimation + Direct Routing

## Goal Description
Implement confidence and difficulty estimation to dynamically classify queries as `EASY`, `UNCERTAIN`, or `HARD`.
For easy/high-confidence queries, the router routes to `DIRECT` mode, returning the answer immediately with a single model call while tracking confidence, strategy, call count, token usage, and latency. For uncertain or hard queries, the router identifies the necessity for deeper reasoning (preparing the ground for Self-Consistency in Phase 4 and Multi-Agent Debate in Phase 5).

## Key Concepts in Phase 3
1. **Confidence & Difficulty Estimation**:
   - The initial LLM call produces the proposed answer, explanation, and an explicit verbalized confidence score (0.0 to 1.0) with difficulty reasoning.
   - Supports both high-efficiency **single-pass estimation** (answer + confidence in 1 model call) and **two-pass estimation** (solve first, critique/score second).
2. **Configurable Thresholds**:
   - `confidence_threshold_high` (default `0.80`): Score >= threshold -> Difficulty: `EASY` -> Strategy: `DIRECT`.
   - `confidence_threshold_low` (default `0.50`): Score between `0.50` and `0.80` -> Difficulty: `UNCERTAIN` (flagged for Phase 4 Self-Consistency).
   - Score < `0.50` -> Difficulty: `HARD` (flagged for Phase 5 Multi-Agent Debate).
3. **Telemetry & Call Tracking**:
   - Accurately tracks `number_of_calls`, accumulated `token_usage`, `latency_seconds`, `confidence_score`, `strategy`, and `external_api_cost` ($0.00 for local).

---

## Proposed Changes

### 1. Data Models
#### [NEW] `src/router/models.py`
- `DifficultyLevel`: Enum (`EASY`, `UNCERTAIN`, `HARD`).
- `ReasoningStrategy`: Enum (`DIRECT`, `SELF_CONSISTENCY`, `MULTI_AGENT_DEBATE`).
- `ConfidenceAssessment`:
  - `score: float` (0.0 - 1.0)
  - `level: str` ("HIGH", "MEDIUM", "LOW")
  - `difficulty: DifficultyLevel`
  - `justification: str`
- `RoutedResponse`:
  - `answer: str`
  - `explanation: str`
  - `strategy: ReasoningStrategy`
  - `difficulty: DifficultyLevel`
  - `confidence: ConfidenceAssessment`
  - `call_count: int`
  - `model: str`
  - `provider: str`
  - `token_usage: TokenUsage`
  - `latency_seconds: float`
  - `external_api_cost: float`
  - `metadata: Dict[str, Any]`

### 2. Difficulty & Confidence Estimator
#### [NEW] `src/router/estimator.py`
- `ConfidenceEstimator`:
  - Builds structured prompts requesting step-by-step reasoning, concise answer, confidence rating (0.0 to 1.0), and difficulty classification.
  - Robust regex/JSON parsers extracting confidence even from noisy model outputs.
  - Fallback mechanisms ensuring confidence defaults cleanly if parsing fails.

### 3. Adaptive Router Engine
#### [NEW] `src/router/router.py`
- `AdaptiveRouter`:
  - Manages routing thresholds loaded from `config.json`.
  - Determines routing decision:
    - If `score >= threshold_high`: Strategy is `DIRECT`. Returns immediate answer with `call_count = 1`.
    - If `threshold_low <= score < threshold_high`: Difficulty is `UNCERTAIN`, strategy is `SELF_CONSISTENCY` (in Phase 3, flags the recommendation and provides the direct preliminary answer).
    - If `score < threshold_low`: Difficulty is `HARD`, strategy is `MULTI_AGENT_DEBATE` (in Phase 3, flags the recommendation and provides the direct preliminary answer).

### 4. Configuration Updates
#### [MODIFY] `config.json`
- Refine router configuration:
  ```json
  "router": {
    "confidence_threshold_high": 0.80,
    "confidence_threshold_low": 0.50,
    "estimation_mode": "single_pass",
    "consistency_samples": 5
  }
  ```

### 5. CLI & Entry Point Integration
#### [MODIFY] `src/main.py`
- Integrate `AdaptiveRouter` as the primary evaluation pipeline.
- CLI displays `STRATEGY: DIRECT`, `CONFIDENCE: 0.95 (HIGH)`, `DIFFICULTY: EASY`, `CALLS: 1`, etc.

### 6. Automated Testing
#### [NEW] `tests/test_router.py`
- Unit tests for:
  - Confidence parsing and bounding (0.0 - 1.0).
  - Threshold classification (EASY -> DIRECT, UNCERTAIN -> SELF_CONSISTENCY, HARD -> MULTI_AGENT_DEBATE).
  - Call count and token usage accumulation.
  - Router execution with mocked provider.

### 7. Documentation & Repository Sync
#### [MODIFY] `implementation_plan.md`
- Attach Phase 3 implementation plan directly in repository root.
#### [MODIFY] `README.md`
- Update with Phase 3 architecture diagram, router thresholds, confidence estimation details, and examples.
- Git commit and push to remote: `feat(phase-3): implement confidence estimation and adaptive direct routing`.

---

## Verification Plan

### Automated Tests
- Run `.\venv\Scripts\python.exe -m pytest tests/ -v` to ensure all existing and new unit tests pass (100% pass rate).

### Manual Verification
1. **Easy Question (Direct Route)**:
   - Run: `python -m src.main -q "What is 25 * 4?" --profile local_fast`
   - Expectation: Strategy = `DIRECT`, Difficulty = `EASY`, Confidence >= 0.80, `call_count = 1`.
2. **Hard Question (Detected for Deeper Reasoning)**:
   - Run: `python -m src.main -q "Compare the economic implications of Keynesian vs Austrian school during a stagflation crisis with conflicting fiscal policies." --profile local_fast`
   - Expectation: Difficulty = `UNCERTAIN` or `HARD`, Strategy recommendation identified, metrics and confidence reported accurately.
