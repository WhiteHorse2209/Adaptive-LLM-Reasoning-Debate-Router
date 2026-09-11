# Dual-Phase Implementation Plan: Phase 10 (Benchmark + Evaluation) & Phase 11 (Experiments + Ablation)

## Goal Description
Implement the quantitative empirical core of the project:
1. **Phase 10 — Benchmark + Evaluation Engine**:
   - Establish an objectively checkable reasoning benchmark (GSM8K arithmetic & logic reasoning dataset with verifiable ground truth).
   - Build an automated evaluation engine comparing 4 system paradigms:
     1. **Single LLM (Direct)**
     2. **Self-Consistency (N-sample majority voting)**
     3. **Always-On Multi-Agent Debate (+ Supreme Judge)**
     4. **Adaptive Router (Confidence/Difficulty-guided dynamic allocation)**
   - Measure and report: Accuracy (Exact Match / canonical numerical equivalence), call counts, token accumulation, wall-clock latency, and routing distribution (% direct, % consistency, % debate).
2. **Phase 11 — Experiments + Ablation Studies**:
   - Systematic experimental framework recording reproducible runs into `experiments/` (JSON & CSV).
   - Ablation studies exploring:
     - Routing threshold sensitivity (High/Low thresholds)
     - 2 vs 3 Debating Agents
     - 1 vs 2 Debate Rounds
     - Judge Adjudication vs Naive Majority Voting in debates
     - Pareto frontier analysis: Accuracy vs Latency vs Compute overhead.

As required, both phases will be implemented in this session with **individual git commits per phase**.

---

## Phase 10: Benchmark + Evaluation Engine

### 1. Benchmark Dataset & Extraction
#### [NEW] `src/evaluation/dataset.py`
- Curated reasoning dataset consisting of verified GSM8K arithmetic and multi-step logic problems across easy, medium, and hard difficulty levels with canonical ground truth answers.
- Normalization and extraction utilities (`extract_numerical_answer`, `normalize_ground_truth`).

### 2. Evaluation Engine
#### [NEW] `src/evaluation/evaluator.py`
- `BenchmarkEvaluator`:
  - Runs queries through all 4 paradigms:
    1. Single LLM Direct
    2. Self-Consistency
    3. Always Debate + Judge
    4. Adaptive Router
  - Calculates accuracy, mean latency, token consumption, call counts, and routing breakdown.
  - Produces structured evaluation reports.

### 3. CLI Evaluation Runner
#### [NEW] `src/evaluation/run_eval.py`
- Executable benchmark script with tabular summary formatting and progress reporting.

### 4. Unit Tests & Commit for Phase 10
#### [NEW] `tests/test_evaluation.py`
- Tests dataset loading, answer extraction, scoring logic, and evaluator flow with mocks.
- Git commit message: `feat(phase-10): implement benchmark evaluation engine with GSM8K reasoning subset and 4-strategy comparator`.

---

## Phase 11: Experiments + Ablation Studies

### 1. Experiment Runner & Artifact Storage
#### [NEW] `src/evaluation/ablation.py`
- `AblationEngine`:
  - Threshold sensitivity ablation:
    - Standard thresholds (0.80 / 0.50)
    - Aggressive debate thresholds (0.90 / 0.70)
    - Cost-saver direct thresholds (0.70 / 0.30)
  - Debate depth ablation: 1 Round vs 2 Rounds vs 3 Rounds.
  - Agent multiplicity: 2 Agents vs 3 Agents.
  - Adjudication ablation: Judge vs simple agent majority voting.
- Exports structured experiment runs into `experiments/results_*.json` and `experiments/ablation_summary.csv`.

### 2. Unit Tests & Commit for Phase 11
#### [NEW] `tests/test_ablation.py`
- Tests ablation parameter grid generation, execution capture, and summary export.
- Git commit message: `feat(phase-11): implement ablation study framework covering thresholds, debate rounds, agent counts, and judge arbitration`.

---

## Verification Plan

### Automated Tests
- Run `pytest tests/test_evaluation.py -v`
- Run `pytest tests/test_ablation.py -v`
- Run `pytest tests/ -v` to ensure 100% pass rate across all 60+ tests.

### Manual Verification
- Execute benchmark evaluation runner:
  `python -m src.evaluation.run_eval --limit 3 --profile local_fast`
- Verify generated experiment JSON files in `experiments/`.
