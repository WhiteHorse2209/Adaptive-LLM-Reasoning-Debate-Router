# Dual-Phase Implementation Plan: Phase 12 (Confidence Calibration & Failure Analysis) & Phase 13 (Cost/Latency/Accuracy Optimization)

## Goal Description
Implement the deep diagnostic, calibration, and optimization engine of the project:
1. **Phase 12 — Confidence Calibration + Failure Analysis**:
   - Quantify verbalized confidence calibration: Do 90% confidence scores actually correspond to 90% correctness?
   - Calculate **Expected Calibration Error (ECE)** and confidence reliability diagrams across probability bins.
   - Comprehensive **8-category Failure Mode Taxonomy**:
     1. `INITIAL_WRONG_DEBATE_CORRECT`: Successful error correction via debate cross-examination.
     2. `INITIAL_WRONG_DEBATE_WRONG`: Intractable fallacy persistent across agents.
     3. `INITIAL_CORRECT_DEBATE_CORRECT`: Stable reasoning preserved through scrutiny.
     4. `INITIAL_CORRECT_DEBATE_WRONG`: Negative peer pressure / faulty concession.
     5. `UNDER_ROUTING`: Router dispatched to Direct on complex problem, leading to error.
     6. `OVER_ROUTING`: Router triggered expensive debate on easy problem, wasting compute.
     7. `JUDGE_SELECTION_ERROR`: Agents debated correctly, but judge favored flawed argument.
     8. `UNANIMOUS_HALLUCINATION`: Both debaters agreed on the same incorrect premise.
2. **Phase 13 — Cost / Latency / Accuracy Optimization**:
   - Multi-objective optimization framework analyzing the Accuracy vs Latency vs Compute trade-off.
   - Pareto frontier identification: Finding the optimal confidence threshold pair $(T_{\text{high}}, T_{\text{low}})$ maximizing accuracy per unit compute.
   - Plotting & reporting utility generating visual ASCII diagrams and metric logs:
     - Accuracy vs Latency
     - Accuracy vs Token Usage
     - Accuracy vs Number of Calls
     - Debate Activation vs Accuracy

As required, both phases will be implemented in this session with **individual git commits per phase**.

---

## Phase 12: Confidence Calibration & Failure Analysis

### 1. Calibration Metrics & ECE
#### [NEW] `src/evaluation/calibration.py`
- `ConfidenceBin`: Bin range, average confidence, empirical accuracy, sample count, calibration gap.
- `CalibrationAnalysis`:
  - Expected Calibration Error (ECE): $\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$.
  - Maximum Calibration Error (MCE).
  - Overconfidence / Underconfidence indicators.

### 2. Failure Mode Taxonomy Engine
#### [NEW] `src/evaluation/failure_analysis.py`
- `FailureMode` Enum (8 distinct diagnostic categories).
- `FailureAnalyzer`: Classifies individual query traces, logs concrete diagnostic examples, and computes distribution percentages.

### 3. CLI Calibration & Failure Runner
#### [NEW] `src/evaluation/run_calibration.py`
- Evaluates benchmark queries, displays reliability diagrams, reports ECE, and outputs failure mode frequencies.

### 4. Unit Tests & Commit for Phase 12
#### [NEW] `tests/test_calibration.py`
- Tests ECE calculation, bin grouping, and 8-category failure classification.
- Git commit message: `feat(phase-12): implement confidence calibration analysis, ECE metric, and 8-category failure taxonomy`.

---

## Phase 13: Cost / Latency / Accuracy Optimization

### 1. Pareto Frontier Optimizer
#### [NEW] `src/evaluation/optimization.py`
- `ParetoOptimizer`:
  - Evaluates threshold candidate pairs $(T_{\text{high}}, T_{\text{low}})$ to find the Pareto non-dominated frontier.
  - Multi-objective fitness function balancing Accuracy against Token and Latency overhead.
  - Recommends the optimal operating configuration based on user priority (e.g. `QUALITY_FIRST`, `BALANCED`, `BUDGET_CONSTRAINED`).

### 2. Optimization Visualizer & Reporter
#### [NEW] `src/evaluation/visualize.py`
- ASCII curve and bar visualizer for terminal and reporting:
  - Accuracy vs Calls
  - Accuracy vs Latency
  - Token Tradeoff
  - Strategy Distribution

### 3. Unit Tests & Commit for Phase 13
#### [NEW] `tests/test_optimization.py`
- Tests Pareto domination sorting, optimal threshold selection, and visualization generation.
- Git commit message: `feat(phase-13): implement multi-objective Pareto optimization and trade-off visualizer`.

---

## Verification Plan

### Automated Tests
- Run `pytest tests/test_calibration.py -v`
- Run `pytest tests/test_optimization.py -v`
- Run `pytest tests/ -v` to ensure 100% pass rate across all 70+ tests.

### Manual Verification
- Execute calibration runner:
  `python -m src.evaluation.run_calibration --limit 5 --profile local_fast`
- Verify optimization recommendations and ASCII Pareto visualization.
