# Comprehensive Architectural & Codebase Guide (Phases 1–14)

This document provides an exhaustive, authoritative breakdown of the entire **Adaptive LLM Reasoning & Debate Router** codebase through **Phase 14**:
1. **Phase-by-Phase Implementation History**: What was built, why, and key metrics.
2. **Directory & Architectural Hierarchy**: The organizational structure and role of each folder.
3. **Comprehensive File Manifest**: The exact purpose, key classes, functions, and responsibilities of every file.

---

## 1. Phase-by-Phase Build Summary (Phases 1 to 14)

| Phase | Title | Commit Hash | Core Deliverables & Milestones |
|---|---|---|---|
| **Phase 1** | Architecture + Foundation | `4e85408` | Git remote setup, initial JSON configuration (`config.json`), project skeleton, requirements definition, foundation tests. |
| **Phase 2** | Ollama Provider + Single LLM | `9af2abb` | Standardized `LLMProvider` interface, `OllamaProvider` with streaming/HTTP, `DirectReasoner`, CLI entrypoint (`src/main.py`), 12 unit tests. |
| **Phase 3** | Confidence & Difficulty Estimation + Direct Routing | `5c809b3` | `ConfidenceEstimator` with XML extraction tags (`<difficulty>`, `<confidence>`, `<justification>`), `AdaptiveRouter` Mode 1 (DIRECT) routing, 20 unit tests. |
| **Phase 4** | Self-Consistency & Majority Voting | `b2a9c7e` | `SelfConsistencyReasoner` (Wang et al., 2022) sampling $N$ paths at $T>0$, exact numerical & text normalization, consensus voting, agreement scores, 26 unit tests. |
| **Phase 5** | Multi-Agent Debate Foundation | `0471e29` | `DebateAgent` with distinct personas, `MultiAgentDebateEngine` managing Round 1 positions and Round 2 cross-examinations, consensus detection, 30 unit tests. |
| **Phase 6** | Debate Judge & Adjudication Pipeline | `8ce1e41` | `DebateJudge` acting as Supreme Judge, fallacy identification, winning agent selection, `DebateWithJudgePipeline` unifying debate + arbitration, 34 unit tests. |
| **Phase 7** | Complete Adaptive Reasoning Engine | `ea7383c` | Full 3-mode autonomous dynamic engine: EASY $\to$ Direct (1 call), UNCERTAIN $\to$ Self-Consistency ($1+N$ calls), HARD $\to$ Debate + Judge ($1+N\cdot R+1$ calls), 37 unit tests. |
| **Phase 8** | Robust LLM Engineering | `d34d404` | Domain exception taxonomy (`OllamaConnectionError`, `OllamaTimeoutError`), exponential backoff retry decorator with jitter (`@retry`), structured JSON logger with request IDs, router graceful fallbacks, 44 unit tests. |
| **Phase 9** | FastAPI REST Server & Streamlit Web UI | `e3b67db` | `GET /health` and `POST /reason` REST endpoints with Pydantic request validation, rich Streamlit web dashboard with interactive cards, transcripts, and telemetry, 51 unit tests. |
| **Phase 10** | Benchmark & Evaluation Engine | `5ef3cb1` | Curated GSM8K reasoning benchmark subset (`src/evaluation/dataset.py`), `BenchmarkEvaluator` running side-by-side 4-strategy comparisons (Direct vs SC vs Debate vs Adaptive), 57 unit tests. |
| **Phase 11** | Parameter Ablations & Experimental Matrix | `66d6818` | `AblationEngine` conducting grid sweeps across confidence thresholds, debate rounds, agent counts, and arbitration schemes, automated export to `experiments/`, 61 unit tests. |
| **Phase 12** | Confidence Calibration & 8-Category Failure Taxonomy | `88d1991` | Expected Calibration Error (ECE), MCE, Brier score, ASCII reliability diagrams, and 8-category diagnostic taxonomy classifying reasoning errors, concessions, and hallucinations, 67 unit tests. |
| **Phase 13** | Multi-Objective Pareto Optimization & Visualizer | `0ffb115` | Mathematical Pareto non-dominance engine, fitness scoring across Quality-First ($80/10/10$), Balanced ($50/25/25$), and Budget ($25/35/40$) presets, ASCII 2D scatter plots and trade-off bar charts, 72 unit tests. |
| **Phase 14** | System Reproducibility & Verification Pipeline | `CURRENT` | Unified reproduction script (`scripts/reproduce.py`) running all 9 system steps end-to-end, comprehensive test coverage (74 tests passing), master documentation guide (`CODEBASE_GUIDE.md`). |

---

## 2. Directory Hierarchy & Architectural Purpose

```
Adaptive LLM Reasoning & Debate Router/
├── config.json                     # System configuration & model profile catalog (JSON format)
├── requirements.txt                # Python package dependency specifications
├── README.md                       # Public documentation, setup instructions, quickstart
├── CODEBASE_GUIDE.md               # Master phase-by-phase architectural & codebase guide
├── implementation_plan.md          # Multi-phase engineering roadmap & design artifact
│
├── src/                            # Core application source code
│   ├── api/                        # FastAPI REST service and request/response schemas
│   ├── config/                     # Configuration loader and profile resolver
│   ├── evaluation/                 # Benchmark, calibration, ablation, Pareto optimization, and visualizer
│   ├── provider/                   # LLM provider abstractions, Ollama engine, token tracking, exceptions
│   ├── reasoning/                  # Direct, Self-Consistency, Multi-Agent Debate, Supreme Judge
│   ├── router/                     # Adaptive dynamic router, confidence estimation, difficulty models
│   ├── ui/                         # Streamlit interactive web dashboard
│   ├── utils/                      # Structured logging and exponential backoff retry utilities
│   └── main.py                     # Primary command-line interface (CLI)
│
├── scripts/                        # Automation & verification scripts
│   └── reproduce.py                # End-to-end automated system reproduction & verification runner
│
├── experiments/                    # Generated experimental data and benchmark reports
│   ├── ablation_results.json       # Exported ablation study raw telemetry
│   └── ablation_summary.csv        # Exported ablation study comparative metrics
│
└── tests/                          # Automated Pytest suite (74 unit and integration tests)
    ├── test_config.py              # Tests configuration loading and profile validation
    ├── test_ollama_provider.py     # Tests Ollama client, HTTP calls, and error handling
    ├── test_router.py              # Tests difficulty estimation and Mode 1 direct routing
    ├── test_consistency.py         # Tests Self-Consistency sampling and majority voting
    ├── test_debate.py              # Tests multi-agent debate rounds and consensus tracking
    ├── test_judge.py               # Tests Supreme Judge parsing and adjudication pipeline
    ├── test_complete_pipeline.py   # Tests complete 3-mode adaptive dynamic engine
    ├── test_robustness.py          # Tests retries, backoff, structured logger, and fallback mechanisms
    ├── test_api.py                 # Tests FastAPI endpoints (/health and /reason)
    ├── test_evaluation.py          # Tests benchmark evaluator and answer verification
    ├── test_ablation.py            # Tests ablation sweeps and export pipelines
    ├── test_calibration.py         # Tests calibration ECE, bins, and 8-category failure modes
    ├── test_optimization.py        # Tests Pareto dominance, fitness scoring, and ASCII visualizer
    └── test_reproducibility.py     # Tests automated end-to-end reproduction runner
```

---

## 3. Comprehensive File Manifest & Specifications

### Root Configuration & Documentation Files

#### [`config.json`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/config.json)
- **Purpose:** Centralized configuration file (strict JSON format) defining active execution profiles, router thresholds, and multi-agent debate personas.
- **Key Sections:**
  - `active_profile`: Active profile key (`local_fast`, `local_reasoning`).
  - `profiles`: Provider type, model name (`llama3.2:latest`, `qwen3:1.7b`), temperature, context limits, and base URLs.
  - `router`: `confidence_threshold_high` (0.80), `confidence_threshold_low` (0.50), `consistency_samples` (3), `sample_temperature` (0.7).
  - `debate`: `num_agents` (2), `num_rounds` (2), and agent persona prompt specifications (`Agent_A`, `Agent_B`).

#### [`requirements.txt`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/requirements.txt)
- **Purpose:** Specifies exact minimum library dependencies for the project (`requests`, `pydantic`, `pytest`, `fastapi`, `uvicorn`, `httpx`, `streamlit`).

#### [`README.md`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/README.md)
- **Purpose:** User-facing documentation containing project features, mathematical explanations, quickstart commands, API reference, and benchmarking instructions.

#### [`CODEBASE_GUIDE.md`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/CODEBASE_GUIDE.md)
- **Purpose:** This file. Master repository guide documenting phase history, architecture, and every file's function.

---

### Package: `src/` (Main CLI)

#### [`src/main.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/main.py)
- **Purpose:** Unified CLI entrypoint allowing users to run queries in any mode (`direct`, `consistency`, `debate`, `adaptive`), inspect reasoning traces, and view token usage and cost metrics.
- **Key Functions:**
  - `main()`: Parses CLI arguments (`--question`, `--mode`, `--profile`, `--samples`, `--rounds`).

---

### Package: `src/config/` (Configuration Management)

#### [`src/config/config.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/config/config.py)
- **Purpose:** Loads, parses, and validates `config.json` with safety checks and default profile fallback.
- **Key Functions:**
  - `load_config(config_path)`: Reads and parses JSON configuration file into a dictionary.
  - `get_active_profile(config)`: Resolves the active model profile block based on `active_profile`.

---

### Package: `src/provider/` (LLM Engine & Telemetry)

#### [`src/provider/base.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/provider/base.py)
- **Purpose:** Defines the abstract base class `LLMProvider` that all local and cloud LLM integrations must implement.
- **Key Classes / Methods:**
  - `LLMProvider(ABC)`: Abstract class defining `generate(request)` and `health_check()`.
  - `calculate_cost(usage)`: Default cost calculation helper (evaluates to $0.00 for local Ollama).

#### [`src/provider/models.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/provider/models.py)
- **Purpose:** Pydantic data schemas representing provider requests, responses, and token telemetry.
- **Key Classes:**
  - `TokenUsage`: Container tracking `prompt_tokens`, `completion_tokens`, and `total_tokens`.
  - `LLMRequest`: Standardized input request (prompt, system prompt, temperature, max tokens, timeout).
  - `LLMResponse`: Standardized completion response (text, model, token usage, latency seconds).
  - `StructuredQAResponse`: High-level output packaging final answer, explanation, and telemetry.

#### [`src/provider/ollama.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/provider/ollama.py)
- **Purpose:** Concrete implementation of `LLMProvider` interfacing directly with Ollama's local REST API (`/api/generate` and `/api/tags`).
- **Key Features:** Integrated with `@retry` for exponential backoff, request ID correlation logging, and specific HTTP error mapping (404, Connection, Timeout).

#### [`src/provider/exceptions.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/provider/exceptions.py)
- **Purpose:** Domain exception hierarchy providing clear, actionable error categorization.
- **Key Classes:**
  - `LLMProviderError`: Base domain exception.
  - `OllamaConnectionError`: Raised when Ollama daemon is unreachable (inherits from `ConnectionError`).
  - `OllamaTimeoutError`: Raised when generation exceeds timeout (inherits from `TimeoutError`).
  - `OllamaModelNotFoundError`: Raised on HTTP 404 model missing errors.
  - `OllamaResponseError`: Raised on malformed JSON or HTTP 5xx responses.

#### [`src/provider/factory.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/provider/factory.py)
- **Purpose:** Factory function instantiating concrete `LLMProvider` instances based on configuration profile dictionaries.
- **Key Functions:**
  - `get_provider(profile_config)`: Instantiates `OllamaProvider` (or future cloud adapters).

---

### Package: `src/reasoning/` (Reasoning Paradigms)

#### [`src/reasoning/direct.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/reasoning/direct.py)
- **Purpose:** Implements Mode 1 (DIRECT) reasoning: single-pass execution extracting answer and explanation via regex.
- **Key Classes / Methods:**
  - `DirectReasoner`: Methods `answer(question)` and `_parse_response(text)`.

#### [`src/reasoning/models.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/reasoning/models.py)
- **Purpose:** Data schemas for Mode 2 (Self-Consistency) candidates and consensus voting outcomes.
- **Key Classes:**
  - `CandidateSolution`: Represents a single sampled reasoning path (`sample_id`, `answer`, `explanation`).
  - `SelfConsistencyResult`: Aggregates $N$ candidates, `agreement_score`, `agreement_distribution`, and final consensus answer.

#### [`src/reasoning/consistency.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/reasoning/consistency.py)
- **Purpose:** Implements Mode 2 (Self-Consistency) reasoning (Wang et al., 2022).
- **Key Methods:**
  - `sample_and_vote(question, num_samples, temperature)`: Samples $N$ parallel candidate paths at $T>0$, normalizes candidate answers, and conducts majority voting.
  - `_normalize_answer(answer)`: Canonicalizes punctuation, whitespaces, and currency formatting.

#### [`src/reasoning/debate_models.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/reasoning/debate_models.py)
- **Purpose:** Data models capturing multi-agent debate participants, turns, rounds, and transcripts.
- **Key Classes:**
  - `AgentConfig`: Name, persona prompt, temperature, max tokens.
  - `DebateTurn`: An agent's argument, proposed answer, and revised flag.
  - `DebateRound`: Collection of turns in a single debate round.
  - `DebateTranscript`: Full audit log of all rounds, consensus check, and aggregated telemetry.

#### [`src/reasoning/debate.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/reasoning/debate.py)
- **Purpose:** Implements the Multi-Agent Debate Engine managing interactive rounds between contrasting personas.
- **Key Classes:**
  - `DebateAgent`: Generates Round 1 initial positions and Round 2 cross-examination critiques/revisions.
  - `MultiAgentDebateEngine`: Orchestrates rounds, feeds opponent arguments to peers, and checks for consensus.

#### [`src/reasoning/judge_models.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/reasoning/judge_models.py)
- **Purpose:** Data models capturing Supreme Debate Judge verdicts and complete pipeline results.
- **Key Classes:**
  - `JudgeVerdict`: Authoritative verdict answer, analytical evaluation summary, winning agent, verdict confidence, and identified flaws.
  - `DebatePipelineResult`: End-to-end outcome packaging question, final answer, transcript, verdict, and telemetry.

#### [`src/reasoning/judge.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/reasoning/judge.py)
- **Purpose:** Implements Mode 3 Supreme Judge adjudication pipeline.
- **Key Classes:**
  - `DebateJudge`: Reviews entire debate transcript and issues structured adjudication with fallacy detection.
  - `DebateWithJudgePipeline`: Coordinates debate execution followed by judge adjudication.

---

### Package: `src/router/` (Autonomous Dynamic Router)

#### [`src/router/models.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/router/models.py)
- **Purpose:** Data schemas for query difficulty levels, reasoning strategies, and routed responses.
- **Key Classes:**
  - `DifficultyLevel(Enum)`: `EASY`, `UNCERTAIN`, `HARD`.
  - `ReasoningStrategy(Enum)`: `DIRECT`, `SELF_CONSISTENCY`, `MULTI_AGENT_DEBATE`.
  - `ConfidenceAssessment`: Estimated score (0.0 to 1.0), categorical level (`HIGH`, `MEDIUM`, `LOW`), difficulty, and justification.
  - `RoutedResponse`: Final answer, explanation, chosen strategy, difficulty, confidence, call count, and embedded transcripts/verdicts.

#### [`src/router/estimator.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/router/estimator.py)
- **Purpose:** Lightweight metacognitive single-pass evaluator estimating problem complexity and verbalized confidence prior to full execution.
- **Key Methods:**
  - `estimate(question)`: Prompts the LLM to analyze the question and return XML-formatted difficulty, confidence score, and justification.

#### [`src/router/router.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/router/router.py)
- **Purpose:** Autonomous dynamic router directing questions to the optimal reasoning paradigm based on confidence thresholds:
  - Confidence $\ge T_{\text{high}}$ (0.80): Dispatches Mode 1 (DIRECT).
  - $T_{\text{low}} \le$ Confidence $< T_{\text{high}}$: Dispatches Mode 2 (SELF-CONSISTENCY).
  - Confidence $< T_{\text{low}}$ (0.50): Dispatches Mode 3 (MULTI-AGENT DEBATE + JUDGE).
- **Key Methods:**
  - `route_and_solve(question)`: Runs estimation, applies decision rules, executes target mode, and handles graceful degradation fallbacks if higher-order modes encounter errors.

---

### Package: `src/api/` (REST Backend)

#### [`src/api/models.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/api/models.py)
- **Purpose:** Pydantic schemas validating incoming FastAPI REST payloads and outgoing reasoning responses.
- **Key Classes:**
  - `ReasoningRequest`: Input query, mode override (`adaptive`, `direct`, `consistency`, `debate`), profile, rounds, samples.
  - `ReasoningResponse`: Complete serialized response with strategy, confidence, call count, latency, answer, and telemetry.
  - `HealthResponse`: System health status, active profile, and provider accessibility.

#### [`src/api/app.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/api/app.py)
- **Purpose:** Production FastAPI server exposing endpoints:
  - `GET /health`: Returns service and local Ollama health status.
  - `POST /reason`: Routes queries through the adaptive engine or manual mode overrides.

---

### Package: `src/ui/` (Web Dashboard)

#### [`src/ui/app.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/ui/app.py)
- **Purpose:** Streamlit interactive web application rendering interactive reasoning cards, debate transcripts, consensus status, and configuration controls.

---

### Package: `src/utils/` (Production Utilities)

#### [`src/utils/logger.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/utils/logger.py)
- **Purpose:** Enterprise structured logger supporting contextual request IDs, JSON log formatting, and colored console outputs.

#### [`src/utils/retry.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/utils/retry.py)
- **Purpose:** Decorator `@retry` implementing exponential backoff with full jitter for transient network and LLM provider errors.

---

### Package: `src/evaluation/` (Evaluation & Optimization Engine)

#### [`src/evaluation/dataset.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/evaluation/dataset.py)
- **Purpose:** Curated benchmark subset of mathematical and logical reasoning questions (GSM8K style) with ground truth answers and objective verification utilities.
- **Key Functions:**
  - `extract_numerical_answer(text)`: Isolates numerical and canonical values from free-form explanations.
  - `is_answer_correct(predicted, ground_truth)`: Evaluates semantic or numerical equivalence.
  - `get_benchmark_dataset(limit, category, difficulty)`: Filters and retrieves test items.

#### [`src/evaluation/evaluator.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/evaluation/evaluator.py)
- **Purpose:** Comprehensive benchmark comparator evaluating Direct, Self-Consistency, Debate, and Adaptive Router across Accuracy, Calls/Query, Token Consumption, and Latency.
- **Key Classes:**
  - `BenchmarkEvaluator`: Methods `evaluate_direct`, `evaluate_self_consistency`, `evaluate_debate`, `evaluate_adaptive`, and `run_full_comparison`.

#### [`src/evaluation/run_eval.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/evaluation/run_eval.py)
- **Purpose:** CLI runner executing comparative benchmark evaluations and printing side-by-side terminal comparison tables.

#### [`src/evaluation/ablation.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/evaluation/ablation.py)
- **Purpose:** Experimental parameter ablation framework exploring threshold sweeps, debate rounds, agent counts, and arbitration mechanisms.
- **Key Classes:**
  - `AblationEngine`: Methods `run_threshold_sweep`, `run_debate_round_sweep`, `run_agent_count_sweep`, `run_arbitration_sweep`, `export_results_json`, and `export_summary_csv`.

#### [`src/evaluation/run_ablation.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/evaluation/run_ablation.py)
- **Purpose:** CLI runner for ablation studies exporting results to `experiments/ablation_results.json` and `experiments/ablation_summary.csv`.

#### [`src/evaluation/calibration.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/evaluation/calibration.py)
- **Purpose:** Quantifies verbalized confidence calibration:
  - Expected Calibration Error (ECE) across probability intervals.
  - Maximum Calibration Error (MCE) and Brier score.
  - Overconfidence / underconfidence indicators.

#### [`src/evaluation/failure_analysis.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/evaluation/failure_analysis.py)
- **Purpose:** 8-category diagnostic taxonomy classifying reasoning errors, concessions, under/over-routing, and hallucinations:
  1. `INITIAL_WRONG_DEBATE_CORRECT`: Successful error recovery.
  2. `INITIAL_WRONG_DEBATE_WRONG`: Intractable fallacy.
  3. `INITIAL_CORRECT_DEBATE_CORRECT`: Truth preservation.
  4. `INITIAL_CORRECT_DEBATE_WRONG`: Harmful concession.
  5. `UNDER_ROUTING`: Direct chosen on hard problem causing error.
  6. `OVER_ROUTING`: Debate triggered on easy question.
  7. `JUDGE_SELECTION_ERROR`: Judge favored incorrect debater.
  8. `UNANIMOUS_HALLUCINATION`: Both debaters agreed on false premise.

#### [`src/evaluation/run_calibration.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/evaluation/run_calibration.py)
- **Purpose:** CLI runner generating ASCII reliability diagrams and 8-category failure distributions.

#### [`src/evaluation/optimization.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/evaluation/optimization.py)
- **Purpose:** Multi-objective Pareto optimization framework isolating the non-dominated Pareto frontier across Accuracy vs Compute vs Latency.
- **Key Classes:**
  - `ParetoOptimizer`: Implements `dominates(a, b)`, `compute_pareto_frontier(candidates)`, and priority-based fitness scoring (`QUALITY_FIRST`, `BALANCED`, `BUDGET_CONSTRAINED`).

#### [`src/evaluation/visualize.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/src/evaluation/visualize.py)
- **Purpose:** ASCII visualizer generating 2D scatter plots with Pareto frontiers, comparative horizontal bar charts, and routing proportions.

---

### Package: `scripts/` (Automation & Reproduction)

#### [`scripts/reproduce.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/scripts/reproduce.py)
- **Purpose:** Unified end-to-end verification script executing all 9 core capabilities (Modes 1, 2, 3, Adaptive Router, Benchmarks, Calibration, Failure Taxonomy, and Pareto Optimization). Supports `--dry-run` and `--live`.

---

### Package: `tests/` (Test Suite - 74 Tests Passing)

#### [`tests/test_config.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_config.py)
- Tests `config.json` loading, profile extraction, and missing config handling.

#### [`tests/test_ollama_provider.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_ollama_provider.py)
- Tests Ollama HTTP communication, response parsing, factory instantiation, and direct reasoning.

#### [`tests/test_router.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_router.py)
- Tests XML difficulty estimation parsing and Mode 1 direct routing.

#### [`tests/test_consistency.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_consistency.py)
- Tests Self-Consistency answer canonicalization, majority voting, and Mode 2 router activation.

#### [`tests/test_debate.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_debate.py)
- Tests multi-agent turns, counter-argument critique generation, and consensus detection.

#### [`tests/test_judge.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_judge.py)
- Tests Supreme Judge verdict parsing, fallacy identification, and end-to-end debate pipeline.

#### [`tests/test_complete_pipeline.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_complete_pipeline.py)
- Integration tests verifying autonomous dynamic dispatch across all 3 reasoning modes.

#### [`tests/test_robustness.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_robustness.py)
- Tests exponential retry decorator, backoff exhaustion, error classification, and graceful fallbacks.

#### [`tests/test_api.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_api.py)
- Integration tests for FastAPI endpoints (`GET /health`, `POST /reason`).

#### [`tests/test_evaluation.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_evaluation.py)
- Tests curated benchmark loading, numerical extraction, and 4-strategy evaluation flows.

#### [`tests/test_ablation.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_ablation.py)
- Tests ablation parameter grids and export to JSON and CSV.

#### [`tests/test_calibration.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_calibration.py)
- Tests Expected Calibration Error (ECE), binning, and all 8 taxonomy classifications.

#### [`tests/test_optimization.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_optimization.py)
- Tests Pareto dominance, multi-objective fitness scoring, and ASCII visualizer renderers.

#### [`tests/test_reproducibility.py`](file:///c:/Users/my%20pc/Desktop/Adaptive%20LLM%20Reasoning%20&%20Debate%20Router/tests/test_reproducibility.py)
- Tests mock provider and end-to-end execution of the reproduction script.
