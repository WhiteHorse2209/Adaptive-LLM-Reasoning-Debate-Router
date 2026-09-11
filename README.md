# Adaptive LLM Reasoning & Debate Router

## Overview
This project builds an adaptive reasoning system that dynamically determines how much computational effort to spend on a user query. It intelligently routes questions across three modes:
1. **Direct Mode:** Fast, zero-shot answer for easy/high-confidence queries.
2. **Self-Consistency:** Evaluates multiple parallel candidate generations for uncertain queries.
3. **Multi-Agent Debate:** Engages multiple agents in a structured, multi-round peer-review debate for hard queries.

The primary initial design focuses entirely on local-first LLM inference using **Ollama**, ensuring **zero external API costs** during development and evaluation, with computation performed on the user's local hardware.

---

## Architecture

```
                    USER QUESTION
                          │
                          ▼
                 CONFIDENCE ESTIMATOR
              (Score: 0.0-1.0, Difficulty)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
      [EASY]         [UNCERTAIN]       [HARD]
   Score >= 0.80    0.50 <= S < 0.80  Score < 0.50
          │               │               │
          ▼               ▼               ▼
       MODE 1          MODE 2          MODE 3
       DIRECT     SELF-CONSISTENCY  MULTI-AGENT DEBATE
    (1 LLM Call)   (N Gen Samples)   (Agent A ↔ Agent B)
          │               │               │
          │               │         Round 1 (Independent)
          │               │         Round 2 (Cross-Exam)
          │               │         Round 3 (Rebuttal)
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
                 LLM PROVIDER LAYER (Abstract)
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
    OLLAMA PROVIDER              (Optional Cloud)
  (Local GPU / CPU)
           │
           ▼
    STRUCTURED ROUTED OUTPUT
  (Answer, Strategy, Confidence, Call Count, Tokens, Latency)
```

---

## Core Components

### Phase 2: Provider Abstraction & Single LLM Direct Reasoning
- **`src/provider/models.py`**: Pydantic data schemas (`LLMRequest`, `LLMResponse`, `TokenUsage`, `StructuredQAResponse`).
- **`src/provider/base.py`**: Abstract base class `LLMProvider` defining uniform `generate()`, `health_check()`, and cost calculation interfaces.
- **`src/provider/ollama.py`**: Full Ollama REST API integration (`/api/generate`) with automatic token count extraction (`prompt_eval_count`, `eval_count`), high-resolution wall-clock latency measurement, and robust error handling.
- **`src/provider/factory.py`**: Provider factory isolating model selection from application logic.
- **`src/reasoning/direct.py`**: `DirectReasoner` handling single-model inference and parsing answers/explanations into structured formats.

### Phase 3: Confidence Estimation & Adaptive Direct Routing
- **`src/router/models.py`**: Enums for `DifficultyLevel` (`EASY`, `UNCERTAIN`, `HARD`), `ReasoningStrategy` (`DIRECT`, `SELF_CONSISTENCY`, `MULTI_AGENT_DEBATE`), `ConfidenceAssessment`, and `RoutedResponse`.
- **`src/router/estimator.py`**: `ConfidenceEstimator` providing single-pass difficulty classification, verbalized confidence extraction (0.0 to 1.0), and justification notes with resilient parsing and fallback heuristics.
- **`src/router/router.py`**: `AdaptiveRouter` comparing confidence against configurable thresholds. Easy queries immediately route to `DIRECT` mode (1 call, 0 extra agents), intelligently preventing unnecessary compute.

### Phase 4: Self-Consistency Reasoning
- **`src/reasoning/models.py`**: `CandidateSolution` and `SelfConsistencyResult` schemas tracking each candidate reasoning chain, sample token usage, individual latency, and overall consensus.
- **`src/reasoning/consistency.py`**: `SelfConsistencyReasoner` executing $N$ independent generation paths with non-zero temperature ($T=0.7$), canonicalizing candidate outputs, and selecting the consensus answer via majority voting:
  $$\text{Agreement} = \frac{\max_a \text{count}(a)}{N}$$
- **`src/router/router.py`**: Seamlessly executes Self-Consistency for queries identified as `UNCERTAIN`, tracking all $1 + N$ calls, accumulated tokens, latency, and sample distributions.

### Phase 5: Multi-Agent Debate Foundation
- **`src/reasoning/debate_models.py`**: Data schemas including `AgentConfig`, `DebateTurn`, `DebateRound`, and `DebateTranscript`.
- **`src/reasoning/debate.py`**:
  - `DebateAgent`: Individual debater driven by distinct personas, evaluating opponent arguments, and actively defending or revising its conclusions.
  - `MultiAgentDebateEngine`: Genuine multi-round peer debate (NOT a sequential Solver -> Critic -> Verifier pipeline).
    - **Round 1 (Parallel Independent Discovery)**: Agents solve the question in isolation to eliminate anchoring bias.
    - **Round 2 (Mutual Cross-Examination)**: Agents review opponents' reasoning, challenge assumptions, defend valid deductions, or concede errors.
    - **Round 3 (Optional Convergence)**: Deepens debate until consensus or principled divergence.
- **`src/main.py`**: Standalone execution via `--mode debate` displaying round-by-round arguments, revisions, and consensus telemetry.

### Phase 6: Debate Judge & Adjudication Pipeline
- **`src/reasoning/judge_models.py`**: Pydantic schemas for `JudgeVerdict` (winning agent, verdict answer, confidence, evaluation summary, identified flaws) and `DebatePipelineResult`.
- **`src/reasoning/judge.py`**:
  - `DebateJudge`: Impartial arbiter reviewing complete debate transcripts. Rather than simple majority voting, it evaluates argument consistency, evidence strength, responsiveness to peer critiques, and identified flaws.
  - `DebateWithJudgePipeline`: Unified coordinator orchestrating multi-round debate execution followed by authoritative judge adjudication.

### Phase 7: Complete Adaptive Reasoning Engine
- **`src/router/models.py`**: Extended `RoutedResponse` embedding optional `DebateTranscript`, `JudgeVerdict`, and `SelfConsistencyResult` along with unified token, call count, and latency metrics.
- **`src/router/router.py`**: Dynamic three-tier dispatch engine:
  - **`EASY`** ($\ge 0.80$ confidence) $\to$ **Mode 1: Direct Reasoning** ($1$ call).
  - **`UNCERTAIN`** ($0.50 \le \text{conf} < 0.80$) $\to$ **Mode 2: Self-Consistency** ($1 + N$ calls with majority voting).
  - **`HARD`** ($< 0.50$ confidence) $\to$ **Mode 3: Multi-Agent Debate + Judge** ($1 + N_{\text{agents}} \times N_{\text{rounds}} + 1$ calls).
- **`src/main.py`**: Unified CLI with automatic adaptive routing (`--mode adaptive`), manual overrides (`--mode direct`, `--mode debate`), and comprehensive telemetry reporting.
- **`tests/test_complete_pipeline.py`**: End-to-end integration test suite validating call count bounds, token accumulation, and routing decisions.

### Phase 8: Robust LLM Engineering
- **`src/provider/exceptions.py`**: Domain exception hierarchy distinguishing connection refusal (`OllamaConnectionError`), missing local model / 404 (`OllamaModelNotFoundError`), wall-clock timeouts (`OllamaTimeoutError`), and corrupted JSON structures (`MalformedOutputError`).
- **`src/utils/logger.py`**: Structured logger tagging traces with correlated `request_id`, timestamps, log levels, and extra telemetry metadata.
- **`src/utils/retry.py`**: Exponential backoff retry engine with jitter, capped retries (preventing infinite loops), and immediate fast-path aborts on non-retryable errors.
- **`src/router/router.py`**: End-to-end request ID propagation and graceful error degradation — safely falling back to direct reasoning if multi-agent debate or self-consistency encounters unrecoverable upstream failures.
- **`tests/test_robustness.py`**: Comprehensive test suite verifying retries, error classifications, fast-path aborts, and router fallback mechanisms.

### Phase 9: FastAPI + Streamlit Serving Layer
- **`src/api/models.py`**: Strict Pydantic models for API requests (`ReasonApiRequest`) and telemetry-rich responses (`ReasonApiResponse`, `HealthApiResponse`).
- **`src/api/app.py`**: Production-grade FastAPI application with CORS support exposing:
  - `GET /health`: Health probe reporting Ollama reachability, active profile, and model status.
  - `POST /reason`: Universal inference endpoint supporting autonomous adaptive dispatch or manual mode overrides with structured telemetry.
- **`src/ui/app.py`**: Modern Streamlit web application with custom dark glassmorphism styling, sample prompts, hyperparameter adjustment sliders, dynamic metric ribbons, candidate voting distribution graphs, round-by-round debate transcripts, and Supreme Judge adjudication cards.
- **`tests/test_api.py`**: Comprehensive endpoint test suite validating health probes, all reasoning modes, and 503 error handling.

### Phase 10: Benchmark & Evaluation Engine
- **`src/evaluation/dataset.py`**: Curated GSM8K arithmetic, multi-step math, and symbolic logic benchmark dataset with canonical ground truth answers and numerical verification (`extract_numerical_answer`, `is_answer_correct`).
- **`src/evaluation/evaluator.py`**: 4-way comparative evaluation engine (`BenchmarkEvaluator`) systematically assessing:
  1. Direct Inference (1 model call)
  2. Self-Consistency (N samples, consensus voting)
  3. Always-On Multi-Agent Debate (+ Supreme Judge)
  4. Adaptive Router (Dynamic difficulty-aware routing)
- **`src/evaluation/run_eval.py`**: Automated CLI benchmark tool reporting accuracy, mean calls per query, token consumption, latency, $0.00 external API cost, and routing breakdown (% Direct, % SC, % Debate).
- **`tests/test_evaluation.py`**: Unit test suite validating answer extraction, tolerance verification, dataset filtering, and comparative evaluator metrics.

### Phase 11: Experiments & Ablation Framework
- **`src/evaluation/ablation.py`**: Systematic experimental engine (`AblationEngine`) evaluating configuration permutations across 4 dimensions:
  1. **Threshold Grids**: Standard (0.80/0.50) vs Conservative/Quality-first (0.90/0.70) vs Cost-saver/Direct (0.70/0.30).
  2. **Debate Depth**: 1 Round vs 2 Rounds vs 3 Rounds.
  3. **Agent Multiplicity**: 2 Agents vs 3 Agents.
  4. **Adjudication Method**: Impartial Supreme Judge vs Naive Agent Majority Voting.
- **`src/evaluation/run_ablation.py`**: CLI tool executing ablation runs and exporting structured JSON/CSV metrics to `experiments/`.
- **`tests/test_ablation.py`**: Unit test suite validating grid generation, execution, majority voting ablation, and artifact export.

---

## Quickstart & Installation

### 1. Prerequisites
- Install [Ollama](https://ollama.com/) locally.
- Start the Ollama server:
  ```bash
  ollama serve
  ```
- Pull local models (e.g., `llama3.2` or `qwen3`):
  ```bash
  ollama pull llama3.2:latest
  ```

### 2. Environment Setup
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Unit & Integration Tests
```bash
python -m pytest tests/ -v
```

### 4. Running the Modes

**Mode 1: Direct Answer (Easy Query):**
```bash
python -m src.main --question "What is 25 * 4?" --profile local_fast
```

**Mode 2: Self-Consistency (Uncertain Query):**
```bash
python -m src.main --question "Which number is larger: 9.11 or 9.9?" --profile local_fast
```

**Mode 3: Multi-Agent Debate (Complex / Contested Query):**
```bash
python -m src.main --question "Should autonomous vehicles prioritize passenger safety or pedestrian safety in an unavoidable accident?" --mode debate --rounds 2 --profile local_fast
```

Example Debate Output:
```
======================================================================
GENUINE MULTI-AGENT DEBATE TRANSCRIPT
Question: Should autonomous vehicles prioritize passenger safety or pedestrian safety?
======================================================================

>>> ROUND 1 <<<
[Agent_A]:
Argument:
  From a utilitarian perspective, minimizing total casualties is paramount...
Proposed Answer: Prioritize pedestrian safety to minimize total harm.

[Agent_B]:
Argument:
  From a contractual perspective, manufacturers owe primary duty to passengers...
Proposed Answer: Prioritize passenger safety based on fiduciary duty.

>>> ROUND 2 <<<
[Agent_A]:
Argument:
  Agent B's argument overlooks the moral asymmetry between passengers...
Proposed Answer: Prioritize pedestrian safety.

[Agent_B] [REVISED ANSWER!]:
Argument:
  Agent A raises a valid point regarding the moral duty to unconsenting pedestrians...
Proposed Answer: Prioritize pedestrian safety with minimial harm routing.

======================================================================
DEBATE TELEMETRY & CONSENSUS:
  Consensus Reached:   YES
  Consensus Answer:    Prioritize pedestrian safety
  Total Agent Calls:   4
  Total Latency:       62.4s
  Total Tokens:        840
  External Cost:       $0.0000 (Zero API Cost - Local Hardware)
======================================================================

>>> IMPARTIAL JUDGE ADJUDICATION <<<
Winning Agent:         Agent_A
Verdict Confidence:    0.95
Adjudication Summary:  Agent A's reasoning remained robust against edge cases while Agent B conceded once externalized pedestrian harm was formalized.
Identified Flaws:      Agent B's initial claim failed to account for involuntary third-party risk.
Final Answer:          Prioritize pedestrian safety
```

**Adaptive Engine Execution (Autonomous Dynamic Routing):**
```bash
python -m src.main --question "What is the capital of France?" --mode adaptive --profile local_fast
# -> Evaluated as EASY (Confidence 0.98) -> Executes DIRECT (1 call)

python -m src.main --question "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?" --mode adaptive --profile local_fast
# -> Evaluated as UNCERTAIN (Confidence 0.65) -> Executes SELF-CONSISTENCY (4 calls, majority voting)

python -m src.main --question "Resolve the ship of Theseus paradox considering physical continuity versus informational identity." --mode adaptive --profile local_fast
# -> Evaluated as HARD (Confidence 0.35) -> Executes MULTI-AGENT DEBATE + JUDGE (6 calls)
```

### 5. Running the FastAPI REST Server
Start the production API server on port 8000:
```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

- **Health Check:**
  ```bash
  curl -X GET http://localhost:8000/health
  ```
- **Reasoning Query (Adaptive):**
  ```bash
  curl -X POST http://localhost:8000/reason \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"Which is larger: 9.11 or 9.9?\", \"mode\": \"adaptive\"}"
  ```
- **Reasoning Query (Manual Debate Override):**
  ```bash
  curl -X POST http://localhost:8000/reason \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"Should AI possess legal rights?\", \"mode\": \"debate\", \"rounds\": 2}"
  ```

### 6. Running the Streamlit Web Dashboard
Launch the interactive web UI:
```bash
python -m streamlit run src/ui/app.py
```
This opens `http://localhost:8501` in your browser with interactive cards, debate round transcripts, consensus graphs, and engine configuration controls.

### 7. Running the Automated Benchmark
Compare all 4 reasoning paradigms on the curated GSM8K benchmark:
```bash
python -m src.evaluation.run_eval --limit 5 --profile local_fast
```
Outputs a side-by-side comparison table showing accuracy, calls per query, token consumption, latency, and routing distribution.

### 8. Running Ablation Studies
Run parameter grid evaluations exploring thresholds, debate rounds, agent counts, and judge arbitration:
```bash
python -m src.evaluation.run_ablation --suite all --limit 3 --profile local_fast
```
Exports experimental results to `experiments/ablation_results.json` and `experiments/ablation_summary.csv`.

---

## Configuration (`config.json`)

Model profiles, router thresholds, and debate hyperparameters are configured independently of application code:
```json
{
  "active_profile": "local_fast",
  "profiles": {
    "local_fast": {
      "provider": "ollama",
      "model": "llama3.2:latest",
      "temperature": 0.1,
      "max_tokens": 500,
      "timeout": 60,
      "api_base": "http://localhost:11434"
    },
    "local_reasoning": {
      "provider": "ollama",
      "model": "qwen3:1.7b",
      "temperature": 0.4,
      "max_tokens": 2000,
      "timeout": 120,
      "api_base": "http://localhost:11434"
    }
  },
  "router": {
    "confidence_threshold_high": 0.80,
    "confidence_threshold_low": 0.50,
    "estimation_mode": "single_pass",
    "consistency_samples": 3,
    "sample_temperature": 0.7
  },
  "debate": {
    "num_agents": 2,
    "num_rounds": 2,
    "agent_temperature": 0.7,
    "agents": [
      {
        "name": "Agent_A",
        "persona": "Analytical logician focused on rigorous first-principles derivation."
      },
      {
        "name": "Agent_B",
        "persona": "Critical empirical thinker focused on edge cases and counterexamples."
      }
    ]
  }
}
```
