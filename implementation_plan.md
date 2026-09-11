# Phase 5 Implementation Plan: Multi-Agent Debate Foundation

## Goal Description
Implement the core **Multi-Agent Debate** engine using local Ollama models.
Rather than a sequential review pipeline (Solver -> Critic -> Verifier), Phase 5 implements a genuine interactive debate where multiple agents (Agent A, Agent B) independently solve a question, inspect each other's full reasoning chains, challenge flawed assumptions, defend valid points, and revise their conclusions over multiple rounds.

## Key Principles of Genuine Multi-Agent Debate
1. **Parallel Independent Discovery (Round 1)**:
   - Agent A and Agent B solve the question with zero knowledge of each other's existence to avoid groupthink and anchoring bias.
2. **Mutual Cross-Examination (Round 2+)**:
   - Agent A is shown Agent B's exact argument and answer.
   - Agent B is shown Agent A's exact argument and answer.
   - Each agent is prompted to:
     - Identify specific logical gaps, arithmetic errors, or unfounded assumptions in the peer's argument.
     - Counter-argue and defend its own stance if sound.
     - Concede and revise if the peer exposed a legitimate mistake.
3. **Multi-Round Convergence & Revision Tracking**:
   - Each turn detects whether an agent maintained its stance or revised its answer (`revised: True/False`).
   - Tracks whether consensus is reached across all agents at the end of the rounds.
4. **Configurable Debate Hyperparameters**:
   - `num_agents`: default 2 (supports 3).
   - `num_rounds`: default 2 (supports 3+).
   - `agent_temperature`: non-zero temperature (e.g. 0.7) for cognitive diversity.
   - Dedicated persona configurations.

---

## Proposed Changes

### 1. Debate Data Models
#### [NEW] `src/reasoning/debate_models.py`
- `AgentConfig`: `name`, `persona`, `model`, `temperature`.
- `DebateTurn`:
  - `round_number: int`
  - `agent_name: str`
  - `argument: str`
  - `current_answer: str`
  - `revised: bool`
  - `token_usage: TokenUsage`
  - `latency_seconds: float`
- `DebateRound`:
  - `round_number: int`
  - `turns: List[DebateTurn]`
- `DebateTranscript`:
  - `question: str`
  - `rounds: List[DebateRound]`
  - `final_agent_answers: Dict[str, str]`
  - `consensus_reached: bool`
  - `total_calls: int` ($N_{\text{agents}} \times N_{\text{rounds}}$)
  - `total_token_usage: TokenUsage`
  - `total_latency_seconds: float`
  - `external_api_cost: float = 0.0`
  - `metadata: Dict[str, Any]`

### 2. Multi-Agent Debate Engine
#### [NEW] `src/reasoning/debate.py`
- `DebateAgent`:
  - Represents an individual debater.
  - Generates initial independent reasoning.
  - Analyzes opponent arguments, challenges flaws, defends valid deductions, and outputs revised or defended answer.
- `MultiAgentDebateEngine`:
  - Orchestrates the rounds.
  - Ensures clean peer transcript sharing.
  - Aggregates the `DebateTranscript`.

### 3. Configuration Updates
#### [MODIFY] `config.json`
- Add debate configuration:
  ```json
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
  ```

### 4. CLI Execution
#### [MODIFY] `src/main.py`
- Support `--mode debate` to run a direct standalone debate and print the complete multi-round debate transcript, agent rebuttals, and consensus status.

### 5. Automated Tests
#### [NEW] `tests/test_debate.py`
- Unit tests for:
  - Round 1 independent reasoning generation.
  - Round 2 prompt construction containing opponent transcripts.
  - Revision detection (`revised: True/False`).
  - Consensus evaluation when agents agree vs disagree.
  - Token aggregation and call counting ($N_{\text{agents}} \times N_{\text{rounds}}$).

### 6. Documentation & Git Sync
#### [MODIFY] `implementation_plan.md`
- Attach Phase 5 plan directly in repository root.
#### [MODIFY] `README.md`
- Add Phase 5 Multi-Agent Debate architecture, debate protocol, CLI instructions, and transcript format.
- Git commit and push to remote: `feat(phase-5): implement multi-agent debate foundation and transcript tracking`.

---

## Verification Plan

### Automated Tests
- Run `.\venv\Scripts\python.exe -m pytest tests/ -v` to ensure all existing (26) and new unit tests pass (100% pass rate).

### Manual Verification
- Run a standalone debate via CLI:
  `python -m src.main --question "Should self-driving cars prioritize passenger safety or pedestrian safety in unavoidable collisions?" --mode debate --profile local_fast`
- Verify that Round 1 shows independent solutions from Agent A and Agent B.
- Verify that Round 2 shows each agent referencing the other's points, challenging or defending arguments, and providing their current answer.
- Verify total call count, token usage, latency, and $0.00 external cost.
