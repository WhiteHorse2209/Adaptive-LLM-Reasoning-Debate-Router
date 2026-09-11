# Dual-Phase Implementation Plan: Phase 14 (Testing & Reproducibility) & Phase 15 (Cloud Model Adapters & Production Polish)

## Goal Description
Complete the final phases of the Adaptive LLM Reasoning & Debate Router portfolio project:
1. **Phase 14 — Testing + Reproducibility + Final Documentation**:
   - Comprehensive test suite validation (ensure 100% pass across all unit and integration layers).
   - Automated end-to-end demonstration and verification script `scripts/reproduce.py` executing all 3 reasoning modes, benchmark evaluation, calibration, and Pareto optimization.
   - Comprehensive documentation polish with ASCII architecture flowcharts, API reference, benchmark table, and full setup guides.
2. **Phase 15 — Cloud Model Adapters + Final Polish**:
   - Cloud provider adapters (`OpenAIProvider`, `AnthropicProvider`) following the uniform `LLMProvider` interface with optional API key support and offline mock capabilities.
   - Final codebase cleanup, configuration validations, and portfolio presentation polish.

---

## Phase 14: Testing + Reproducibility + Final Documentation

### 1. Unified Reproducibility Script
#### [NEW] `scripts/reproduce.py`
- End-to-end verification script executing:
  1. Config validation & provider health checks.
  2. Direct mode execution on sample easy query.
  3. Self-Consistency majority voting on sample uncertain query.
  4. Multi-agent debate and judge adjudication on hard query.
  5. Mini-benchmark evaluation comparison.
  6. Confidence calibration and failure mode analysis.
  7. Pareto optimization and ASCII scatter plot rendering.

### 2. Comprehensive Documentation Polish
#### [MODIFY] `README.md`
- Complete system architecture diagrams with ASCII routing charts.
- API and CLI reference with examples for every command.
- Full installation guide, local Ollama setup, and reproduction instructions.

### 3. Unit & Integration Test Suite Finalization
#### [NEW] `tests/test_reproducibility.py`
- Tests end-to-end flow of the reproduction pipeline.
- Git commit message: `feat(phase-14): implement end-to-end reproducibility pipeline and comprehensive system documentation`.

---

## Phase 15: Cloud Model Adapters + Final Polish

### 1. Cloud Provider Adapters
#### [NEW] `src/provider/cloud.py`
- `OpenAIProvider`: Implements `LLMProvider` for OpenAI `/v1/chat/completions` API schema.
- `AnthropicProvider`: Implements `LLMProvider` for Anthropic `/v1/messages` API schema.
- Built-in graceful fallbacks and offline mock testing without requiring paid API keys.
#### [MODIFY] `src/provider/factory.py`
- Wire `"openai"` and `"anthropic"` into provider factory.
#### [MODIFY] `config.json`
- Add optional `cloud_openai` and `cloud_anthropic` profiles to `config.json`.

### 2. Unit Tests for Cloud Adapters
#### [NEW] `tests/test_cloud_providers.py`
- Tests request formation, error handling, token tracking, and factory integration for cloud providers using mocks.
- Git commit message: `feat(phase-15): implement OpenAI and Anthropic cloud provider adapters and final production polish`.

---

## Verification Plan

### Automated Tests
- Run `python -m pytest tests/ -v` to ensure all 80+ tests pass with zero errors.
- Run `python -m scripts.reproduce --dry-run` to verify end-to-end pipeline execution.

### Manual Verification
- Test CLI adaptive mode with local Ollama models (`llama3.2:latest`, `qwen3:1.7b`).
- Verify Streamlit UI and FastAPI endpoints.
