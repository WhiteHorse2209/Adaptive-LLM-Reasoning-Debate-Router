"""Unit tests for the end-to-end reproducibility and demonstration pipeline."""

from scripts.reproduce import MockReproduceProvider, run_reproducibility
from src.provider.models import LLMRequest


def test_mock_reproduce_provider():
    provider = MockReproduceProvider()
    assert provider.health_check() is True

    # Test judge prompt detection
    req_judge = LLMRequest(prompt="Please adjudicate this supreme judge debate.")
    resp_judge = provider.generate(req_judge)
    assert "FINAL_VERDICT" in resp_judge.text
    assert resp_judge.token_usage.total_tokens > 0

    # Test difficulty estimation prompt detection
    req_easy = LLMRequest(prompt="Assess the difficulty of this cupcake question.")
    resp_easy = provider.generate(req_easy)
    assert "<difficulty>EASY</difficulty>" in resp_easy.text

    req_hard = LLMRequest(prompt="Assess the difficulty of this theseus paradox question.")
    resp_hard = provider.generate(req_hard)
    assert "<difficulty>HARD</difficulty>" in resp_hard.text


def test_reproducibility_pipeline_dry_run():
    # Run full dry-run reproducibility pipeline and verify success
    success = run_reproducibility(dry_run=True)
    assert success is True
