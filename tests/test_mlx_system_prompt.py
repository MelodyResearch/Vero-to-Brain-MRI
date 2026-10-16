"""CPU-only tests for the pinned official Vero prompt override."""

import hashlib

import pytest

from mri_rl import mlx_gspo_smoke, mlx_multiarm_train


def test_official_prompt_loader_returns_exact_text_and_pinned_hash(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Exact Vero prompt.\n")
    expected_sha = "76b0d4522bfb9fa67edf82ac2342da9292879f4e953fa6ee8b63d4881f8fad0d"
    # The fixture SHA is independent of the production loader.
    assert hashlib.sha256(prompt.read_bytes()).hexdigest() == expected_sha
    assert mlx_gspo_smoke.load_system_prompt_file(prompt, expected_sha=expected_sha) == (
        "Exact Vero prompt.\n", expected_sha)


def test_official_prompt_loader_rejects_modified_or_empty_file(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("altered")
    with pytest.raises(ValueError, match="SHA-256"):
        mlx_gspo_smoke.load_system_prompt_file(prompt, expected_sha="0" * 64)
    prompt.write_text("")
    with pytest.raises(ValueError, match="empty"):
        mlx_gspo_smoke.load_system_prompt_file(prompt, expected_sha=hashlib.sha256(b"").hexdigest())


def test_multiarm_forwards_only_explicit_system_prompt_override(tmp_path):
    prompt = tmp_path / "official.txt"
    assert mlx_multiarm_train.system_prompt_cli_args(None) == []
    assert mlx_multiarm_train.system_prompt_cli_args(prompt) == ["--system-prompt-file", str(prompt)]
