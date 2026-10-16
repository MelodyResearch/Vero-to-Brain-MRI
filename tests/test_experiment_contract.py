import json
import pytest
import tempfile
from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from mri_rl.experiment_contract import (
    ExperimentContract, QWEN_BASE_REVISION, STRICT_GRAMMAR, STRICT_PARSER,
    VERSION, load_contract, preflight, require_launch_preflight,
    resolve_artifact_paths, review_scope,
    sha256_bytes, sha256_file,
)
from mri_rl.resource_supervisor import ResourceSample
import mri_rl.experiment_contract as contract_module
from mri_rl.mlx_sft_warmstart import run as run_sft
from mri_rl.mlx_multiarm_train import run as run_multiarm
import mri_rl.mlx_sft_warmstart as sft_module
import mri_rl.mlx_multiarm_train as multiarm_module


def _fixture(tmp_path: Path):
    image = tmp_path / "real.png"
    Image.new("RGB", (8, 8), "gray").save(image)
    artifacts = {}
    content = {
        "model_config": '{"model_type":"qwen3_vl"}',
        "processor_config": '{"image_processor_type":"Qwen2VLImageProcessorFast"}',
        "tokenizer_config": '{"tokenizer_class":"Qwen2Tokenizer"}',
        "chat_template": "template-v1",
        "rendered_prompt": "<|im_start|>system\nObserve MRI carefully.\n<|im_start|>user\n<|image_pad|>Which sequence is shown?\n<|im_start|>assistant\n",
        "prompt_token_ids": "[1,2,3]",
        "image_token_ids": "[999]",
        "image_bytes": image.read_bytes(),
    }
    for name, value in content.items():
        path = tmp_path / f"{name}.bin"
        path.write_bytes(value if isinstance(value, bytes) else value.encode())
        artifacts[name] = path

    def row(patient, split, answer):
        return {
            "id": f"{patient}-{split}", "data_source": "OpenNeuro_ds007045",
            "ability": "mri_sequence", "images": [str(image)],
            "prompt": [{"role": "user", "content": [{"type": "image", "image": str(image)},
                        {"type": "text", "text": "Which sequence is shown?"}]}],
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"reward_type": "mri_sequence", "patient_key": patient,
                           "patient_lineage": "BIDS_SUBJECT_VERIFIED",
                           "internal_split": split},
        }

    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text(json.dumps(row("patient-a", "train", "FLAIR")) + "\n")
    dev.write_text(json.dumps(row("patient-b", "internal_dev", "contrast-enhanced T1")) + "\n")
    contract = ExperimentContract(
        version=VERSION, model_id="Qwen/Qwen3-VL-8B-Instruct",
        model_revision=QWEN_BASE_REVISION,
        processor_revision=QWEN_BASE_REVISION,
        tokenizer_revision=QWEN_BASE_REVISION,
        model_config_sha256=sha256_file(artifacts["model_config"]),
        processor_config_sha256=sha256_file(artifacts["processor_config"]),
        tokenizer_config_sha256=sha256_file(artifacts["tokenizer_config"]),
        chat_template_sha256=sha256_file(artifacts["chat_template"]),
        system_instruction="Observe MRI carefully.",
        system_instruction_sha256=sha256_bytes(b"Observe MRI carefully."),
        assistant_prefix="<|im_start|>assistant\n", enable_thinking=False,
        add_generation_prompt=True, response_includes_prefill=False,
        generation={"temperature": 0, "max_tokens": 256},
        strict_answer_grammar=STRICT_GRAMMAR, reward_parser_version=STRICT_PARSER,
        reward_profile="direct_vero", diagnostic_record_id="patient-a-train",
        train_manifest_sha256=sha256_file(train),
        dev_manifest_sha256=sha256_file(dev),
        rendered_prompt_sha256=sha256_file(artifacts["rendered_prompt"]),
        prompt_token_ids_sha256=sha256_file(artifacts["prompt_token_ids"]),
        image_token_ids_sha256=sha256_file(artifacts["image_token_ids"]),
        image_bytes_sha256=sha256_file(artifacts["image_bytes"]),
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(asdict(contract)))
    return contract_path, train, dev, artifacts


def _ac():
    return ResourceSample(monotonic_seconds=1.0, available_gib=8,
                          pressure_free_percent=50, power_source="AC")


def test_legacy_artifact_paths_resolve_independent_of_cwd(tmp_path, monkeypatch):
    contract_path = tmp_path / "contract.json"
    (tmp_path / "artifact_paths.json").write_text(json.dumps({
        "relative": "local_data/pinned/image.png",
        "absolute": str(tmp_path / "fixed.png"),
    }))
    monkeypatch.chdir(tmp_path)
    resolved = resolve_artifact_paths(contract_path)
    repository_root = Path(contract_module.__file__).resolve().parents[2]
    assert resolved["relative"] == repository_root / "local_data/pinned/image.png"
    assert resolved["absolute"] == tmp_path / "fixed.png"


def test_cpu_preflight_passes_but_dynamic_gates_remain_unverified(tmp_path):
    report = preflight(*_fixture(tmp_path), sample=_ac())
    assert report["cpu_preflight_passed"] is True
    assert report["dynamic_gates_verified"] is False
    assert report["substantial_job_launch_ready"] is False
    assert report["data"]["decoded_images"] == 2


def test_manifest_mutation_and_patient_overlap_fail_closed(tmp_path):
    contract, train, dev, artifacts = _fixture(tmp_path)
    row = json.loads(dev.read_text())
    row["extra_info"]["patient_key"] = "patient-a"
    dev.write_text(json.dumps(row) + "\n")
    report = preflight(contract, train, dev, artifacts, sample=_ac())
    assert "dev manifest hash mismatch" in report["failures"]
    assert "train/internal-dev patient overlap" in report["failures"]


def test_prompt_and_artifact_tampering_fail_closed(tmp_path):
    contract, train, dev, artifacts = _fixture(tmp_path)
    artifacts["rendered_prompt"].write_text("changed")
    report = preflight(contract, train, dev, artifacts, sample=_ac())
    assert "rendered_prompt: missing or hash mismatch" in report["failures"]


def test_substantial_run_blocked_on_battery(tmp_path):
    contract, train, dev, artifacts = _fixture(tmp_path)
    sample = ResourceSample(monotonic_seconds=1.0, available_gib=8,
                            pressure_free_percent=50, power_source="BATTERY",
                            battery_percent=69)
    report = preflight(contract, train, dev, artifacts, sample=sample)
    assert report["cpu_preflight_passed"] is False
    assert any("AC power" in failure for failure in report["failures"])


def test_wrong_reward_parser_version_is_rejected(tmp_path):
    contract, train, dev, artifacts = _fixture(tmp_path)
    raw = json.loads(contract.read_text())
    raw["reward_parser_version"] = "exploratory_semantic"
    contract.write_text(json.dumps(raw))
    try:
        preflight(contract, train, dev, artifacts, sample=_ac())
    except ValueError as exc:
        assert "strict reward parser" in str(exc)
    else:
        raise AssertionError("exploratory parser must not pass a direct Vero preflight")


def test_ac_header_with_discharge_detail_fails_closed(tmp_path, monkeypatch):
    contract, train, dev, artifacts = _fixture(tmp_path)
    monkeypatch.setattr(contract_module, "sample_macos", lambda: _ac())
    monkeypatch.setattr(contract_module.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="Now drawing from 'AC Power'\n61%; discharging"))
    report = preflight(contract, train, dev, artifacts)
    assert report["battery_discharge_detected"] is True
    assert report["cpu_preflight_passed"] is False


def test_both_substantial_runners_block_without_contract_before_output(tmp_path):
    sft_output = tmp_path / "sft"
    rl_output = tmp_path / "rl"
    try:
        run_sft(tmp_path / "model", tmp_path / "train.verl.jsonl", sft_output)
    except ValueError as exc:
        assert "Private contract" in str(exc)
    else:
        raise AssertionError("SFT must require a contract before loading MLX")
    try:
        run_multiarm(tmp_path / "model", tmp_path / "train.verl.jsonl", rl_output,
                     tmp_path / "start.safetensors", "uniform")
    except ValueError as exc:
        assert "Private contract" in str(exc)
    else:
        raise AssertionError("multiarm RL must require a contract before loading MLX")
    assert not sft_output.exists() and not rl_output.exists()


def _launch_paths(tmp_path):
    contract_path, train, dev, artifacts = _fixture(tmp_path)
    model_path = tmp_path / "model"
    model_path.mkdir()
    for name, filename in (("model_config", "config.json"),
                           ("processor_config", "preprocessor_config.json"),
                           ("tokenizer_config", "tokenizer_config.json"),
                           ("chat_template", "chat_template.json")):
        contents = artifacts[name].read_bytes()
        artifacts[name] = model_path / filename
        artifacts[name].write_bytes(contents)
    (tmp_path / "artifact_paths.json").write_text(json.dumps(
        {name: str(path) for name, path in artifacts.items()}))
    dynamic = tmp_path / "dynamic.json"
    dynamic.write_text(json.dumps({"contract_id": load_contract(contract_path).identity()}))
    review = tmp_path / "review.json"
    review.write_text("{}")
    return contract_path, train, dev, model_path, dynamic, review


def test_runner_gate_failure_precedes_model_load_and_output(tmp_path, monkeypatch):
    def battery_block(**kwargs):
        raise ValueError("CPU preflight failed: battery reports discharging")

    def dynamic_block(**kwargs):
        raise ValueError("Dynamic diagnostic lacks verified real-model gates")

    monkeypatch.setattr(sft_module, "load_contract", lambda path: SimpleNamespace(
        generation={"max_tokens": 256, "temperature": 0.0}))
    monkeypatch.setattr(sft_module, "require_launch_preflight", battery_block)
    monkeypatch.setattr(multiarm_module, "load_contract", lambda path:
                        SimpleNamespace(system_instruction="test system"))
    monkeypatch.setattr(multiarm_module, "require_launch_preflight", dynamic_block)
    try:
        run_sft(tmp_path / "missing_model", tmp_path / "missing_records",
                tmp_path / "sft_output", contract_path=tmp_path / "contract")
    except ValueError as exc:
        assert "battery reports discharging" in str(exc)
    else:
        raise AssertionError("SFT must block before model-load checks")
    try:
        run_multiarm(tmp_path / "missing_model", tmp_path / "missing_records",
                     tmp_path / "rl_output", tmp_path / "missing_adapter", "uniform",
                     contract_path=tmp_path / "contract")
    except ValueError as exc:
        assert "real-model gates" in str(exc)
    else:
        raise AssertionError("RL must block before adapter/model checks")
    assert not (tmp_path / "sft_output").exists()
    assert not (tmp_path / "rl_output").exists()



@pytest.fixture
def authorized_launch(monkeypatch):
    """Synthetic gate fixtures only; never evidence of an MRI experiment."""
    private_root = Path(contract_module.__file__).resolve().parents[2] / "local_data"
    private_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=private_root) as directory:
        directory = Path(directory)
        contract, train, dev, model, dynamic, review = _launch_paths(directory)
        adapter = directory / "start.safetensors"
        adapter.write_bytes(b"synthetic adapter bytes for gate unit test")
        args = dict(contract_path=contract, train=train, dev=dev, model_path=model,
                    dynamic_receipt_path=None, review_verdict_path=review,
                    expected_system="Observe MRI carefully.", expected_max_tokens=256,
                    expected_temperature=0.0, expected_starting_adapter=adapter,
                    execution_tier="B", phase="diagnostic",
                    launch_configuration={"runner": "synthetic_unit_test", "seed": 1},
                    launch_budget={"optimizer_steps": 1, "max_groups": 3,
                                   "rollouts_per_group": 2, "max_tokens": 256})
        monkeypatch.setattr(contract_module, "sample_macos", lambda: _ac())
        monkeypatch.setattr(contract_module.subprocess, "run", lambda *a, **k:
                            SimpleNamespace(stdout="Now drawing from 'AC Power'\n90%; charged"))
        _approve(args)
        yield args


def _approve(args, **extra):
    scope = review_scope(**{key: args[key] for key in (
        "contract_path", "execution_tier", "phase", "launch_budget",
        "expected_starting_adapter", "dynamic_receipt_path", "launch_configuration")})
    verdict = {**scope, "verdict": "PASS_WITH_LIMITATIONS",
               "reviewer_kind": "isolated_codex", "reviewer_id": "unit-test-reviewer",
               "reviewer_findings": "Synthetic gate test; no scientific evidence.",
               "limitations": "Only the bound diagnostic budget is approved.", **extra}
    args["review_verdict_path"].write_text(json.dumps(verdict))
    args["expected_review_sha256"] = sha256_file(args["review_verdict_path"])


def test_legitimate_tier_b_diagnostic_accepts_without_prior_update(authorized_launch):
    args = authorized_launch
    assert require_launch_preflight(**args) == load_contract(args["contract_path"]).identity()


def test_tier_a_cpu_runs_on_battery_without_reviewer(authorized_launch, monkeypatch):
    args = {**authorized_launch, "execution_tier": "A", "phase": "cpu",
            "review_verdict_path": None, "expected_review_sha256": None,
            "expected_starting_adapter": None, "launch_budget": None}
    monkeypatch.setattr(contract_module, "sample_macos", lambda: ResourceSample(
        monotonic_seconds=1.0, power_source="BATTERY", battery_percent=50))
    assert require_launch_preflight(**args)


def test_model_tier_blocks_discharge_before_launch(authorized_launch, monkeypatch):
    monkeypatch.setattr(contract_module.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(stdout="AC Power\n61%; discharging"))
    with pytest.raises(ValueError, match="discharging"):
        require_launch_preflight(**authorized_launch)


@pytest.mark.parametrize("field,value,error", [
    ("expected_review_sha256", None, "review digest"),
    ("expected_review_sha256", "0" * 64, "review digest"),
    ("expected_temperature", 1.0, "temperature differs"),
    ("expected_system", "changed prompt", "system instruction differs"),
    ("execution_tier", "invented", "Unknown execution tier"),
])
def test_invalid_authorization_rejected(authorized_launch, field, value, error):
    with pytest.raises(ValueError, match=error):
        require_launch_preflight(**{**authorized_launch, field: value})


def test_forged_review_file_cannot_replace_out_of_band_digest(authorized_launch):
    path = authorized_launch["review_verdict_path"]
    fake = json.loads(path.read_text())
    fake.update(reviewer_kind="claude_code", reviewer_id="pretend Claude")
    path.write_text(json.dumps(fake))
    with pytest.raises(ValueError, match="review digest"):
        require_launch_preflight(**authorized_launch)


@pytest.mark.parametrize("mutation", ["adapter", "source", "contract", "budget"])
def test_stale_review_binding_rejected(authorized_launch, monkeypatch, mutation):
    args = authorized_launch
    if mutation == "adapter":
        args["expected_starting_adapter"].write_bytes(b"different adapter")
    elif mutation == "source":
        real_scope = contract_module.review_scope
        def stale(**kwargs):
            scope = real_scope(**kwargs)
            scope["source_sha256"]["mlx_gspo_smoke.py"] = "0" * 64
            return scope
        monkeypatch.setattr(contract_module, "review_scope", stale)
    elif mutation == "contract":
        value = json.loads(args["contract_path"].read_text())
        value["generation"]["temperature"] = 0.5
        args["contract_path"].write_text(json.dumps(value))
        args["expected_temperature"] = 0.5
    else:
        args["launch_budget"] = {**args["launch_budget"], "max_groups": 2}
    with pytest.raises(ValueError, match="stale or mismatches"):
        require_launch_preflight(**args)


@pytest.mark.parametrize("key,value", [("optimizer_steps", 2), ("max_groups", 4),
                                        ("rollouts_per_group", 4), ("max_tokens", 1024)])
def test_diagnostic_budget_cannot_expand(authorized_launch, key, value):
    args = authorized_launch
    args["launch_budget"] = {**args["launch_budget"], key: value}
    _approve(args)
    with pytest.raises(ValueError, match="budget"):
        require_launch_preflight(**args)


def test_continuation_requires_dynamic_evidence(authorized_launch):
    args = authorized_launch
    args["phase"] = "continuation"
    _approve(args)
    with pytest.raises(ValueError, match="dynamic receipt"):
        require_launch_preflight(**args)


def _dynamic_fixture(args):
    """Structurally valid simulated receipt for branch testing, not MLX evidence."""
    directory = args["contract_path"].parent
    contract = load_contract(args["contract_path"])
    before_path = directory / "before_adapters.safetensors"
    after_path = directory / "after_adapters.safetensors"
    before_path.write_bytes(b"unit-test-before")
    after_path.write_bytes(b"unit-test-after")
    before, after = sha256_file(before_path), sha256_file(after_path)
    raw = [{"group_index": 0, "record_id": "patient-a-train", "rollouts": [
        {"text": "<think>image</think><answer>\\boxed{FLAIR}</answer>", "token_ids": [1]},
        {"text": "<think>image</think><answer>\\boxed{definitely-wrong}</answer>", "token_ids": [2]}]}]
    rendered = resolve_artifact_paths(args["contract_path"])["rendered_prompt"].read_text()
    raw[0].update(rendered_prompt=rendered, prompt_token_ids=[[1, 2, 3]],
                  image_sha256=contract.image_bytes_sha256)
    raw_path = directory / "raw_rollouts.json"
    raw_path.write_text(json.dumps(raw))
    source = {
        "status": "TRAINING_COMPLETED_ENGINEERING_SMOKE", "reward_profile": "direct_vero",
        "contract_id": contract.identity(), "temperature": 0.0, "max_tokens": 256,
        "image_source": "OpenNeuro_ds007045", "policy_updated": True,
        "system_prompt_sha256": contract.system_instruction_sha256,
        "model_config_sha256": contract.model_config_sha256,
        "training_records_sha256": contract.train_manifest_sha256,
        "selected_active_group": 0,
        "optimizer_steps": 1, "groups_attempted": 1, "science_claim": False,
        "starting_adapter_sha256": sha256_file(args["expected_starting_adapter"]),
        "before_adapter_sha256": before, "after_adapter_sha256": after, "gradient_l2": 1.0,
        "attempts": [{"group_index": 0, "source": "OpenNeuro_ds007045", "route": "mri_sequence",
                      "active": True, "image_sha256": contract.image_bytes_sha256,
                      "rendered_prompt_sha256": sha256_bytes(rendered.encode()),
                      "prompt_token_ids_sha256": sha256_bytes(json.dumps([[1, 2, 3]]).encode()),
                      "rewards": [{"score": 1., "accuracy": 1., "format": 1.},
                                                 {"score": .2, "accuracy": 0., "format": 1.}]}]}
    source_path = directory / "receipt.json"
    source_path.write_text(json.dumps(source))
    reload_path = directory / "reload.json"
    reload_path.write_text(json.dumps({"loaded_adapter_sha256": after,
        "image_sha256": contract.image_bytes_sha256, "actual_model_generation": True,
        "generated_text": "unit-test simulated reload"}))
    dynamic = {"contract_id": contract.identity(), "actual_model_generation": True,
        "real_mri_image_verified": True, "strict_positive_reward_observed": True,
        "strict_negative_reward_observed": True, "finite_gradients": True,
        "weight_changed": True, "checkpoint_reloaded": True,
        "task_reward_values": [1., 0.], "gradient_norm": 1.,
        "starting_adapter_sha256": source["starting_adapter_sha256"],
        "before_adapter_sha256": before, "after_adapter_sha256": after,
        "reloaded_adapter_sha256": after, "source_receipt_path": str(source_path),
        "source_receipt_sha256": sha256_file(source_path),
        "raw_rollouts_sha256": sha256_file(raw_path),
        "reload_inference_receipt_path": str(reload_path),
        "reload_inference_receipt_sha256": sha256_file(reload_path)}
    path = directory / "dynamic.json"
    path.write_text(json.dumps(dynamic))
    args.update(dynamic_receipt_path=path, phase="continuation")
    _approve(args)


def test_valid_continuation_has_real_accept_branch(authorized_launch):
    _dynamic_fixture(authorized_launch)
    assert require_launch_preflight(**authorized_launch)


def test_tier_c_requires_explicit_stronger_review(authorized_launch):
    _dynamic_fixture(authorized_launch)
    authorized_launch["execution_tier"] = "C"
    _approve(authorized_launch)
    with pytest.raises(ValueError, match="Tier C needs explicit"):
        require_launch_preflight(**authorized_launch)
    _approve(authorized_launch, substantial_training_authorized=True,
             acceptance_criteria="Prespecified bounded synthetic test",
             matched_budget_checks="Equal updates/rollouts/precision")
    assert require_launch_preflight(**authorized_launch)


@pytest.mark.parametrize("mutation", ["flags", "source", "adapter", "reward", "reload"])
def test_dynamic_forgery_rejected_even_with_reviewed_file(authorized_launch, mutation):
    args = authorized_launch
    _dynamic_fixture(args)
    path = args["dynamic_receipt_path"]
    dynamic = json.loads(path.read_text())
    if mutation == "flags":
        dynamic["finite_gradients"] = False
    elif mutation == "source":
        source = Path(dynamic["source_receipt_path"])
        source.write_text("{}")
        dynamic["source_receipt_sha256"] = sha256_file(source)
    elif mutation == "adapter":
        (path.parent / "after_adapters.safetensors").write_bytes(b"tampered")
    elif mutation == "reward":
        raw_path = path.parent / "raw_rollouts.json"
        raw = json.loads(raw_path.read_text())
        raw[0]["rollouts"][0]["text"] = "wrong"
        raw_path.write_text(json.dumps(raw))
        dynamic["raw_rollouts_sha256"] = sha256_file(raw_path)
    else:
        reload_path = Path(dynamic["reload_inference_receipt_path"])
        value = json.loads(reload_path.read_text())
        value["loaded_adapter_sha256"] = "0" * 64
        reload_path.write_text(json.dumps(value))
        dynamic["reload_inference_receipt_sha256"] = sha256_file(reload_path)
    path.write_text(json.dumps(dynamic))
    _approve(args)
    with pytest.raises(ValueError):
        require_launch_preflight(**args)
