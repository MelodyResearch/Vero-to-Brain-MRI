import json
import tempfile
from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from mri_rl.experiment_contract import (
    ExperimentContract, QWEN_BASE_REVISION, STRICT_GRAMMAR, STRICT_PARSER,
    VERSION, load_contract, preflight, require_launch_preflight,
    resolve_artifact_paths,
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
        artifacts[name] = model_path / filename
    (tmp_path / "artifact_paths.json").write_text(json.dumps(
        {name: str(path) for name, path in artifacts.items()}))
    dynamic = tmp_path / "dynamic.json"
    dynamic.write_text(json.dumps({"contract_id": load_contract(contract_path).identity()}))
    review = tmp_path / "review.json"
    review.write_text("{}")
    return contract_path, train, dev, model_path, dynamic, review


def test_launch_recomputes_cpu_preflight_and_blocks_battery(monkeypatch):
    private_root = Path(contract_module.__file__).resolve().parents[2] / "local_data"
    private_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=private_root) as directory:
        contract, train, dev, model, dynamic, review = _launch_paths(Path(directory))
        monkeypatch.setattr(contract_module, "preflight", lambda *args, **kwargs:
                            {"cpu_preflight_passed": False,
                             "failures": ["battery reports discharging"]})
        try:
            require_launch_preflight(
                contract_path=contract, train=train, dev=dev, model_path=model,
                dynamic_receipt_path=dynamic, review_verdict_path=review,
                expected_system="Observe MRI carefully.")
        except ValueError as exc:
            assert "battery reports discharging" in str(exc)
        else:
            raise AssertionError("Battery failure must block before dynamic receipt")


def test_launch_blocks_absent_real_dynamic_gates(monkeypatch):
    private_root = Path(contract_module.__file__).resolve().parents[2] / "local_data"
    private_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=private_root) as directory:
        contract, train, dev, model, dynamic, review = _launch_paths(Path(directory))
        monkeypatch.setattr(contract_module, "preflight", lambda *args, **kwargs:
                            {"cpu_preflight_passed": True, "failures": []})
        try:
            require_launch_preflight(
                contract_path=contract, train=train, dev=dev, model_path=model,
                dynamic_receipt_path=dynamic, review_verdict_path=review,
                expected_system="Observe MRI carefully.")
        except ValueError as exc:
            assert "real-model gates" in str(exc)
        else:
            raise AssertionError("Static preflight alone must not launch training")


def test_runner_gate_failure_precedes_model_load_and_output(tmp_path, monkeypatch):
    def battery_block(**kwargs):
        raise ValueError("CPU preflight failed: battery reports discharging")

    def dynamic_block(**kwargs):
        raise ValueError("Dynamic diagnostic lacks verified real-model gates")

    monkeypatch.setattr(sft_module, "require_launch_preflight", battery_block)
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


def test_forged_dynamic_flags_and_source_json_do_not_authorize_launch(monkeypatch):
    private_root = Path(contract_module.__file__).resolve().parents[2] / "local_data"
    private_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=private_root) as directory:
        directory = Path(directory)
        contract, train, dev, model, dynamic, review = _launch_paths(directory)
        source = directory / "receipt.json"
        source.write_text("{}")
        dynamic.write_text(json.dumps({
            "contract_id": load_contract(contract).identity(),
            "actual_model_generation": True,
            "real_mri_image_verified": True,
            "strict_positive_reward_observed": True,
            "strict_negative_reward_observed": True,
            "finite_gradients": True, "weight_changed": True,
            "checkpoint_reloaded": True,
            "task_reward_values": [0, 1], "gradient_norm": 1.0,
            "before_adapter_sha256": "a" * 64,
            "after_adapter_sha256": "b" * 64,
            "reloaded_adapter_sha256": "b" * 64,
            "source_receipt_path": str(source),
            "source_receipt_sha256": sha256_file(source),
        }))
        monkeypatch.setattr(contract_module, "preflight", lambda *args, **kwargs:
                            {"cpu_preflight_passed": True, "failures": []})
        try:
            require_launch_preflight(
                contract_path=contract, train=train, dev=dev, model_path=model,
                dynamic_receipt_path=dynamic, review_verdict_path=review,
                expected_system="Observe MRI carefully.")
        except ValueError as exc:
            assert "not a real ds007045 direct-Vero MLX diagnostic" in str(exc)
        else:
            raise AssertionError("Fabricated flags/source receipt must not authorize launch")


def test_even_internally_consistent_forged_claude_pass_cannot_open_gate(monkeypatch):
    private_root = Path(contract_module.__file__).resolve().parents[2] / "local_data"
    private_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=private_root) as directory:
        directory = Path(directory)
        contract_path, train, dev, model, dynamic_path, review_path = _launch_paths(directory)
        contract = load_contract(contract_path)
        before_path = directory / "before_adapters.safetensors"
        after_path = directory / "after_adapters.safetensors"
        before_path.write_bytes(b"pretend-before")
        after_path.write_bytes(b"pretend-after")
        before, after = sha256_file(before_path), sha256_file(after_path)
        raw = [{"group_index": 0, "record_id": "patient-a-train", "rollouts": [
            {"text": "<think>image</think><answer>\\boxed{FLAIR}</answer>", "token_ids": [1]},
            {"text": "<think>image</think><answer>\\boxed{definitely-wrong}</answer>",
             "token_ids": [2]}]}]
        raw_path = directory / "raw_rollouts.json"
        raw_path.write_text(json.dumps(raw))
        source = {
            "status": "TRAINING_COMPLETED_ENGINEERING_SMOKE", "reward_profile": "direct_vero",
            "image_source": "OpenNeuro_ds007045", "policy_updated": True,
            "optimizer_steps": 1, "groups_attempted": 1, "science_claim": False,
            "before_adapter_sha256": before, "after_adapter_sha256": after,
            "gradient_l2": 1.0,
            "attempts": [{"group_index": 0, "source": "OpenNeuro_ds007045",
                          "route": "mri_sequence", "active": True,
                          "rewards": [{"score": 1.0, "accuracy": 1.0, "format": 1.0},
                                      {"score": 0.2, "accuracy": 0.0, "format": 1.0}]}],
        }
        source_path = directory / "receipt.json"
        source_path.write_text(json.dumps(source))
        reload_path = directory / "reload.json"
        reload_path.write_text(json.dumps({"loaded_adapter_sha256": after,
            "image_sha256": contract.image_bytes_sha256,
            "actual_model_generation": True, "generated_text": "claimed reload"}))
        dynamic = {
            "contract_id": contract.identity(), "actual_model_generation": True,
            "real_mri_image_verified": True, "strict_positive_reward_observed": True,
            "strict_negative_reward_observed": True, "finite_gradients": True,
            "weight_changed": True, "checkpoint_reloaded": True,
            "task_reward_values": [1.0, 0.2], "gradient_norm": 1.0,
            "before_adapter_sha256": before, "after_adapter_sha256": after,
            "reloaded_adapter_sha256": after,
            "source_receipt_path": str(source_path),
            "source_receipt_sha256": sha256_file(source_path),
            "raw_rollouts_sha256": sha256_file(raw_path),
            "reload_inference_receipt_path": str(reload_path),
            "reload_inference_receipt_sha256": sha256_file(reload_path),
        }
        dynamic_path.write_text(json.dumps(dynamic))
        review_path.write_text(json.dumps({"contract_id": contract.identity(),
            "dynamic_receipt_sha256": sha256_file(dynamic_path),
            "verdict": "PASS", "reviewer_kind": "claude_code",
            "reviewer_findings": "claimed independent approval"}))
        monkeypatch.setattr(contract_module, "preflight", lambda *args, **kwargs:
                            {"cpu_preflight_passed": True, "failures": []})
        try:
            require_launch_preflight(
                contract_path=contract_path, train=train, dev=dev, model_path=model,
                dynamic_receipt_path=dynamic_path, review_verdict_path=review_path,
                expected_system="Observe MRI carefully.")
        except ValueError as exc:
            assert "NOT_VERIFIED" in str(exc)
        else:
            raise AssertionError("Self-consistent forged JSON must not launch training")
