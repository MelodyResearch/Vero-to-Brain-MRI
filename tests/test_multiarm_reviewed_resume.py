"""CPU orchestration fixtures; simulated model receipts are not research evidence."""
import json
import subprocess
import hashlib
import sys
from types import ModuleType
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import mri_rl.mlx_multiarm_train as module
from mri_rl.mlx_gspo_smoke import _sha256


@pytest.fixture
def arm_fixture(monkeypatch):
    private_root = Path(module.__file__).resolve().parents[2] / "local_data"
    with tempfile.TemporaryDirectory(dir=private_root) as directory:
        root = Path(directory)
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        image = root / "image.png"
        image.write_bytes(b"Synthetic orchestration fixture")
        rows = [{"id": f"row-{i}", "ability": "mri_sequence",
                 "data_source": "OpenNeuro_ds007045", "images": [str(image)],
                 "reward_model": {"ground_truth": "FLAIR"},
                 "extra_info": {"reward_type": "mri_sequence", "patient_key": f"p{i}",
                                "question": "Which MRI sequence is shown?"}}
                for i in range(4)]
        records = root / "train.verl.jsonl"
        records.write_text("".join(json.dumps(row) + "\n" for row in rows))
        adapter = root / "start.safetensors"
        adapter.write_bytes(b"starting fixture")
        calls = []
        monkeypatch.setattr(module, "build_schedule", lambda *a, **k:
            (rows, {"arm": "uniform", "groups": 4, "seed": 7, "schedule_sha256": "testschedule"}))
        monkeypatch.setattr(module, "load_contract", lambda p:
            SimpleNamespace(system_instruction="Read MRI carefully."))

        def gate(**kwargs):
            if kwargs["execution_tier"] == "B":
                spec = json.loads(kwargs["contract_path"].read_text())
                if spec["starting_sha"] != _sha256(kwargs["expected_starting_adapter"]):
                    raise ValueError("Stale child starting adapter")
                if kwargs["expected_review_sha256"] != "d" * 64:
                    raise ValueError("Child review digest mismatch")
                assert kwargs["launch_budget"]["max_groups"] == 1
                return f"child-{spec['index']}"
            return "parent-contract"
        monkeypatch.setattr(module, "require_launch_preflight", gate)

        def simulate_child(cmd, **kwargs):
            calls.append(cmd)
            flags = dict(zip(cmd[3::2], cmd[4::2]))
            out = Path(flags["--output"])
            index = int(out.name.split("_")[1])
            start = Path(flags["--adapter-checkpoint"])
            child = f"child-{index}"
            out.mkdir(exist_ok=True)
            assert not any(out.iterdir()), "real smoke output preflight requires empty output"
            rendered = "<|im_start|>system\nRead MRI carefully.\n<|im_start|>user\n<|image_pad|>Which MRI sequence is shown?\n<|im_start|>assistant\n"
            prompt_ids = [[1, 2, 3]]
            raw = [{"record_id": rows[index]["id"], "image_sha256": _sha256(image),
                    "group_index": 0, "rendered_prompt": rendered, "prompt_token_ids": prompt_ids, "rollouts": [
                {"text": "<think>image</think><answer>\\boxed{FLAIR}</answer>", "token_ids": [1]},
                {"text": "<think>image</think><answer>\\boxed{wrong}</answer>", "token_ids": [2]}]}]
            (out / "raw_rollouts.json").write_text(json.dumps(raw))
            before, after = out / "before_adapters.safetensors", out / "after_adapters.safetensors"
            before.write_bytes(start.read_bytes())
            after.write_bytes(start.read_bytes() + b"updated")
            reload_path = out / "reload_inference_receipt.json"
            reload_path.write_text(json.dumps({"loaded_adapter_sha256": _sha256(after),
                "contract_id": child, "image_sha256": _sha256(image),
                "actual_model_generation": True, "exact_adapter_tensor_match": True,
                "fresh_model_loaded": True, "generated_text": "synthetic test"}))
            receipt = {"contract_id": child, "review_sha256": "d" * 64,
                "starting_adapter_sha256": _sha256(start), "seed": 7 + index,
                "reward_profile": "direct_vero", "temperature": 1.0, "max_tokens": 256,
                "group_size": 2, "groups_attempted": 1,
                "training_records_sha256": _sha256(Path(flags["--records"])),
                "science_claim": False, "policy_updated": True,
                "status": "TRAINING_COMPLETED_ENGINEERING_SMOKE", "optimizer_steps": 1,
                "gradient_l2": 0.1, "checkpoint_reloaded": True,
                "old_current_logp_max_abs_errors": [0.0, 0.0],
                "before_adapter_sha256": _sha256(before), "after_adapter_sha256": _sha256(after),
                "reload_inference_receipt_sha256": _sha256(reload_path),
                "attempts": [{"source": rows[index]["data_source"], "route": "mri_sequence",
                "group_index": 0, "prompt_tokens": 3, "image_sha256": _sha256(image),
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "prompt_token_ids_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
                "response_tokens": [1, 1], "rewards": [{"score": 1., "accuracy": 1., "format": 1.},
                                            {"score": .2, "accuracy": 0., "format": 1.}]}]}
            (out / "receipt.json").write_text(json.dumps(receipt))
            return subprocess.CompletedProcess(cmd, 0, stdout="synthetic child finished")
        monkeypatch.setattr(module.subprocess, "run", simulate_child)
        args = dict(model=model, records=records, output=root / "arm", starting_adapter=adapter,
                    arm="uniform", groups=4, contract_path=root / "parent.json",
                    dev_records=root / "dev.jsonl", expected_review_sha256="e" * 64)

        def authorize(index, start):
            contract = root / f"child-{index}.json"
            contract.write_text(json.dumps({"index": index, "starting_sha": _sha256(start)}))
            return {"contract_path": contract, "review_verdict_path": root / f"review-{index}.json",
                    "expected_review_sha256": "d" * 64}
        yield args, authorize, calls


def test_missing_child_pauses_durably_without_subprocess(arm_fixture):
    args, _, calls = arm_fixture
    result = module.run(**args)
    assert result["status"] == "PAUSED_FOR_CHILD_REVIEW"
    assert result["completed_prompt_groups"] == 0
    assert result["pending_child_review"]["step"] == 0
    assert calls == []
    assert (args["output"] / "arm_receipt_PRIVATE.json").exists()


def test_reviewed_child_runs_then_resume_preserves_prefix(arm_fixture):
    args, authorize, calls = arm_fixture
    result = module.run(**args, child_authorizations={0: authorize(0, args["starting_adapter"])})
    assert result["optimizer_steps"] == 1
    assert result["completed_prompt_groups"] == 1
    assert result["pending_child_review"]["step"] == 1
    before_raw = (args["output"] / "step_00/raw_rollouts.json").read_bytes()
    next_adapter = args["output"] / "step_00/after_adapters.safetensors"
    result = module.run(**args, resume=True, child_authorizations={1: authorize(1, next_adapter)})
    assert result["optimizer_steps"] == 2 and result["completed_prompt_groups"] == 2
    assert len(calls) == 2
    assert (args["output"] / "step_00/raw_rollouts.json").read_bytes() == before_raw


def test_stale_child_authorization_blocks_before_next_model(arm_fixture):
    args, authorize, calls = arm_fixture
    stale = authorize(1, args["starting_adapter"])
    module.run(**args, child_authorizations={0: authorize(0, args["starting_adapter"])})
    with pytest.raises(ValueError, match="Stale child"):
        module.run(**args, resume=True, child_authorizations={1: stale})
    assert len(calls) == 1


@pytest.mark.parametrize("artifact", ["after_adapters.safetensors", "raw_rollouts.json"])
def test_resume_rejects_changed_completed_evidence(arm_fixture, artifact):
    args, authorize, calls = arm_fixture
    module.run(**args, child_authorizations={0: authorize(0, args["starting_adapter"])})
    path = args["output"] / "step_00" / artifact
    if artifact.endswith("json"):
        raw = json.loads(path.read_text())
        raw[0]["rollouts"][0]["text"] = "tampered"
        path.write_text(json.dumps(raw))
    else:
        path.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        module.run(**args, resume=True)
    assert len(calls) == 1


def test_completed_arm_has_finite_legitimate_path(arm_fixture):
    args, authorize, calls = arm_fixture
    start = args["starting_adapter"]
    for index in range(4):
        result = module.run(**args, resume=index > 0,
                            child_authorizations={index: authorize(index, start)})
        start = args["output"] / f"step_{index:02d}" / "after_adapters.safetensors"
    assert result["status"] == "TRAINING_COMPLETED_BOUNDED_ARM"
    assert result["optimizer_steps"] == 4 and len(calls) == 4
    assert result["pending_child_review"] is None


@pytest.mark.parametrize("mutation", ["before", "parity", "prompt", "token_inventory"])
def test_resume_rejects_invalid_optimizer_or_prompt_linkage(arm_fixture, mutation):
    args, authorize, _ = arm_fixture
    module.run(**args, child_authorizations={0: authorize(0, args["starting_adapter"])})
    out = args["output"] / "step_00"
    receipt_path = out / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if mutation == "before":
        before = out / "before_adapters.safetensors"
        before.write_bytes(b"different starting weights")
        receipt["before_adapter_sha256"] = _sha256(before)
    elif mutation == "parity":
        receipt["old_current_logp_max_abs_errors"] = [0.0, float("nan")]
    elif mutation == "prompt":
        raw_path = out / "raw_rollouts.json"
        raw = json.loads(raw_path.read_text())
        raw[0]["rendered_prompt"] += "changed"
        raw_path.write_text(json.dumps(raw))
    else:
        receipt["attempts"][0]["response_tokens"] = [2, 2]
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        module.run(**args, resume=True)


def test_parent_pending_layout_reaches_real_smoke_preload_output_check(arm_fixture, monkeypatch):
    """Run actual smoke.run through its real empty-output check, stop at load.

    MLX imports are inert stubs so this is a CPU integration test, not a model run.
    """
    import mri_rl.mlx_gspo_smoke as smoke
    args, _, _ = arm_fixture
    pending = module.run(**args)["pending_child_review"]
    records, output = Path(pending["records"]), Path(pending["output"])
    assert records.parent != output and records.name == "train.verl.jsonl"
    assert not output.exists()
    names = ["mlx", "mlx.core", "mlx.nn", "mlx.optimizers", "mlx.utils", "mlx_vlm",
             "mlx_vlm.prompt_utils", "mlx_vlm.trainer", "mlx_vlm.trainer.common",
             "mlx_vlm.trainer.common.model", "mlx_vlm.trainer.common.utils", "mlx_vlm.utils"]
    for name in names:
        fake = ModuleType(name)
        fake.__path__ = []
        monkeypatch.setitem(sys.modules, name, fake)
    for parent, child in [("mlx", "core"), ("mlx", "nn"), ("mlx", "optimizers")]:
        setattr(sys.modules[parent], child, sys.modules[parent + "." + child])
    def reached_load(*a, **k):
        raise RuntimeError("REACHED_MODEL_LOAD_AFTER_REAL_OUTPUT_CHECK")
    sys.modules["mlx.core"].random = SimpleNamespace(seed=lambda seed: None)
    sys.modules["mlx.utils"].tree_flatten = lambda x: []
    sys.modules["mlx.utils"].tree_map = lambda *a: None
    sys.modules["mlx_vlm"].generate = lambda *a, **k: None
    sys.modules["mlx_vlm"].load = reached_load
    sys.modules["mlx_vlm.prompt_utils"].apply_chat_template = lambda *a, **k: None
    sys.modules["mlx_vlm.trainer.common.model"].prepare_model_for_training = lambda *a, **k: None
    sys.modules["mlx_vlm.trainer.common.utils"].save_trainable_weights = lambda *a, **k: None
    sys.modules["mlx_vlm.utils"].prepare_inputs = lambda *a, **k: None
    sys.modules["mlx_vlm.utils"].should_add_special_tokens = lambda *a, **k: None
    monkeypatch.setattr(smoke, "load_contract", lambda p: SimpleNamespace(
        system_instruction="Read MRI carefully.", system_instruction_sha256="a" * 64))
    monkeypatch.setattr(smoke, "require_launch_preflight", lambda **k: "test-contract")
    monkeypatch.setattr(smoke, "_records_for_route", lambda *a: [json.loads(records.read_text())])
    monkeypatch.setattr(smoke, "ResourceGate", lambda p: SimpleNamespace(before_unit=lambda *a: None))
    with pytest.raises(RuntimeError, match="REACHED_MODEL_LOAD_AFTER_REAL_OUTPUT_CHECK"):
        smoke.run(args["model"], records, output, "mri_sequence", 2, 256, 7, max_groups=1,
                  adapter_checkpoint=args["starting_adapter"], reward_profile="direct_vero",
                  contract_path=args["contract_path"])


@pytest.mark.parametrize("mutation", ["wrong_question", "extra_image"])
def test_rehashed_wrong_question_or_image_inventory_rejected(arm_fixture, mutation):
    args, authorize, _ = arm_fixture
    module.run(**args, child_authorizations={0: authorize(0, args["starting_adapter"])})
    out = args["output"] / "step_00"
    raw_path, receipt_path = out / "raw_rollouts.json", out / "receipt.json"
    raw, receipt = json.loads(raw_path.read_text()), json.loads(receipt_path.read_text())
    if mutation == "wrong_question":
        raw[0]["rendered_prompt"] = raw[0]["rendered_prompt"].replace(
            "Which MRI sequence is shown?", "Which unrelated animal is shown?")
    else:
        raw[0]["rendered_prompt"] = raw[0]["rendered_prompt"].replace(
            "<|image_pad|>", "<|image_pad|><|image_pad|>")
    receipt["attempts"][0]["rendered_prompt_sha256"] = hashlib.sha256(
        raw[0]["rendered_prompt"].encode()).hexdigest()
    raw_path.write_text(json.dumps(raw))
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="raw prompt/image/token inventory"):
        module.run(**args, resume=True)
