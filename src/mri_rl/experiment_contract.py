"""Versioned, fail-closed CPU preflight for future MRI SFT/RL experiments.

This module does not certify model-generated rewards or optimizer behavior.
Those require a separate real-image, weight-updating diagnostic receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mri_rl.resource_supervisor import ResourceSample, sample_macos
from mri_rl.vero_mri_reward import compute_score_from_data_source


VERSION = "mri-prompt-reward-v1"
EXECUTION_POLICY = "mri-execution-tiers-v1"
QWEN_BASE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
STRICT_PARSER = "vero_mri_reward.strict_tags_boxed_v1"
STRICT_GRAMMAR = "<think>nonempty</think><answer>\\boxed{one answer}</answer>"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact_paths(contract_path: Path) -> dict[str, Path]:
    """Read current absolute paths and legacy repository-relative paths."""
    repository_root = Path(__file__).resolve().parents[2]
    declared = json.loads((contract_path.parent / "artifact_paths.json").read_text())
    paths = {}
    for name, value in declared.items():
        path = Path(value)
        paths[name] = path if path.is_absolute() else repository_root / path
    return paths


def materialize_local_contract(model_path: Path, train: Path, dev: Path,
                               output: Path, system_instruction: str, *,
                               enable_thinking: bool = False,
                               max_tokens: int = 256,
                               temperature: float = 0.0) -> tuple[Path, dict[str, Path]]:
    """Build a private CPU prompt specimen from pinned local tokenizer and MRI row.

    The image-token artifact contains tokenizer image-placeholder IDs, not the
    processor's expanded visual patches; the latter need a runtime diagnostic.
    Existing output directories are never overwritten.
    """
    if output.exists():
        raise FileExistsError("Contract output already exists; do not overwrite a run")
    if not output.resolve().is_relative_to(Path(__file__).resolve().parents[2] / "local_data"):
        raise ValueError("Contract artifacts must stay under ignored local_data")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    # Import the pure-Python formatter file directly. Importing mlx_vlm's package
    # __init__ would initialize Metal merely to format a CPU prompt.
    package_spec = importlib.util.find_spec("mlx_vlm")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise RuntimeError("MLX-VLM package unavailable")
    formatter_path = Path(next(iter(package_spec.submodule_search_locations))) / "prompt_utils.py"
    spec = importlib.util.spec_from_file_location("mri_prompt_utils_cpu", formatter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("MLX-VLM prompt formatter unavailable")
    formatter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(formatter)
    records = _read_jsonl(train)
    row = next((r for r in records if r.get("data_source") == "OpenNeuro_ds007045" and
                r.get("extra_info", {}).get("internal_split") == "train"), None)
    if row is None or len(row.get("images", [])) != 1:
        raise ValueError("No eligible train image record")
    messages = [{"role": "system", "content": system_instruction}, *row["prompt"]]
    rendered = formatter.apply_chat_template(
        tokenizer, json.loads((model_path / "config.json").read_text()), messages,
        add_generation_prompt=True, num_images=1, enable_thinking=enable_thinking)
    if rendered.count("<|image_pad|>") != 1:
        raise ValueError("Rendered MRI prompt has wrong image-placeholder count")
    assistant_prefix = "<|im_start|>assistant\n"
    if not rendered.endswith(assistant_prefix):
        raise ValueError("Unexpected native assistant prefix")
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    image_ids = tokenizer.encode("<|image_pad|>", add_special_tokens=False)
    if not prompt_ids or not image_ids:
        raise ValueError("Empty prompt or image-placeholder token IDs")
    artifact_paths = {
        "model_config": model_path / "config.json",
        "processor_config": model_path / "preprocessor_config.json",
        "tokenizer_config": model_path / "tokenizer_config.json",
        "chat_template": model_path / "chat_template.json",
        "image_bytes": Path(row["images"][0]),
    }
    for path in artifact_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True)
    for name, value in (("rendered_prompt", rendered),
                        ("prompt_token_ids", json.dumps(prompt_ids)),
                        ("image_token_ids", json.dumps(image_ids))):
        path = output / (name + ".txt")
        path.write_text(value)
        artifact_paths[name] = path
    contract = ExperimentContract(
        version=VERSION, model_id="Qwen/Qwen3-VL-8B-Instruct",
        model_revision=QWEN_BASE_REVISION,
        processor_revision=QWEN_BASE_REVISION,
        tokenizer_revision=QWEN_BASE_REVISION,
        model_config_sha256=sha256_file(artifact_paths["model_config"]),
        processor_config_sha256=sha256_file(artifact_paths["processor_config"]),
        tokenizer_config_sha256=sha256_file(artifact_paths["tokenizer_config"]),
        chat_template_sha256=sha256_file(artifact_paths["chat_template"]),
        system_instruction=system_instruction,
        system_instruction_sha256=sha256_bytes(system_instruction.encode()),
        assistant_prefix=assistant_prefix, enable_thinking=enable_thinking,
        add_generation_prompt=True, response_includes_prefill=False,
        generation={"temperature": temperature, "max_tokens": max_tokens},
        strict_answer_grammar=STRICT_GRAMMAR, reward_parser_version=STRICT_PARSER,
        reward_profile="direct_vero", diagnostic_record_id=row["id"],
        train_manifest_sha256=sha256_file(train), dev_manifest_sha256=sha256_file(dev),
        rendered_prompt_sha256=sha256_file(artifact_paths["rendered_prompt"]),
        prompt_token_ids_sha256=sha256_file(artifact_paths["prompt_token_ids"]),
        image_token_ids_sha256=sha256_file(artifact_paths["image_token_ids"]),
        image_bytes_sha256=sha256_file(artifact_paths["image_bytes"]),
    )
    contract_path = output / "contract.json"
    contract_path.write_text(json.dumps(asdict(contract), indent=2))
    (output / "artifact_paths.json").write_text(json.dumps(
        {name: str(path.resolve()) for name, path in artifact_paths.items()}, indent=2))
    return contract_path, artifact_paths


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False).encode("utf-8"))


@dataclass(frozen=True)
class ExperimentContract:
    version: str
    model_id: str
    model_revision: str
    processor_revision: str
    tokenizer_revision: str
    model_config_sha256: str
    processor_config_sha256: str
    tokenizer_config_sha256: str
    chat_template_sha256: str
    system_instruction: str
    system_instruction_sha256: str
    assistant_prefix: str
    enable_thinking: bool
    add_generation_prompt: bool
    response_includes_prefill: bool
    generation: dict[str, Any]
    strict_answer_grammar: str
    reward_parser_version: str
    reward_profile: str
    diagnostic_record_id: str
    train_manifest_sha256: str
    dev_manifest_sha256: str
    rendered_prompt_sha256: str
    prompt_token_ids_sha256: str
    image_token_ids_sha256: str
    image_bytes_sha256: str

    def identity(self) -> str:
        return canonical_hash(asdict(self))


def load_contract(path: Path) -> ExperimentContract:
    raw = json.loads(path.read_text())
    contract = ExperimentContract(**raw)
    if contract.version != VERSION or contract.reward_parser_version != STRICT_PARSER:
        raise ValueError("Unsupported contract or strict reward parser version")
    if (contract.model_revision != QWEN_BASE_REVISION or
            contract.processor_revision != QWEN_BASE_REVISION or
            contract.tokenizer_revision != QWEN_BASE_REVISION):
        raise ValueError("Unexpected model, processor, or tokenizer revision")
    if contract.strict_answer_grammar != STRICT_GRAMMAR or contract.reward_profile != "direct_vero":
        raise ValueError("Preflight requires the strict direct Vero reward contract")
    if contract.system_instruction_sha256 != sha256_bytes(contract.system_instruction.encode()):
        raise ValueError("System instruction hash mismatch")
    for name in ("model_config_sha256", "processor_config_sha256",
                 "tokenizer_config_sha256", "chat_template_sha256",
                 "train_manifest_sha256", "dev_manifest_sha256",
                 "rendered_prompt_sha256", "prompt_token_ids_sha256",
                 "image_token_ids_sha256", "image_bytes_sha256"):
        value = getattr(contract, name)
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing or malformed {name}")
    if not isinstance(contract.enable_thinking, bool) or not isinstance(contract.response_includes_prefill, bool):
        raise ValueError("Thinking and prefill semantics must be explicit booleans")
    if not contract.assistant_prefix or not contract.system_instruction:
        raise ValueError("Missing exact prompt components")
    if not contract.diagnostic_record_id:
        raise ValueError("Missing diagnostic record ID")
    if (not isinstance(contract.generation.get("max_tokens"), int) or
            contract.generation["max_tokens"] <= 0 or
            "temperature" not in contract.generation):
        raise ValueError("Missing generation token limit or decoding temperature")
    return contract


def check_artifacts(contract: ExperimentContract, paths: dict[str, Path]) -> list[str]:
    failures = []
    for name in ("model_config", "processor_config", "tokenizer_config", "chat_template",
                 "rendered_prompt", "prompt_token_ids", "image_token_ids", "image_bytes"):
        path = paths.get(name)
        expected = getattr(contract, f"{name}_sha256")
        if path is None or not path.is_file() or sha256_file(path) != expected:
            failures.append(f"{name}: missing or hash mismatch")
    if failures:
        return failures
    try:
        model = json.loads(paths["model_config"].read_text())
        processor = json.loads(paths["processor_config"].read_text())
        tokenizer = json.loads(paths["tokenizer_config"].read_text())
        rendered = paths["rendered_prompt"].read_text()
        if model.get("model_type") != "qwen3_vl":
            failures.append("model config is not pinned Qwen3-VL family")
        if processor.get("image_processor_type") != "Qwen2VLImageProcessorFast":
            failures.append("image processor differs from pinned checkpoint")
        if tokenizer.get("tokenizer_class") != "Qwen2Tokenizer":
            failures.append("tokenizer differs from pinned checkpoint")
        if contract.system_instruction not in rendered:
            failures.append("rendered prompt omits contract system instruction")
        if not rendered.endswith(contract.assistant_prefix):
            failures.append("rendered prompt assistant prefix mismatch")
        if not paths["prompt_token_ids"].stat().st_size or not paths["image_token_ids"].stat().st_size:
            failures.append("empty token IDs")
    except (OSError, ValueError, TypeError, UnicodeError):
        failures.append("model/processor/tokenizer/prompt artifact is unreadable")
    return failures


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_data(contract: ExperimentContract, train: Path, dev: Path) -> dict[str, Any]:
    failures: list[str] = []
    if sha256_file(train) != contract.train_manifest_sha256:
        failures.append("train manifest hash mismatch")
    if sha256_file(dev) != contract.dev_manifest_sha256:
        failures.append("dev manifest hash mismatch")
    train_rows, dev_rows = _read_jsonl(train), _read_jsonl(dev)
    patients: dict[str, set[str]] = {"train": set(), "internal_dev": set()}
    ids: set[str] = set()
    checked_images = 0
    for split, rows in (("train", train_rows), ("internal_dev", dev_rows)):
        if not rows:
            failures.append(f"empty {split} manifest")
        for row in rows:
            extra = row.get("extra_info", {})
            patient = extra.get("patient_key")
            image_paths = row.get("images", [])
            if (row.get("data_source") != "OpenNeuro_ds007045" or
                    extra.get("patient_lineage") != "BIDS_SUBJECT_VERIFIED" or
                    extra.get("internal_split") != split or not patient):
                failures.append(f"{split}: invalid source/split/patient lineage")
                continue
            patients[split].add(patient)
            if not row.get("id") or row["id"] in ids:
                failures.append(f"{split}: missing or duplicate record ID")
            ids.add(row.get("id"))
            if len(image_paths) != 1 or not Path(image_paths[0]).is_file():
                failures.append(f"{split}: missing single real image")
            else:
                image = Path(image_paths[0])
                try:
                    from PIL import Image
                    with Image.open(image) as decoded:
                        decoded.verify()
                    checked_images += 1
                except (OSError, ValueError, ImportError):
                    failures.append(f"{split}: undecodable image")
            prompt = row.get("prompt")
            target = row.get("reward_model", {}).get("ground_truth")
            if (not isinstance(prompt, list) or not prompt or
                    any(message.get("role") not in {"system", "user"} for message in prompt) or
                    not isinstance(target, str) or not target):
                failures.append(f"{split}: malformed prompt or hidden target")
                continue
            if any("ground_truth" in json.dumps(message) or '"answer"' in json.dumps(message)
                   for message in prompt):
                failures.append(f"{split}: target field leaked into model-visible prompt")
            try:
                good = compute_score_from_data_source(
                    row["data_source"],
                    f"<think>Observed image.</think><answer>\\boxed{{{target}}}</answer>",
                    target, extra)
                bad = compute_score_from_data_source(
                    row["data_source"],
                    "<think>Observed image.</think><answer>\\boxed{definitely-wrong}</answer>",
                    target, extra)
                if not (good["accuracy"] == 1 and good["format"] == 1 and
                        bad["accuracy"] == 0 and good["score"] > bad["score"]):
                    failures.append(f"{split}: positive/negative strict reward failed")
            except (ValueError, KeyError, TypeError):
                failures.append(f"{split}: invalid reward route or target")
    if patients["train"] & patients["internal_dev"]:
        failures.append("train/internal-dev patient overlap")
    return {"passed": not failures, "failures": sorted(set(failures)),
            "train_rows": len(train_rows), "dev_rows": len(dev_rows),
            "train_patients": len(patients["train"]),
            "dev_patients": len(patients["internal_dev"]),
            "decoded_images": checked_images}


def check_resource(sample: ResourceSample) -> list[str]:
    failures = []
    if sample.power_source != "AC":
        failures.append("substantial training requires verified AC power")
    if sample.pressure_free_percent is None or sample.pressure_free_percent < 15:
        failures.append("memory pressure free percentage unavailable or below 15")
    if sample.available_gib is None or sample.available_gib < 2:
        failures.append("estimated available memory unavailable or below 2 GiB")
    if sample.thermal_state == "THROTTLED":
        failures.append("thermal throttling reported")
    return failures


def preflight(contract_path: Path, train: Path, dev: Path,
              artifacts: dict[str, Path], sample: ResourceSample | None = None, *,
              require_model_resources: bool = True) -> dict:
    contract = load_contract(contract_path)
    failures = check_artifacts(contract, artifacts)
    data = check_data(contract, train, dev)
    failures.extend(data["failures"])
    diagnostic_rows = [row for row in _read_jsonl(train)
                       if row.get("id") == contract.diagnostic_record_id]
    if len(diagnostic_rows) != 1:
        failures.append("diagnostic record absent from train split")
    else:
        row = diagnostic_rows[0]
        if (len(row.get("images", [])) != 1 or
                not Path(row["images"][0]).is_file() or
                sha256_file(Path(row["images"][0])) != contract.image_bytes_sha256):
            failures.append("diagnostic image bytes do not match training row")
        rendered_path = artifacts.get("rendered_prompt")
        if rendered_path is not None and rendered_path.is_file():
            rendered = rendered_path.read_text(errors="replace")
            text_parts = [item["text"] for message in row.get("prompt", [])
                          for item in message.get("content", [])
                          if isinstance(item, dict) and item.get("type") == "text"]
            if not text_parts or any(part not in rendered for part in text_parts):
                failures.append("rendered prompt does not contain diagnostic question")
            if "<|image_pad|>" not in rendered:
                failures.append("rendered prompt lacks MRI image placeholder")
    resource = sample if sample is not None else sample_macos()
    if require_model_resources:
        failures.extend(check_resource(resource))
    battery_status_conflict = False
    if sample is None and require_model_resources:
        try:
            battery_text = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                          text=True, timeout=2, check=False).stdout.lower()
            battery_status_conflict = "discharging" in battery_text
        except (OSError, subprocess.TimeoutExpired):
            failures.append("battery charging state unavailable")
        if battery_status_conflict:
            failures.append("battery reports discharging, even if power source says AC")
    return {"contract_version": contract.version, "contract_id": contract.identity(),
            "cpu_preflight_passed": not failures, "failures": sorted(set(failures)),
            "data": data, "resource": asdict(resource),
            "battery_discharge_detected": battery_status_conflict,
            "dynamic_gates_verified": False, "substantial_job_launch_ready": False,
            "dynamic_gates_required": ["real image model generation",
               "positive/negative strict reward from actual outputs",
               "nonzero task reward variance in an eligible rollout group",
               "finite optimizer gradient and changed weight hash",
            "checkpoint reload reproduces saved weights"]}


def review_scope(*, contract_path: Path, execution_tier: str, phase: str,
                 launch_budget: dict[str, int], expected_starting_adapter: Path | None,
                 dynamic_receipt_path: Path | None = None,
                 launch_configuration: dict[str, Any] | None = None) -> dict[str, Any]:
    """Binding the independent reviewer must inspect, author, and send by digest.

    The caller obtains the approval digest from the reviewer message, not from
    the candidate approval file. This is a trusted orchestration boundary, not
    cryptographic authentication against another process owned by the same user.
    """
    root = Path(__file__).resolve().parents[2]
    sources = ("experiment_contract.py", "vero_mri_reward.py", "mlx_gspo_smoke.py",
               "mlx_sft_warmstart.py", "mlx_multiarm_train.py", "resource_supervisor.py",
               "sft_evidence.py", "frozen_eval.py", "ds007045_internal_dev_eval.py",
               "paced_resources.py", "optimizer_checkpoint.py")
    contract = load_contract(contract_path)
    return {
        "execution_policy": EXECUTION_POLICY,
        "contract_id": contract.identity(), "execution_tier": execution_tier,
        "phase": phase, "launch_budget": launch_budget,
        "launch_configuration": launch_configuration,
        "generation": contract.generation,
        "starting_adapter_sha256": (sha256_file(expected_starting_adapter)
                                     if expected_starting_adapter is not None else None),
        "starting_adapter_config_sha256": (sha256_file(expected_starting_adapter.parent / "adapter_config.json")
            if expected_starting_adapter is not None and
            (expected_starting_adapter.parent / "adapter_config.json").is_file() else None),
        "dynamic_receipt_sha256": (sha256_file(dynamic_receipt_path)
                                    if dynamic_receipt_path is not None else None),
        "source_sha256": {name: sha256_file(root / "src" / "mri_rl" / name)
                          for name in sources},
    }


def _check_review(*, review_verdict_path: Path | None, expected_review_sha256: str | None,
                  scope: dict[str, Any]) -> None:
    if (review_verdict_path is None or not review_verdict_path.is_file() or
            not expected_review_sha256 or
            sha256_file(review_verdict_path) != expected_review_sha256):
        raise ValueError("NOT_VERIFIED: independently supplied review digest missing or mismatched")
    verdict = json.loads(review_verdict_path.read_text())
    if verdict.get("verdict") not in {"PASS", "PASS_WITH_LIMITATIONS"}:
        raise ValueError("Independent review did not authorize this configuration")
    if any(verdict.get(key) != value for key, value in scope.items()):
        raise ValueError("Independent review scope is stale or mismatches code, contract, adapter, or budget")
    if (verdict.get("reviewer_kind") not in {"isolated_codex", "claude_code"} or
            not verdict.get("reviewer_id") or not verdict.get("reviewer_findings")):
        raise ValueError("Independent reviewer identity/findings missing")
    if verdict["verdict"] == "PASS_WITH_LIMITATIONS" and not verdict.get("limitations"):
        raise ValueError("Limited approval requires explicit limitations")
    if scope["execution_tier"] == "C":
        if (verdict.get("substantial_training_authorized") is not True or
                not verdict.get("acceptance_criteria") or not verdict.get("matched_budget_checks")):
            raise ValueError("Tier C needs explicit substantial approval, acceptance criteria and matched budgets")


def require_launch_preflight(*, contract_path: Path | None, train: Path,
                             dev: Path | None, model_path: Path,
                             dynamic_receipt_path: Path | None,
                             review_verdict_path: Path | None,
                             expected_system: str,
                             expected_max_tokens: int | None = None,
                             expected_reward_profile: str = "direct_vero",
                             expected_starting_adapter: Path | None = None,
                             execution_tier: str = "C", phase: str = "continuation",
                             launch_budget: dict[str, int] | None = None,
                             expected_review_sha256: str | None = None,
                             expected_temperature: float | None = None,
                             launch_configuration: dict[str, Any] | None = None) -> str:
    """Verify exact tier before model loading; Tier A authorizes CPU work only.

    Tier B diagnostic resolves the bootstrap dependency: static safety and an
    actual independent approval permit at most one attempted update. The runner
    must enforce task variance/finite gradients and verify save/reload itself.
    Continuation requires reviewed dynamic receipts from that diagnostic.
    """
    if contract_path is None or dev is None:
        raise ValueError("Private contract and dev split are required")
    if execution_tier not in {"A", "B", "C"}:
        raise ValueError("Unknown execution tier")
    if (execution_tier == "A" and phase != "cpu") or (execution_tier != "A" and phase not in {"diagnostic", "continuation"}):
        raise ValueError("Invalid execution tier/phase")
    if execution_tier == "C" and phase != "continuation":
        raise ValueError("Tier C requires independently verified prior dynamic evidence")
    private_root = Path(__file__).resolve().parents[2] / "local_data"
    for path in (contract_path, dynamic_receipt_path, review_verdict_path):
        if path is not None and not path.resolve().is_relative_to(private_root):
            raise ValueError("Contract and evidence must stay under ignored local_data")
    contract = load_contract(contract_path)
    if contract.system_instruction != expected_system:
        raise ValueError("Training system instruction differs from versioned contract")
    if contract.reward_profile != expected_reward_profile:
        raise ValueError("Training reward profile differs from versioned contract")
    if expected_max_tokens is not None and contract.generation["max_tokens"] != expected_max_tokens:
        raise ValueError("Training max_tokens differs from versioned contract")
    if expected_temperature is not None and contract.generation["temperature"] != expected_temperature:
        raise ValueError("Training temperature differs from versioned contract")
    if execution_tier != "A":
        if not isinstance(launch_configuration, dict) or not launch_configuration.get("runner"):
            raise ValueError("Exact launcher configuration is required")
        required_budget = {"optimizer_steps", "max_groups", "rollouts_per_group", "max_tokens"}
        if (not isinstance(launch_budget, dict) or set(launch_budget) != required_budget or
                any(type(value) is not int or value < 1 for value in launch_budget.values())):
            raise ValueError("Explicit positive integer launch budget is required")
        if launch_budget["max_tokens"] != contract.generation["max_tokens"]:
            raise ValueError("Launch token budget differs from contract")
        if execution_tier == "B":
            step_limit = 1 if phase == "diagnostic" else 3
            if (launch_budget["optimizer_steps"] > step_limit or launch_budget["max_groups"] > 3 or
                    launch_budget["max_groups"] * launch_budget["rollouts_per_group"] > 8 or
                    launch_budget["rollouts_per_group"] < 2 or launch_budget["max_tokens"] > 512):
                raise ValueError("Tier B exceeds bounded diagnostic/continuation budget")
        if phase == "diagnostic" and dynamic_receipt_path is not None:
            raise ValueError("First diagnostic must not depend on a prior dynamic receipt")
        if expected_starting_adapter is None or not expected_starting_adapter.is_file():
            raise ValueError("Exact starting adapter file is required for model launch")
    paths = resolve_artifact_paths(contract_path)
    for name, filename in (("model_config", "config.json"),
                           ("processor_config", "preprocessor_config.json"),
                           ("tokenizer_config", "tokenizer_config.json"),
                           ("chat_template", "chat_template.json")):
        if paths[name].resolve() != (model_path / filename).resolve():
            raise ValueError("Contract model/processor path differs from launch model")
    report = preflight(contract_path, train, dev, paths,
                       require_model_resources=execution_tier != "A")
    if not report["cpu_preflight_passed"]:
        raise ValueError("CPU preflight failed: " + "; ".join(report["failures"]))
    if execution_tier == "A":
        return contract.identity()
    scope = review_scope(contract_path=contract_path, execution_tier=execution_tier,
                         phase=phase, launch_budget=launch_budget,
                         expected_starting_adapter=expected_starting_adapter,
                         dynamic_receipt_path=dynamic_receipt_path,
                         launch_configuration=launch_configuration)
    _check_review(review_verdict_path=review_verdict_path,
                  expected_review_sha256=expected_review_sha256, scope=scope)
    if execution_tier == "B" and phase == "diagnostic":
        return contract.identity()
    if dynamic_receipt_path is None:
        raise ValueError("Continuation requires verified real-model dynamic receipt")
    dynamic = json.loads(dynamic_receipt_path.read_text())
    if dynamic.get("kind") == "supervised_control_v1":
        if execution_tier != "C" or launch_configuration.get("runner") != "mlx_sft_warmstart":
            raise ValueError("Supervised evidence cannot authorize RL training")
        from mri_rl.sft_evidence import verify_supervised_evidence
        verify_supervised_evidence(dynamic, contract=contract, train=train,
                                   starting_adapter=expected_starting_adapter,
                                   configuration=launch_configuration, budget=launch_budget,
                                   private_root=private_root)
        return contract.identity()
    if dynamic.get("contract_id") != contract.identity():
        raise ValueError("Dynamic diagnostic contract mismatch")
    required_true = ("actual_model_generation", "real_mri_image_verified",
                     "strict_positive_reward_observed", "strict_negative_reward_observed",
                     "finite_gradients", "weight_changed", "checkpoint_reloaded")
    if any(dynamic.get(key) is not True for key in required_true):
        raise ValueError("Dynamic diagnostic lacks verified real-model gates")
    task_rewards = dynamic.get("task_reward_values")
    if (not isinstance(task_rewards, list) or len(task_rewards) < 2 or
            any(not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in task_rewards) or max(task_rewards) == min(task_rewards)):
        raise ValueError("No finite nonzero task reward variance in diagnostic group")
    grad = dynamic.get("gradient_norm")
    if not isinstance(grad, (int, float)) or not math.isfinite(grad) or grad <= 0:
        raise ValueError("No finite positive diagnostic gradient norm")
    before, after, reloaded = (dynamic.get(key) for key in (
        "before_adapter_sha256", "after_adapter_sha256", "reloaded_adapter_sha256"))
    if (not all(isinstance(value, str) and len(value) == 64 for value in
                (before, after, reloaded)) or before == after or after != reloaded):
        raise ValueError("Optimizer change or checkpoint reload hash not verified")
    source_path = Path(dynamic.get("source_receipt_path", ""))
    if (not source_path.resolve().is_relative_to(private_root) or
            not source_path.is_file() or
            sha256_file(source_path) != dynamic.get("source_receipt_sha256")):
        raise ValueError("Underlying real-model diagnostic receipt unavailable or changed")
    source = json.loads(source_path.read_text())
    if (source.get("status") != "TRAINING_COMPLETED_ENGINEERING_SMOKE" or
            source.get("reward_profile") != "direct_vero" or
            source.get("image_source") != "OpenNeuro_ds007045" or
            source.get("policy_updated") is not True or
            source.get("optimizer_steps") != 1 or
            source.get("groups_attempted", 0) < 1 or
            source.get("science_claim") is not False or
            source.get("contract_id") != contract.identity() or
            source.get("temperature") != contract.generation["temperature"] or
            source.get("max_tokens") != contract.generation["max_tokens"] or
            source.get("system_prompt_sha256") != contract.system_instruction_sha256 or
            source.get("model_config_sha256") != contract.model_config_sha256 or
            source.get("training_records_sha256") != contract.train_manifest_sha256):
        raise ValueError("Source receipt is not a real ds007045 direct-Vero MLX diagnostic")
    if (source.get("before_adapter_sha256") != before or
            source.get("after_adapter_sha256") != after or
            source.get("gradient_l2") != grad):
        raise ValueError("Dynamic optimizer claims disagree with source receipt")
    source_dir = source_path.parent
    before_path, after_path = (source_dir / name for name in
                               ("before_adapters.safetensors", "after_adapters.safetensors"))
    if (not before_path.is_file() or not after_path.is_file() or
            sha256_file(before_path) != before or sha256_file(after_path) != after):
        raise ValueError("Diagnostic adapter bytes do not match source receipt")
    raw_path = source_dir / "raw_rollouts.json"
    if (not raw_path.is_file() or
            sha256_file(raw_path) != dynamic.get("raw_rollouts_sha256")):
        raise ValueError("Actual raw rollout artifact missing or changed")
    raw = json.loads(raw_path.read_text())
    attempts = source.get("attempts", [])
    if not isinstance(raw, list) or len(raw) != len(attempts):
        raise ValueError("Raw rollout/source group count mismatch")
    train_by_id = {row["id"]: row for row in _read_jsonl(train)}
    verified_variance = False
    selected_image_sha256 = None
    for raw_group, attempt in zip(raw, attempts):
        row = train_by_id.get(raw_group.get("record_id"))
        rollouts = raw_group.get("rollouts", [])
        saved_rewards = attempt.get("rewards", [])
        if (row is None or attempt.get("source") != "OpenNeuro_ds007045" or
                attempt.get("route") != row.get("extra_info", {}).get("reward_type") or
                raw_group.get("group_index") != attempt.get("group_index") or
                len(rollouts) < 2 or len(rollouts) != len(saved_rewards)):
            raise ValueError("Rollout linkage or group shape invalid")
        image_sha = sha256_file(Path(row["images"][0]))
        rendered = raw_group.get("rendered_prompt", "")
        prompt_ids = raw_group.get("prompt_token_ids")
        if (raw_group.get("image_sha256") != image_sha or attempt.get("image_sha256") != image_sha
                or contract.system_instruction not in rendered
                or not rendered.endswith(contract.assistant_prefix)
                or not isinstance(prompt_ids, list) or not prompt_ids
                or attempt.get("rendered_prompt_sha256") != sha256_bytes(rendered.encode())
                or attempt.get("prompt_token_ids_sha256") != sha256_bytes(json.dumps(prompt_ids).encode())):
            raise ValueError("Raw group image/prompt binding invalid")
        scores, accuracies, formats = [], [], []
        for rollout, saved in zip(rollouts, saved_rewards):
            tokens = rollout.get("token_ids")
            if not isinstance(tokens, list) or not tokens or not isinstance(rollout.get("text"), str):
                raise ValueError("Raw generated text/tokens absent")
            grade = compute_score_from_data_source(
                row["data_source"], rollout["text"],
                row["reward_model"]["ground_truth"], row["extra_info"])
            if any(not math.isclose(grade[key], saved[key], rel_tol=0, abs_tol=1e-9)
                   for key in ("score", "accuracy", "format")):
                raise ValueError("Saved reward disagrees with strict recomputation")
            scores.append(grade["score"])
            accuracies.append(grade["accuracy"])
            formats.append(grade["format"])
        # Accuracy variation, with both candidates satisfying the strict format,
        # prevents a format-only or length-only difference authorizing expansion.
        strict_accuracies = [accuracy for accuracy, fmt in zip(accuracies, formats) if fmt == 1]
        if (all(fmt == 1 for fmt in formats) and strict_accuracies and min(strict_accuracies) == 0 and max(strict_accuracies) == 1
                and max(scores) > min(scores) and attempt.get("active") is True
                and task_rewards == accuracies
                and source.get("selected_active_group") == raw_group.get("group_index")):
            verified_variance = True
            selected_image_sha256 = image_sha
    if not verified_variance:
        raise ValueError("No independently recomputed active strict task-reward group")
    reload_path = Path(dynamic.get("reload_inference_receipt_path", ""))
    if (not reload_path.resolve().is_relative_to(private_root) or
            not reload_path.is_file() or
            sha256_file(reload_path) != dynamic.get("reload_inference_receipt_sha256")):
        raise ValueError("Independent checkpoint reload inference receipt unavailable")
    reload_receipt = json.loads(reload_path.read_text())
    if (reload_receipt.get("loaded_adapter_sha256") != after or
            reload_receipt.get("image_sha256") != selected_image_sha256 or
            reload_receipt.get("actual_model_generation") is not True or
            not isinstance(reload_receipt.get("generated_text"), str) or
            not reload_receipt["generated_text"]):
        raise ValueError("Checkpoint reload inference is not bound to adapter/image")
    if expected_starting_adapter is not None:
        if (not expected_starting_adapter.is_file() or
                dynamic.get("starting_adapter_sha256") != sha256_file(expected_starting_adapter) or
                source.get("starting_adapter_sha256") != sha256_file(expected_starting_adapter)):
            raise ValueError("Diagnostic starting adapter differs from training launch")
    return contract.identity()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    for name in ("model_config", "processor_config", "tokenizer_config", "chat_template",
                 "rendered_prompt", "prompt_token_ids", "image_token_ids", "image_bytes"):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("model_config", "processor_config",
             "tokenizer_config", "chat_template", "rendered_prompt", "prompt_token_ids",
             "image_token_ids", "image_bytes")}
    report = preflight(args.contract, args.train, args.dev, paths)
    print(json.dumps(report, sort_keys=True, indent=2))
    raise SystemExit(0 if report["cpu_preflight_passed"] else 2)


if __name__ == "__main__":
    main()
