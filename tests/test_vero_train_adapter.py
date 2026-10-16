import json
from collections import Counter
from pathlib import Path

import pytest

from mri_rl.vero_mri_reward import compute_score_from_data_source
from mri_rl.vero_official_launch import build_launch, expected_batch_counts
from mri_rl.mlx_gspo_smoke import advantages_from_rewards
from mri_rl.vero_train_adapter import convert_jsonl


def _record(path: Path, index: int, task: str, answer):
    image = path / f"scan_{index:03d}_central.png"
    image.write_bytes(b"image fixture")
    return {
        "input_image": str(image), "task_id": task, "answer": answer,
        "prompt": "Read the MRI image.",
        "split_status": "UNVERIFIED_PATIENT_IDENTITY_NO_HELD_OUT_CLAIM",
    }


def test_official_schema_and_scan_disjoint_engineering_split(tmp_path):
    tasks = [
        _record(tmp_path, 0, "central_slice_hippocampal_centroid", [10, 12]),
        _record(tmp_path, 0, "central_slice_anterior_posterior_dominance", "anterior"),
        _record(tmp_path, 1, "central_slice_hippocampal_centroid", [15, 17]),
        _record(tmp_path, 1, "central_slice_anterior_posterior_dominance", "posterior"),
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(x) + "\n" for x in tasks))
    counts = convert_jsonl(source, tmp_path / "out")
    train = [json.loads(x) for x in (tmp_path / "out/train.verl.jsonl").read_text().splitlines()]
    dev = [json.loads(x) for x in (tmp_path / "out/dev_scan_disjoint_only.verl.jsonl").read_text().splitlines()]
    assert counts["train_rows"] == counts["dev_rows"] == 2
    assert {x["extra_info"]["scan_key"] for x in train} == {"001"}
    assert {x["extra_info"]["scan_key"] for x in dev} == {"000"}
    assert {"prompt", "images", "reward_model", "ability", "extra_info", "data_source"} <= train[0].keys()
    assert all(not x["prompt"][0]["content"][0].get("mask") for x in train)
    assert counts["patient_disjoint"] is False


def test_unsupported_and_duplicate_rows_fail_closed(tmp_path):
    row = _record(tmp_path, 0, "central_slice_hippocampal_centroid", [10, 12])
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        convert_jsonl(source, tmp_path / "out")


def test_mri_reward_routes_and_format():
    args = {"data_source": "MSD_Task04_Hippocampus", "ground_truth": "10,12",
            "extra_info": {"reward_type": "mri_centroid_2d"}}
    good = compute_score_from_data_source(solution_str="<think>Locate it.</think><answer>\\boxed{12,14}</answer>", **args)
    assert good["score"] == 1.0
    assert good["voxel_distance"] < 3
    bad = compute_score_from_data_source(solution_str="<think>Locate it.</think><answer>\\boxed{30,30}</answer>", **args)
    assert bad["accuracy"] == 0.0 and bad["score"] == 0.2
    dominance = compute_score_from_data_source(
        data_source="MSD_Task04_Hippocampus", ground_truth="posterior",
        extra_info={"reward_type": "mri_dominance"},
        solution_str="<think>Compare regions.</think><answer>\\boxed{posterior}</answer>",
    )
    assert dominance["score"] == 1.0
    with pytest.raises(ValueError, match="Unsupported MRI"):
        compute_score_from_data_source(
            data_source="MSD_Task04_Hippocampus", ground_truth="posterior",
            extra_info={"reward_type": "llm_judge"}, solution_str="posterior",
        )


def test_official_launcher_routes_verified_schema(tmp_path):
    source = tmp_path / "official"
    (source / "vero-rl/verl/trainer").mkdir(parents=True)
    (source / "vero-rl/verl/trainer/main_ppo.py").write_text("")
    model = tmp_path / "model"
    model.mkdir()
    records = tmp_path / "records"
    records.mkdir()
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    rows = [
        {"ability": category, "images": [str(image)],
         "extra_info": {"patient_lineage": "UNVERIFIED"}}
        for category in ("anatomical_comparison", "spatial_localization")
    ]
    payload = "".join(json.dumps(x) + "\n" for x in rows)
    (records / "train.verl.jsonl").write_text(payload)
    (records / "dev_scan_disjoint_only.verl.jsonl").write_text(payload)
    args, env, details = build_launch(source, records, model, tmp_path / "out")
    assert "reward_model.reward_manager=vero_vllm_judge" in args
    assert "--config-name=gspo_qwen3vl" in args
    assert any("custom_reward_function.path=" in x for x in args)
    assert details["proportional_equals_uniform"] is True
    assert env["REPO_ROOT"] == str((source / "vero-rl").resolve())


def test_mlx_smoke_group_advantages_match_pinned_sample_std():
    values = advantages_from_rewards([0.0, 1.0], epsilon=0.0)
    assert values == pytest.approx([-1 / 2**0.5, 1 / 2**0.5])
    assert advantages_from_rewards([0.2, 0.2]) == [0.0, 0.0]


def test_combined_task01_routes_and_mixture_weights(tmp_path):
    source = tmp_path / "source.jsonl"
    mri_rows = []
    for index in range(5):
        mri_rows.extend([
            _record(tmp_path, index, "central_slice_hippocampal_centroid", [10, 12]),
            _record(tmp_path, index, "central_slice_anterior_posterior_dominance", "posterior"),
        ])
    source.write_text("".join(json.dumps(x) + "\n" for x in mri_rows))
    task01 = tmp_path / "task01.jsonl"
    tumor_rows = []
    for index in range(4):
        image = tmp_path / f"case_{index}_z077.png"
        image.write_bytes(b"real-image fixture")
        for category, gold in (("tumor_enhancing_presence", "yes"),
                               ("tumor_grid", "center-middle")):
            tumor_rows.append({"record_id": f"{image.stem}_{category}",
                               "category": category, "image": str(image),
                               "prompt": "Read the MRI.", "gold": gold})
    task01.write_text("".join(json.dumps(x) + "\n" for x in tumor_rows))
    stats = convert_jsonl(source, tmp_path / "combined", task01_source=task01)
    assert stats["train_rows"] == 14 and stats["dev_rows"] == 4
    train = [json.loads(x) for x in (tmp_path / "combined/train.verl.jsonl").read_text().splitlines()]
    assert Counter(x["ability"] for x in train) == {
        "anatomical_comparison": 4, "spatial_localization": 4,
        "tumor_enhancing_presence": 3, "tumor_spatial_grid": 3,
    }
    assert compute_score_from_data_source(
        "MSD_Task01_BrainTumour", "<think>visible.</think><answer>\\boxed{yes}</answer>",
        "yes", {"reward_type": "mri_enhancing_presence"},
    )["score"] == 1.0
    assert compute_score_from_data_source(
        "MSD_Task01_BrainTumour", "<think>locate.</think><answer>\\boxed{center-middle}</answer>",
        "center-middle", {"reward_type": "mri_tumor_grid"},
    )["score"] == 1.0
    with pytest.raises(ValueError, match="mismatch"):
        compute_score_from_data_source(
            "MSD_Task04_Hippocampus", "<think>visible.</think><answer>\\boxed{yes}</answer>",
            "yes", {"reward_type": "mri_enhancing_presence"},
        )
    uniform = expected_batch_counts({category: 0.25 for category in (
        "anatomical_comparison", "spatial_localization",
        "tumor_enhancing_presence", "tumor_spatial_grid")}, 256)
    assert sorted(uniform.values()) == [64, 64, 64, 64]
