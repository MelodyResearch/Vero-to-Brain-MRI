import json
import pytest

from mri_rl.ds007045_train_adapter import convert_ds007045
from mri_rl.mlx_sft_warmstart import _schedule_train_rows, _selected_train_rows
from mri_rl.mlx_gspo_smoke import _records_for_route
from mri_rl.mlx_multiarm_train import build_schedule
from mri_rl.vero_mri_reward import compute_score_from_data_source
from mri_rl.vero_official_launch import build_launch


def test_patient_keyed_conversion_never_writes_heldout_gold(tmp_path):
    cases = []
    prompts = []
    gold = []
    image = tmp_path / "fixed_image.png"
    image.write_bytes(b"image")
    for patient in range(40):
        split = "train" if patient < 32 else "test"
        token = f"token{patient:03}"
        cases.append({"private_subject": f"sub-{patient:03}",
                      "anonymous_token": token, "private_site": "SITE", "split": split})
        categories = (["sequence", "tumor_presence", "enhancing_presence", "tumor_grid"] * 3)
        if split == "train" and patient < 26:
            categories += ["tumor_grid", "tumor_grid"]
        if split == "test":
            categories += ["sequence"]
            if patient < 35:
                categories += ["sequence"]
        for ordinal, category in enumerate(categories):
            prompt_id = f"{token}_{ordinal:02}_{category}"
            prompts.append({"prompt_id": prompt_id, "split": split,
                            "category": category, "image_path": str(image),
                            "question": "Read the MRI."})
            answer = ("FLAIR" if category == "sequence" else
                      "center-middle" if category == "tumor_grid" else "yes")
            gold.append({"prompt_id": prompt_id, "split": split,
                         "category": category, "private_subject": f"sub-{patient:03}",
                         "answer": answer if split == "train" else "secret_test_gold"})
    assert len(prompts) == 543 and sum(r["split"] == "train" for r in prompts) == 436
    prompt_file, gold_file, manifest_file = (tmp_path / name for name in
                                              ("prompts.jsonl", "gold.jsonl", "manifest.json"))
    prompt_file.write_text("".join(json.dumps(row) + "\n" for row in prompts))
    gold_file.write_text("".join(json.dumps(row) + "\n" for row in gold))
    manifest_file.write_text(json.dumps({
        "source": "OpenNeuro ds007045 v2.0.1", "train_patient_count": 32,
        "heldout_patient_count": 8, "private_cases": cases,
    }))
    out = tmp_path / "out"
    stats = convert_ds007045(prompt_file, gold_file, manifest_file, out)
    train = [json.loads(line) for line in (out / "train.verl.jsonl").read_text().splitlines()]
    dev = [json.loads(line) for line in (out / "internal_dev.verl.jsonl").read_text().splitlines()]
    assert stats["internal_train_patients"] == 28
    assert stats["internal_dev_patients"] == 4
    assert len(train) + len(dev) == 436
    assert {r["extra_info"]["patient_key"] for r in train}.isdisjoint(
        {r["extra_info"]["patient_key"] for r in dev})
    assert "secret_test_gold" not in "".join(
        path.read_text() for path in out.rglob("*.jsonl"))
    assert all(r["data_source"] == "OpenNeuro_ds007045" for r in train + dev)
    sft = _selected_train_rows(out / "train.verl.jsonl")
    assert [r["ability"] for r in sft] == [
        "mri_sequence", "mri_tumor_presence", "mri_enhancing_presence", "mri_tumor_grid"]
    schedule = _schedule_train_rows(out / "train.verl.jsonl", steps=4, seed=7, schedule_policy="legacy_lexical_v1")
    assert schedule == _schedule_train_rows(out / "train.verl.jsonl", steps=4, seed=7, schedule_policy="legacy_lexical_v1")
    for category in ("mri_sequence", "mri_tumor_presence", "mri_enhancing_presence",
                     "mri_tumor_grid"):
        assert len({next(r for r in step if r["ability"] == category)["extra_info"]["patient_key"]
                    for step in schedule}) == 4
    full_schedule = _schedule_train_rows(out / "train.verl.jsonl", steps=28, seed=7, schedule_policy="legacy_lexical_v1")
    assert len(full_schedule) == 28
    for category in ("mri_sequence", "mri_tumor_presence", "mri_enhancing_presence",
                     "mri_tumor_grid"):
        assert len({next(r for r in step if r["ability"] == category)["extra_info"]["patient_key"]
                    for step in full_schedule}) == 28
    with pytest.raises(ValueError, match="1–28"):
        _schedule_train_rows(out / "train.verl.jsonl", steps=29, seed=7)
    sampled = _records_for_route(out / "train.verl.jsonl", "mri_sequence", 3)
    assert len(sampled) == 3
    assert all(r["extra_info"]["patient_lineage"] == "BIDS_SUBJECT_VERIFIED"
               for r in sampled)
    source = tmp_path / "official"
    (source / "vero-rl/verl/trainer").mkdir(parents=True)
    (source / "vero-rl/verl/trainer/main_ppo.py").write_text("")
    model = tmp_path / "model"
    model.mkdir()
    _, _, uniform = build_launch(source, out, model, tmp_path / "launch",
                                 profile="ds007045", arm="uniform")
    _, _, proportional = build_launch(source, out, model, tmp_path / "launch",
                                      profile="ds007045", arm="proportional")
    _, _, specialist = build_launch(source, out, model, tmp_path / "launch",
                                    profile="ds007045", arm="specialist",
                                    specialist_category="mri_tumor_grid")
    assert set(uniform["expected_category_draws_per_batch"].values()) == {64}
    assert proportional["expected_category_draws_per_batch"] != uniform[
        "expected_category_draws_per_batch"]
    assert specialist["expected_category_draws_per_batch"] == {"mri_tumor_grid": 256}
    assert uniform["heldout_test_used_for_training_or_validation"] is False
    uniform_schedule, uniform_plan = build_schedule(out / "train.verl.jsonl", "uniform", 8, 7)
    proportional_schedule, proportional_plan = build_schedule(
        out / "train.verl.jsonl", "proportional", 8, 7)
    assert len(uniform_schedule) == len(proportional_schedule) == 8
    assert set(uniform_plan["category_groups"].values()) == {2}
    assert sum(proportional_plan["category_groups"].values()) == 8
    assert all(r["extra_info"]["internal_split"] == "train" for r in uniform_schedule)


def test_patient_keyed_reward_routes_use_exact_mask_or_sidecar_labels():
    for route, truth in (("mri_sequence", "FLAIR"),
                         ("mri_tumor_presence", "yes"),
                         ("mri_enhancing_presence", "no"),
                         ("mri_tumor_grid", "left-middle")):
        result = compute_score_from_data_source(
            "OpenNeuro_ds007045",
            f"<think>Read visible MRI evidence.</think><answer>\\boxed{{{truth}}}</answer>",
            truth, {"reward_type": route})
        assert result["accuracy"] == result["format"] == result["score"] == 1.0


def _sft_schedule_fixture(tmp_path, patients=28):
    image = tmp_path / "image.png"
    image.write_bytes(b"CPU schedule fixture only")
    rows = []
    from mri_rl.mlx_sft_warmstart import CATEGORIES, SEQUENCE_LABELS
    for patient in range(patients):
        for category in CATEGORIES:
            labels = SEQUENCE_LABELS if category == "mri_sequence" else ("yes",)
            for index, label in enumerate(labels):
                rows.append({"id": f"p{patient:02}-{category}-{index}",
                    "data_source": "OpenNeuro_ds007045", "ability": category,
                    "images": [str(image)], "reward_model": {"ground_truth": label},
                    "extra_info": {"patient_key": f"p{patient:02}",
                                   "patient_lineage": "BIDS_SUBJECT_VERIFIED",
                                   "internal_split": "train"}})
    path = tmp_path / "train.verl.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, rows


def test_future_sft_balances_labels_without_changing_patients_or_budget(tmp_path):
    from mri_rl.mlx_sft_warmstart import CATEGORIES, schedule_label_exposure
    path, rows = _sft_schedule_fixture(tmp_path)
    legacy = _schedule_train_rows(path, 28, 7, "legacy_lexical_v1")
    future = _schedule_train_rows(path, 28, 7, "balanced_sequence_v2")
    assert schedule_label_exposure(legacy)["mri_sequence"] == {"FLAIR": 28}
    assert schedule_label_exposure(future)["mri_sequence"] == {"FLAIR": 14, "contrast-enhanced T1": 14}
    assert sum(map(len, future)) == 112
    for category in CATEGORIES:
        old = [row for step in legacy for row in step if row["ability"] == category]
        new = [row for step in future for row in step if row["ability"] == category]
        assert [r["extra_info"]["patient_key"] for r in old] == [r["extra_info"]["patient_key"] for r in new]
        assert len({r["extra_info"]["patient_key"] for r in new}) == 28
        if category != "mri_sequence":
            assert old == new
    assert future == _schedule_train_rows(path, 28, 7)
    # Input file ordering cannot change the deterministic schedule.
    path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    assert future == _schedule_train_rows(path, 28, 7)


@pytest.mark.parametrize("steps", [2, 3, 7, 28])
def test_future_sequence_counts_differ_by_at_most_one(tmp_path, steps):
    from mri_rl.mlx_sft_warmstart import schedule_label_exposure
    path, _ = _sft_schedule_fixture(tmp_path)
    counts = schedule_label_exposure(_schedule_train_rows(path, steps, 13))["mri_sequence"]
    assert set(counts) == {"FLAIR", "contrast-enhanced T1"}
    assert abs(counts["FLAIR"] - counts["contrast-enhanced T1"]) <= 1


def test_future_sft_rejects_missing_assigned_label_and_nontrain_rows(tmp_path):
    path, rows = _sft_schedule_fixture(tmp_path)
    flair_only = [r for r in rows if r["reward_model"]["ground_truth"] != "contrast-enhanced T1"]
    path.write_text("".join(json.dumps(row) + "\n" for row in flair_only))
    with pytest.raises(ValueError, match="lacks the assigned label"):
        _schedule_train_rows(path, 28, 7)
    # Legacy remains available for exact historical reconstruction.
    assert len(_schedule_train_rows(path, 28, 7, "legacy_lexical_v1")) == 28
    rows[0]["extra_info"]["internal_split"] = "internal_dev"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Unexpected source or patient lineage"):
        _schedule_train_rows(path, 28, 7)
