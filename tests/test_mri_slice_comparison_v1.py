import copy
import hashlib
import json
import numpy as np
import pytest
from PIL import Image

from mri_rl import mri_slice_comparison_v1 as sc
from mri_rl import paired_z045_evaluation_v1 as ev

TEXT = "Which MRI sequence is this? Give the final answer in \\boxed{...}."


def png(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (436, 364, 3), dtype=np.uint8)).save(path)
    return str(path)


def old_row(rid, path, truth, patient):
    return {"id": rid, "data_source": "OpenNeuro_ds007045", "ability": "mri_sequence",
            "prompt": [{"role": "user", "content": [{"type": "image", "image": path}, {"type": "text", "text": TEXT}]}],
            "images": [path], "reward_model": {"style": "rule", "ground_truth": truth},
            "extra_info": {"reward_type": "mri_sequence", "patient_key": patient, "level": 45}}


def binding(tmp_path, people=3):
    rows = []
    for i in range(people):
        for m, truth in (("flair", "FLAIR"), ("t1ce", "contrast-enhanced T1")):
            oid = f"r{i}{m}"
            a = png(tmp_path / f"{oid}_old.png", 10 * i + (m == "t1ce"))
            a_copy = tmp_path / f"{oid}_legacy.png"
            a_copy.write_bytes(open(a, "rb").read())
            b = png(tmp_path / f"{oid}_med.png", 100 + 10 * i + (m == "t1ce"))
            rows.append({"opaque_id": oid, "pair_key": f"p{i}", "modality": m, "truth": truth, "old_row_PRIVATE": old_row("x" + oid, a, truth, f"tok{i}"),
                         "A_image": str(a_copy), "B_image": b, "B_abstain": False})
    return {"schema": sc.BINDING_SCHEMA, "rows": rows}


def test_records_bind_both_image_fields_and_keep_text_bit_exact(tmp_path):
    prep = sc.build_records(binding(tmp_path))
    assert len(prep["records"]["inA"]) == len(prep["records"]["inB"]) == 6
    for arm in ("inA", "inB"):
        for rec in prep["records"][arm]:
            row = rec["row_PRIVATE"]
            typed = [c["image"] for c in row["prompt"][0]["content"] if c["type"] == "image"]
            texts = [c["text"] for c in row["prompt"][0]["content"] if c["type"] == "text"]
            assert typed == row["images"] and len(typed) == 1 and texts == [TEXT]
            assert rec["image_sha256"] == hashlib.sha256(open(typed[0], "rb").read()).hexdigest()
    a, b = prep["records"]["inA"][0]["row_PRIVATE"], prep["records"]["inB"][0]["row_PRIVATE"]
    assert a["images"] != b["images"] and a["reward_model"] == b["reward_model"] and a["prompt"][0]["content"][1] == b["prompt"][0]["content"][1]


def test_only_the_image_differs_between_inputs(tmp_path):
    prep = sc.build_records(binding(tmp_path))
    for ra, rb in zip(prep["records"]["inA"], prep["records"]["inB"]):
        x, y = copy.deepcopy(ra["row_PRIVATE"]), copy.deepcopy(rb["row_PRIVATE"])
        for r in (x, y):
            r["images"] = ["IMG"]
            r["prompt"][0]["content"][0]["image"] = "IMG"
            r.pop("id")
        assert x == y and ra["id"] != rb["id"]


def test_old_png_must_equal_input_a_pixels(tmp_path):
    b = binding(tmp_path)
    p = b["rows"][0]["A_image"]
    im = np.asarray(Image.open(p).convert("RGB")).copy()
    im[0, 0] = (im[0, 0].astype(int) + 1) % 256
    Image.fromarray(im).save(p)
    with pytest.raises(ValueError, match="old PNG"):
        sc.build_records(b)


def test_typed_and_toplevel_image_mismatch_in_source_row_refused(tmp_path):
    b = binding(tmp_path)
    b["rows"][0]["old_row_PRIVATE"]["images"] = [str(tmp_path / "other.png")]
    with pytest.raises(ValueError, match="image"):
        sc.build_records(b)


def test_expected_sha_and_leakage_and_pairing_refused(tmp_path):
    b = binding(tmp_path)
    b["rows"][0]["B_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        sc.build_records(b)
    b = binding(tmp_path)
    b["rows"][0]["old_row_PRIVATE"]["prompt"][0]["content"][1]["text"] += " r0flair"
    with pytest.raises(ValueError, match="identifier"):
        sc.build_records(b)
    b = binding(tmp_path)
    b["rows"][1]["modality"] = "flair"
    with pytest.raises(ValueError, match="pair"):
        sc.build_records(b)


def test_b_equal_to_a_is_flagged_not_hidden(tmp_path):
    b = binding(tmp_path)
    b["rows"][0]["B_image"] = b["rows"][0]["A_image"]
    prep = sc.build_records(b)
    assert prep["report"]["B_identical_to_A_pixels"] == 1


def test_swap_map_is_same_person_opposite_modality_same_input(tmp_path):
    prep = sc.build_records(binding(tmp_path))
    for arm in ("inA", "inB"):
        sm = prep["swap_map"][arm]
        assert len(sm) == 6
        by_id = {r["id"]: r for r in prep["records"][arm]}
        for k, v in sm.items():
            ex = by_id[k]["row_PRIVATE"]["extra_info"]
            ey = by_id[v]["row_PRIVATE"]["extra_info"]
            assert ex["patient_key"] == ey["patient_key"] and by_id[k]["row_PRIVATE"]["reward_model"] != by_id[v]["row_PRIVATE"]["reward_model"]


# ---- frozen protocol --------------------------------------------------------
def test_protocol_frozen_caps_and_digest_stable():
    p = sc.protocol()
    assert p["inputs"] == {"inA": "single legacy z045 image", "inB": "single physical-support median (f=0.50) image"}
    assert {"patient", "modality", "parent_checkpoint", "processor", "prompt_text", "normalization", "image_size", "decoder", "scorer"} <= set(p["held_constant"])
    caps = p["caps_per_checkpoint"]
    assert caps["greedy_calls"] == 12 and caps["candidate_score_API_calls"] == 36 and caps["generated_tokens"] == 12 * 256
    assert caps["teacherforced_input_positions"] == 36 * 425
    assert p["status"] == "NOT_RUN" and p["model_authorization"] == "NONE" and sc.protocol_digest() == sc.protocol_digest()
    assert "Direct" in json.dumps(p["effect_kinds"]) and "RL" in p["effect_kinds"]["data_input_effect"]


def test_protocol_prepare_writes_nothing_model_related(tmp_path):
    out = tmp_path / "o"
    res = sc.prepare(binding(tmp_path), out, tmp_path / "pub.json")
    j = json.loads((tmp_path / "pub.json").read_text())
    assert j["status"] == "NOT_RUN" and j["models_loaded"] == 0 and j["gpu_authorization"] == "NONE"
    assert str(tmp_path) not in json.dumps(j) and "r0flair" not in json.dumps(j)
    assert (out / "comparison_inputs_PRIVATE.json").exists() and res["records"] == 12
    with pytest.raises(ValueError):
        sc.prepare(binding(tmp_path), out, tmp_path / "pub2.json")


def test_run_entry_is_dormant_and_refuses():
    with pytest.raises(sc.NotAuthorized):
        sc.main(["run"])


# ---- analyzer (SYNTHETIC unit fixtures; never model predictions) ------------------
def resp(label):
    return f"<think>t</think><answer>\\boxed{{{label}}}</answer>"


def fake_rows(prep, arm, preds, scores=None):
    out = []
    for i, rec in enumerate(prep["records"][arm]):
        row = rec["row_PRIVATE"]
        lab = ev.LABELS["mri_sequence"]
        truth = row["reward_model"]["ground_truth"]
        txt = preds[i] if preds[i] is None else resp(preds[i])
        s = scores[i] if scores else (0.0, -1.0) if truth == lab[0] else (-1.0, 0.0)
        out.append({"id": rec["id"], "task": "mri_sequence", "truth": truth, "greedy": {"text": txt or "garbage"},
                    "candidates": {"per_label": [{"label": lab[0], "mean": s[0], "sum": s[0], "first": s[0]}, {"label": lab[1], "mean": s[1], "sum": s[1], "first": s[1]}]},
                    "costs": {"greedy_calls": 1, "candidate_score_API_calls": 3, "generated_tokens": 20, "teacherforced_input_positions": 300}})
    return out


def test_condition_metrics_strict_invalid_and_ties(tmp_path):
    prep = sc.build_records(binding(tmp_path))
    truths = [r["row_PRIVATE"]["reward_model"]["ground_truth"] for r in prep["records"]["inA"]]
    preds = list(truths)
    preds[0] = None  # invalid greedy
    rows = fake_rows(prep, "inA", preds)
    rows[1]["candidates"]["per_label"][0]["mean"] = rows[1]["candidates"]["per_label"][1]["mean"]  # tie
    m = sc.condition_metrics(rows, prep["meta"])
    assert m["greedy_invalid"] == 1 and m["strict_accuracy"] == pytest.approx(5 / 6)
    assert m["candidate"]["mean"]["invalid_ties"] == 1
    assert set(m["candidate"]) == {"mean", "sum", "first"} and m["costs"]["greedy_calls"] == 6
    assert m["flair_recall"] is not None and m["t1_recall"] is not None and "margin_correct_minus_wrong" in m["candidate"]["first"]
    assert m["swap"]["comparisons"] == 6


def test_analyze_refuses_without_provenance_and_labels_synthetic(tmp_path):
    prep = sc.build_records(binding(tmp_path))
    truths = [r["row_PRIVATE"]["reward_model"]["ground_truth"] for r in prep["records"]["inA"]]
    cond = {"parent|inA": fake_rows(prep, "inA", truths), "parent|inB": fake_rows(prep, "inB", truths)}
    with pytest.raises(ValueError, match="provenance"):
        sc.analyze(cond, prep["meta"], provenance=None)
    with pytest.raises(ValueError, match="receipt"):
        sc.analyze(cond, prep["meta"], provenance="ACTUAL_MODEL_OUTPUT")
    out = sc.analyze(cond, prep["meta"], provenance="SYNTHETIC_FIXTURE")
    assert out["model_prediction_claim"] is False and out["MRI_accuracy_improvement_claim"] == "NOT_ESTABLISHED"
    assert out["effects"]["data_input_effect_at_parent"]["kind"] == "DATA_INPUT_EFFECT_ONLY_NOT_RL_GAIN"


def test_three_way_effects_are_separated(tmp_path):
    prep = sc.build_records(binding(tmp_path))
    truths = [r["row_PRIVATE"]["reward_model"]["ground_truth"] for r in prep["records"]["inA"]]
    wrong = [ev.LABELS["mri_sequence"][1 - ev.LABELS["mri_sequence"].index(t)] for t in truths]
    cond = {}
    for ck, pa, pb in (("parent", wrong, wrong), ("trainedA", wrong, truths), ("trainedB", truths, truths)):
        cond[f"{ck}|inA"] = fake_rows(prep, "inA", pa)
        cond[f"{ck}|inB"] = fake_rows(prep, "inB", pb)
    out = sc.analyze(cond, prep["meta"], provenance="SYNTHETIC_FIXTURE")
    e = out["effects"]
    assert e["data_input_effect_at_parent"]["delta"]["strict_accuracy"] == 0
    assert e["data_input_effect_at_trainedA"]["delta"]["strict_accuracy"] == pytest.approx(1.0)
    assert e["training_arm_effect_trainedB_minus_trainedA_at_inA"]["kind"] == "TRAINING_ARM_EFFECT_AT_FIXED_INPUT"
    assert e["training_arm_effect_trainedB_minus_trainedA_at_inA"]["delta"]["strict_accuracy"] == pytest.approx(1.0)
    assert e["training_arm_effect_trainedB_minus_trainedA_at_inB"]["delta"]["strict_accuracy"] == 0
    assert e["input_x_training_interaction"]["delta"]["strict_accuracy"] == pytest.approx(-1.0)
    assert e["input_x_training_interaction"]["exploratory"] is True


def test_no_effects_invented_when_conditions_missing(tmp_path):
    prep = sc.build_records(binding(tmp_path))
    truths = [r["row_PRIVATE"]["reward_model"]["ground_truth"] for r in prep["records"]["inA"]]
    out = sc.analyze({"parent|inA": fake_rows(prep, "inA", truths)}, prep["meta"], provenance="SYNTHETIC_FIXTURE")
    assert out["effects"] == {}
