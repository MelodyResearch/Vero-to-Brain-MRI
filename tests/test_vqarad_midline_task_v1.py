import copy
import hashlib
import pytest
from mri_rl.vqarad_midline_task_v1 import build_manifest, score_response


def fixture(tmp_path):
    rows = []
    plan = []
    images = {}
    for i, (split, target) in enumerate([("train", "yes"), ("train", "no"), ("eval", "yes"), ("eval", "no")]):
        name = f"image{i}.jpg"
        path = tmp_path / name
        path.write_bytes(f"synthetic image {i}".encode())
        images[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(dict(qid=str(i), image_name=name, image_case_url=f"https://medpix.nlm.nih.gov/case?id=case{i}", image_organ="HEAD", evaluation="evaluated", question="Is there a midline shift on this MRI?", answer=target, answer_type="CLOSED", question_type="PRES"))
        plan.append(dict(qid=str(i), split=split))
    return rows, plan, images


def test_balanced_manifest_preserves_original_and_hides_key(tmp_path):
    rows, plan, hashes = fixture(tmp_path)
    m = build_manifest(rows, plan, tmp_path, hashes)
    assert m["counts"] == {"train": {"yes": 1, "no": 1}, "eval": {"yes": 1, "no": 1}}
    assert m["gpu_authorized"] is False
    assert m["rows"][0]["question"] == rows[0]["question"]
    assert set(m["rows"][0]["model_input"]) == {"image", "question"}


def test_connected_case_alias_is_not_split_across_roles(tmp_path):
    rows, plan, hashes = fixture(tmp_path)
    # A non-selected paraphrase connects train image0 to eval image2.
    alias = copy.deepcopy(rows[0]); alias["qid"] = "alias"; alias["image_case_url"] = rows[2]["image_case_url"]
    with pytest.raises(ValueError, match="case component"):
        build_manifest(rows + [alias], plan, tmp_path, hashes)


def test_modified_image_rejected(tmp_path):
    rows, plan, hashes = fixture(tmp_path)
    (tmp_path / rows[0]["image_name"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA"):
        build_manifest(rows, plan, tmp_path, hashes)


@pytest.mark.parametrize("field,value", [("image_organ", "CHEST"), ("evaluation", "not evaluated"), ("answer", "uncertain"), ("question", "What causes brain edema?")])
def test_unqualified_source_row_rejected(tmp_path, field, value):
    rows, plan, hashes = fixture(tmp_path); rows[0][field] = value
    with pytest.raises(ValueError):
        build_manifest(rows, plan, tmp_path, hashes)


def test_duplicate_qid_and_duplicate_image_rejected(tmp_path):
    rows, plan, hashes = fixture(tmp_path)
    with pytest.raises(ValueError, match="qid"):
        build_manifest(rows + [rows[0]], plan, tmp_path, hashes)
    rows[2]["image_name"] = rows[0]["image_name"]
    with pytest.raises(ValueError):
        build_manifest(rows, plan, tmp_path, hashes)


def test_original_vero_format_and_reward_components():
    good = r"<think>Inspect the image.</think><answer>\boxed{yes}</answer>"
    assert score_response(good, "yes") == {"score": 1., "accuracy": 1., "format": 1., "invalid": False, "strict_accuracy": 1.}
    assert score_response(good, "no")["score"] == pytest.approx(.2)
    assert score_response("yes", "yes")["accuracy"] == 0.
    assert score_response("yes", "yes")["invalid"] is True
    conflicting = r"<think>Inspect.</think><answer>\boxed{no} \boxed{yes}</answer>"
    assert score_response(conflicting, "yes")["invalid"] is True
    assert score_response(conflicting, "yes")["strict_accuracy"] == 0.
    with pytest.raises(ValueError):
        score_response(good, "maybe")


def test_source_shifted_wording_and_integer_qid_are_preserved(tmp_path):
    rows, plan, hashes = fixture(tmp_path)
    rows[0]["qid"] = 0
    rows[0]["question"] = "Are brain structures shifted across the midline?"
    assert build_manifest(rows, plan, tmp_path, hashes)["rows"][0]["question"] == rows[0]["question"]
