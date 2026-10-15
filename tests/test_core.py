import json
from pathlib import Path

import pytest
from PIL import Image

from mri_rl.core import (
    audit_records,
    balanced_sample,
    evaluate,
    frame_iou,
    render_frames,
    reward,
)


def test_frame_iou_requires_matching_frame():
    target = {"frame": 2, "box": [0, 0, 10, 10]}
    assert frame_iou(target, target) == 1.0
    assert frame_iou({"frame": 1, "box": [0, 0, 10, 10]}, target) == 0.0
    assert frame_iou({"frame": 2, "box": [10, 10, 0, 0]}, target) == 0.0


def test_task_rewards_and_invalid_output():
    assert reward("categorical", "T1", {"label": "t1"}) == 1.0
    assert reward("numeric", "10.4", {"value": 10, "absolute_tolerance": 0.5}) == 1.0
    assert reward("numeric", "100", {"value": 10, "absolute_tolerance": 0.5}) == 0.0
    assert reward("longitudinal_direction", "decrease", {"direction": "decrease"}) == 1.0
    assert reward("spatial", '{"frame": 2, "box": [0, 0, 10, 10]}', {"frame": 2, "box": [0, 0, 10, 10]}) == 1.0
    assert reward("spatial", "no lesion", {"frame": 2, "box": [0, 0, 10, 10]}) == 0.0


def test_subject_split_and_image_audit(tmp_path):
    p = tmp_path / "frame.png"
    Image.new("L", (8, 8), 100).save(p)
    base = {"study_id": "study1", "task": "categorical", "frames": [{"path": str(p), "index": 0}], "target": {"label": "T1"}}
    train = [{**base, "id": "a", "subject_id": "s1"}]
    test = [{**base, "id": "b", "subject_id": "s1"}]
    report = audit_records({"train": train, "test": test})
    assert report["subject_overlap"] == ["s1"]
    assert report["decodable_frames"] == 2
    assert report["usable_records"] == 0


def test_spatial_target_must_resolve_inside_source_frame(tmp_path):
    p = tmp_path / "frame.png"
    Image.new("L", (8, 8), 100).save(p)
    row = {"id": "box", "subject_id": "s1", "study_id": "st1", "task": "spatial",
           "frames": [{"path": str(p), "index": 0}], "target": {"frame": 0, "box": [0, 0, 9, 5]}}
    report = audit_records({"test": [row]})
    assert report["usable_records"] == 0
    assert report["exclusion_reasons"]["invalid_spatial_target"] == 1


def test_render_preserves_frame_order(tmp_path):
    frames = []
    for index, val in [(3, 30), (1, 100)]:
        p = tmp_path / f"{index}.png"
        Image.new("L", (8, 8), val).save(p)
        frames.append({"index": index, "path": str(p)})
    out = tmp_path / "montage.png"
    info = render_frames(frames, out, tile_size=16)
    assert info["frame_indices"] == [1, 3]
    assert out.exists()


def test_balanced_sampler_determinism_and_evaluation():
    records = [{"id": f"a{i}", "task": "a"} for i in range(10)] + [{"id": "b0", "task": "b"}]
    sample = balanced_sample(records, 6, seed=7)
    assert [r["task"] for r in sample].count("a") == 3
    assert [r["task"] for r in sample].count("b") == 3
    assert sample == balanced_sample(records, 6, seed=7)
    eval_rows = [
        {"id": "a", "subject_id": "s1", "task": "categorical", "target": {"label": "T1"}},
        {"id": "b", "subject_id": "s2", "task": "categorical", "target": {"label": "T2"}},
    ]
    report = evaluate(eval_rows, {"a": "T1", "b": "wrong"})
    assert report["task_scores"]["categorical"]["mean"] == 0.5
    assert report["subjects"] == 2
