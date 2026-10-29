"""Saved-result reproduction must reject inconsistent scientific counts."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mri_rl.final_results import build_final_metrics, summarize_confusion

ROOT = Path(__file__).resolve().parents[1]


def test_confusion_recomputes_accuracy_and_rejects_count_mismatch():
    got = summarize_confusion([[10, 2, 0], [4, 8, 0]], [12, 12])
    assert got["correct"] == 18
    assert got["accuracy"] == 0.75
    assert got["class_recalls"] == [10 / 12, 8 / 12]
    assert got["invalid"] == 0
    with pytest.raises(ValueError, match="row"):
        summarize_confusion([[10, 3, 0], [4, 8, 0]], [12, 12])
    with pytest.raises(ValueError):
        summarize_confusion([[10, -1, 0], [4, 8, 0]], [9, 12])


def test_actual_recount_keeps_datasets_separate_and_holdout_unrun():
    got = build_final_metrics(ROOT)
    dev = got["explored_DEV24"]["models"]
    assert [dev[k]["correct"] for k in ["Parent_Adam52", "Direct_Vero_A54", "Paired_Vero_B54", "Balanced_SFT_Adam80_seed0"]] == [18, 16, 16, 15]
    assert dev["Parent_Adam52"]["confusion"] == [[10, 2, 0], [4, 8, 0]]
    assert dev["Balanced_SFT_Adam80_seed0"]["class_recalls"] == [1.0, 0.25]
    assert dev["Parent_Adam52"]["descriptive_four_patient_cluster_percentile95"] == pytest.approx([7/12, 11/12])
    assert got["original_frozen_final100"]["unique_model_runs"] == 22
    assert len(got["original_frozen_final100"]["families"]) == 8
    assert got["independent_holdout"]["status"] == "NOT_RUN_INPUT_METADATA_NOT_LOCATED_AND_NOT_UNSEALED"
    assert got["training_authorized"] is False


def test_changed_source_hash_is_rejected(tmp_path):
    from mri_rl.final_results import SOURCE_SHA256
    (tmp_path / "results").mkdir()
    for name in SOURCE_SHA256:
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    name = "results/ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json"
    d = json.loads((tmp_path / name).read_text())
    d["metrics"]["Parent"]["correct"] = 19
    (tmp_path / name).write_text(json.dumps(d))
    with pytest.raises(ValueError, match="SHA256"):
        build_final_metrics(tmp_path)


def test_inconsistent_reported_metric_is_rejected():
    from mri_rl.final_results import validate_reported
    with pytest.raises(ValueError, match="accuracy"):
        validate_reported({"correct": 18, "accuracy": 0.9}, summarize_confusion([[10, 2, 0], [4, 8, 0]], [12, 12]))


def test_cli_deterministic_output_check_and_corruption(tmp_path):
    out = tmp_path / "metrics.json"
    cmd = [sys.executable, str(ROOT / "scripts/reproduce_results.py")]
    a = subprocess.run(cmd, cwd=tmp_path, text=True, capture_output=True, check=True)
    b = subprocess.run(cmd + ["--output", str(out)], cwd=tmp_path, text=True, capture_output=True, check=True)
    assert json.loads(a.stdout) == json.loads(b.stdout) == json.loads(out.read_text())
    subprocess.run(cmd + ["--check", str(out)], text=True, capture_output=True, check=True)
    d = json.loads(out.read_text())
    d["explored_DEV24"]["models"]["Parent_Adam52"]["correct"] = 19
    out.write_text(json.dumps(d))
    bad = subprocess.run(cmd + ["--check", str(out)], text=True, capture_output=True)
    assert bad.returncode != 0

def test_package_entry_prints_same_result_without_private_or_model_imports(capsys):
    from mri_rl.final_results import main
    assert main([]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["explored_DEV24"]["models"]["Parent_Adam52"]["correct"] == 18
    assert not any(name == "mlx" or name.startswith("mlx.") or name == "torch" for name in sys.modules)


@pytest.mark.parametrize("collection_args", [[], ["."]])
def test_checkout_collection_does_not_import_private_runtime_archives(tmp_path, collection_args):
    """Collect only supported tests, even when local runtime archives are present."""
    (tmp_path / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    public = tmp_path / "tests"
    public.mkdir()
    (public / "test_public_boundary.py").write_text("def test_public_boundary():\n    assert True\n")
    markers = []
    for name in ["local_data", "work", ".worktrees"]:
        private = tmp_path / name
        private.mkdir()
        marker = tmp_path / (name.replace(".", "_") + "_IMPORTED")
        markers.append(marker)
        (private / "test_forbidden_archive.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('unexpected archive import')\n"
            "raise RuntimeError('private runtime archive must not be imported')\n"
        )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"] + collection_args,
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_public_boundary" in result.stdout
    assert all(not marker.exists() for marker in markers)
