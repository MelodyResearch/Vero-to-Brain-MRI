"""Paired verifier checks, including one test backed by real Dallas NIfTI bytes."""

import copy
from pathlib import Path

import pytest

from mri_rl.paired_verifier import load_verified_cohort, verify_pair, provenance_baseline, acquisition_compatible, run_challenge


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cohort():
    if not (ROOT / "local_data/dallas_expansion_private_receipt.json").exists():
        pytest.skip("Dallas public image payload absent")
    return load_verified_cohort(ROOT)


def _pair(cohort):
    return next((a, b) for a, b in cohort["pairs"] if a["subject"] == b["subject"])


def test_real_image_bytes_and_correct_pair_abstains_on_unvalidated_change(cohort):
    a, b = _pair(cohort)
    assert a["bytes_verified"] and b["bytes_verified"]
    result = verify_pair(a, b, cohort["trusted"])
    assert result["provenance"] == "ACCEPT"
    assert result["change"] == "NOT_EVALUABLE"


def test_reversed_and_wrong_person_pairs_rejected(cohort):
    a, b = _pair(cohort)
    other = next(x for x in cohort["scans"] if x["subject"] != a["subject"] and x["session"] == "ses-wave1")
    assert verify_pair(b, a, cohort["trusted"])["reason"] == "VISIT_ORDER"
    assert verify_pair(other, b, cohort["trusted"])["reason"] == "SUBJECT_MISMATCH"


def test_wrong_sequence_and_affine_mutation_rejected(cohort):
    a, b = _pair(cohort)
    flair = next(x for x in cohort["scans"] if x["modality"] == "FLAIR" and x["session"] == "ses-wave1")
    matching_current = next(x for x in cohort["scans"] if x["subject"] == flair["subject"] and x["session"] == "ses-wave2" and x["modality"] == "T1")
    assert verify_pair(flair, matching_current, cohort["trusted"])["reason"] == "SEQUENCE_MISMATCH"
    flipped = copy.deepcopy(a)
    flipped["affine"][0][0] *= -1
    assert verify_pair(flipped, b, cohort["trusted"])["reason"] == "AFFINE_MISMATCH"


def test_registration_and_measurement_gate(cohort):
    a, b = _pair(cohort)
    bad = verify_pair(a, b, cohort["trusted"], registration={"qc_pass": False})
    assert bad["reason"] == "REGISTRATION_QC_FAIL"
    assert bad["provenance"] == "ACCEPT"
    assert bad["image_evidence"] == "REJECT"
    fabricated = verify_pair(a, b, cohort["trusted"], measurement={"region": "hippocampus", "delta": -1.0})
    assert fabricated["reason"] == "UNSUPPORTED_MEASUREMENT"
    subthreshold = verify_pair(a, b, cohort["trusted"], measurement={"region": "hippocampus", "delta": -0.2,
        "source": "independent", "repeatability_limit": 0.5, "independent_reference": True})
    assert subthreshold["change"] == "NOT_EVALUABLE"
    assert subthreshold["reason"] == "BELOW_REPEATABILITY"


def test_simple_provenance_baseline_cannot_check_geometry(cohort):
    a, b = _pair(cohort)
    flipped = copy.deepcopy(a)
    flipped["affine"][0][0] *= -1
    assert provenance_baseline(flipped, b)["provenance"] == "ACCEPT"
    assert verify_pair(flipped, b, cohort["trusted"])["provenance"] == "REJECT"


def test_post_receipt_file_change_is_rejected(cohort, tmp_path):
    a, b = _pair(cohort)
    altered = copy.deepcopy(a)
    destination = tmp_path / "altered.nii.gz"
    destination.write_bytes(Path(a["local_path"]).read_bytes() + b"x")
    altered["local_path"] = str(destination)
    assert verify_pair(altered, b, cohort["trusted"])["reason"] == "SOURCE_BYTE_MISMATCH"


def test_real_flair_acquisition_shift_detected(cohort):
    flair = [x for x in cohort["scans"] if x["modality"] == "FLAIR"]
    pairs = [(a, b) for a in flair for b in flair if a["subject"] == b["subject"] and a["session"] == "ses-wave1" and b["session"] == "ses-wave2"]
    assert len(pairs) == 2
    assert sum(not acquisition_compatible(a, b) for a, b in pairs) >= 1


def test_patient_level_challenge_uses_real_registration_results(cohort):
    path = ROOT / "local_data/longitudinal_t1_registration_private.json"
    if not path.exists():
        pytest.skip("Registration QC receipt absent")
    output = run_challenge(cohort, path)
    assert output["subjects"] == 20
    assert output["scenarios"]["correct_pair"]["provenance_accept"] == 20
    assert output["scenarios"]["correct_pair"]["change_not_evaluable"] == 20
    assert output["scenarios"]["correct_pair"]["image_hold"] == 1
    assert output["scenarios"]["wrong_person"]["provenance_reject"] == 20
    assert output["scenarios"]["flipped_header"]["provenance_reject"] == 20
    assert output["scenarios"]["pixel_lr_flip"]["provenance_reject"] == 1
    assert output["scenarios"]["unsupported_measurement"]["change_reject"] == 20
    assert output["scenarios"]["below_repeatability_fixture"]["change_not_evaluable"] == 20
