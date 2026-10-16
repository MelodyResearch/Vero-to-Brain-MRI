"""Deterministic paired-MRI provenance gate and explicit biological abstention.

The trusted source receipt is generated from real Dallas NIfTI bytes. Its BIDS
identity is an engineering reference, not independent anatomical evidence.
"""

from __future__ import annotations

import hashlib
import json
import copy
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_verified_cohort(root: Path) -> dict:
    """Load local T1 and FLAIR, verify bytes, and form 12 T1 pairs."""
    data_root = root / "local_data/ds004856"
    pilot = json.loads((root / "local_data/ds004856_evidence/private_receipt.json").read_text())
    expanded = json.loads((root / "local_data/dallas_expansion_private_receipt.json").read_text())
    raw = list(pilot["records"])
    for person in expanded["records"]:
        for image in person["images"]:
            raw.append({"subject": person["subject"], "session": Path(image["relative_path"]).parts[1],
                        "modality": "T1", **image})
    scans = []
    for item in raw:
        rel = item["relative_path"]
        path = data_root / rel
        if not path.is_file() or path.stat().st_size != item["bytes"] or _hash_file(path) != item["sha256"]:
            raise ValueError("Source byte/size mismatch")
        image = nib.load(str(path))
        affine = np.asarray(image.affine, dtype=float)
        if len(image.shape) != 3 or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-6:
            raise ValueError("Invalid NIfTI geometry")
        # The original pilot receipt records canonical RAS geometry; the
        # expansion receipt records the native NIfTI header geometry.
        receipt_image = nib.as_closest_canonical(image) if "modality" in item and "intensity_min_max" in item else image
        if tuple(item["shape"]) != tuple(receipt_image.shape) or not np.allclose(item["affine"], receipt_image.affine, atol=1e-3):
            raise ValueError("Source receipt geometry mismatch")
        scans.append({"relative_path": rel, "local_path": str(path), "subject": item["subject"],
                      "session": item["session"], "modality": item["modality"],
                      "sha256": item["sha256"], "bytes": item["bytes"], "bytes_verified": True,
                      "affine": affine.tolist(), "orientation": "".join(nib.aff2axcodes(affine)),
                      "spacing_mm": list(nib.affines.voxel_sizes(affine)),
                      "manufacturer": item.get("manufacturer"),
                      "repetition_time_s": item.get("repetition_time_s", item.get("repetition_time")),
                      "echo_time_s": item.get("echo_time_s", item.get("echo_time"))})
    by_person = {}
    for scan in scans:
        if scan["modality"] == "T1":
            by_person.setdefault(scan["subject"], {})[scan["session"]] = scan
    pairs = []
    for person, visits in by_person.items():
        if "ses-wave1" in visits and "ses-wave2" in visits:
            pairs.append((visits["ses-wave1"], visits["ses-wave2"]))
    return {"scans": scans, "pairs": pairs, "trusted": {s["relative_path"]: s for s in scans}}


def provenance_baseline(prior: dict, current: dict) -> dict:
    """Minimal subject/chronology/sequence rule, without source or image QC."""
    if prior["subject"] != current["subject"]:
        return {"provenance": "REJECT", "reason": "SUBJECT_MISMATCH"}
    if prior["session"] != "ses-wave1" or current["session"] != "ses-wave2":
        return {"provenance": "REJECT", "reason": "VISIT_ORDER"}
    if prior["modality"] != "T1" or current["modality"] != "T1":
        return {"provenance": "REJECT", "reason": "SEQUENCE_MISMATCH"}
    return {"provenance": "ACCEPT", "reason": "PROVENANCE_ONLY"}


def acquisition_compatible(prior: dict, current: dict) -> bool:
    """Conservative acquisition comparability; not a full scanner harmonization test."""
    if prior["modality"] != current["modality"] or prior["manufacturer"] != current["manufacturer"]:
        return False
    for field in ("repetition_time_s", "echo_time_s"):
        lo, hi = float(prior[field]), float(current[field])
        if lo <= 0 or hi <= 0 or max(lo, hi) / min(lo, hi) > 1.03:
            return False
    for lo, hi in zip(prior["spacing_mm"], current["spacing_mm"]):
        if lo <= 0 or hi <= 0 or max(lo, hi) / min(lo, hi) > 1.1:
            return False
    return True


def verify_pair(prior: dict, current: dict, trusted: dict,
                registration: dict | None = None, measurement: dict | None = None) -> dict:
    """Gate source/geometry first; withhold biological claims absent reference."""
    def result(provenance: str, image_evidence: str, change: str, reason: str) -> dict:
        return {"provenance": provenance, "image_evidence": image_evidence,
                "change": change, "reason": reason}

    for candidate in (prior, current):
        source = trusted.get(candidate.get("relative_path"))
        if source is None or not source.get("bytes_verified"):
            return result("REJECT", "NOT_EVALUABLE", "NOT_EVALUABLE", "UNTRUSTED_SOURCE")
        actual_path = Path(candidate.get("local_path", ""))
        if (not actual_path.is_file() or actual_path.stat().st_size != source["bytes"]
                or _hash_file(actual_path) != source["sha256"]):
            return result("REJECT", "NOT_EVALUABLE", "NOT_EVALUABLE", "SOURCE_BYTE_MISMATCH")
        for field in ("subject", "session", "modality", "sha256", "bytes", "manufacturer",
                      "repetition_time_s", "echo_time_s"):
            if candidate.get(field) != source.get(field):
                return result("REJECT", "NOT_EVALUABLE", "NOT_EVALUABLE", "SCAN_METADATA_TAMPER")
        if not np.allclose(candidate.get("affine"), source["affine"], atol=1e-3):
            return result("REJECT", "NOT_EVALUABLE", "NOT_EVALUABLE", "AFFINE_MISMATCH")
        if not np.allclose(candidate.get("spacing_mm"), source["spacing_mm"], atol=1e-3):
            return result("REJECT", "NOT_EVALUABLE", "NOT_EVALUABLE", "SPACING_MISMATCH")
        if candidate.get("orientation") != source["orientation"]:
            return result("REJECT", "NOT_EVALUABLE", "NOT_EVALUABLE", "ORIENTATION_MISMATCH")
    base = provenance_baseline(prior, current)
    if base["provenance"] == "REJECT":
        return result("REJECT", "NOT_EVALUABLE", "NOT_EVALUABLE", base["reason"])
    if not acquisition_compatible(prior, current):
        return result("REJECT", "NOT_EVALUABLE", "NOT_EVALUABLE", "PROTOCOL_MISMATCH")
    if registration is not None and not registration.get("qc_pass", False):
        return result("ACCEPT", "REJECT", "NOT_EVALUABLE", "REGISTRATION_QC_FAIL")
    image_status = "NOT_EVALUABLE" if registration is None else "NUMERIC_PASS_UNCERTIFIED"
    if registration is not None and registration.get("visual_qc_status") == "HELD_ARTIFACT":
        image_status = "HOLD"
    if measurement is None:
        return result("ACCEPT", image_status, "NOT_EVALUABLE", "NO_VALIDATED_CHANGE_REFERENCE")
    if not measurement.get("source") or not measurement.get("independent_reference"):
        return result("ACCEPT", image_status, "REJECT", "UNSUPPORTED_MEASUREMENT")
    threshold = measurement.get("repeatability_limit")
    delta = measurement.get("delta")
    if threshold is None or delta is None or not np.isfinite([threshold, delta]).all() or threshold <= 0:
        return result("ACCEPT", image_status, "NOT_EVALUABLE", "REPEATABILITY_UNKNOWN")
    if abs(delta) <= threshold:
        return result("ACCEPT", image_status, "NOT_EVALUABLE", "BELOW_REPEATABILITY")
    return result("ACCEPT", image_status, "NOT_EVALUABLE", "REFERENCE_INDEPENDENCE_NOT_AUDITED")


def run_challenge(cohort: dict, registration_receipt: Path) -> dict:
    """Execute patient-level source/geometry and abstention adversarial cases.

    Returns aggregate counts only. The registration receipt remains private.
    Controlled header and measurement edits are engineering attacks on real
    source observations, not claims about actual patient pathology.
    """
    reg = {r["subject"]: r for r in json.loads(registration_receipt.read_text())["records"]}
    pairs = sorted(cohort["pairs"], key=lambda p: p[0]["subject"])
    if len(pairs) != len(reg):
        raise ValueError("Registration and image cohorts differ")
    results = defaultdict(Counter)

    def add(name: str, answer: dict, base: dict | None = None) -> None:
        results[name]["n"] += 1
        results[name]["provenance_" + answer["provenance"].lower()] += 1
        results[name]["image_" + answer["image_evidence"].lower()] += 1
        results[name]["change_" + answer["change"].lower()] += 1
        results[name]["reason_" + answer["reason"].lower()] += 1
        if base is not None:
            results[name]["baseline_" + base["provenance"].lower()] += 1

    for i, (prior, current) in enumerate(pairs):
        registration = reg[prior["subject"]]
        if [Path(x).resolve() for x in registration["paths"]] != [Path(prior["local_path"]).resolve(), Path(current["local_path"]).resolve()]:
            raise ValueError("Registration input files do not match the trusted source")
        add("correct_pair", verify_pair(prior, current, cohort["trusted"], registration), provenance_baseline(prior, current))
        add("reversed_visits", verify_pair(current, prior, cohort["trusted"]), provenance_baseline(current, prior))
        other_prior = pairs[(i + 1) % len(pairs)][0]
        add("wrong_person", verify_pair(other_prior, current, cohort["trusted"]), provenance_baseline(other_prior, current))
        flipped = copy.deepcopy(prior)
        for row in range(3):
            flipped["affine"][row][0] *= -1
        add("flipped_header", verify_pair(flipped, current, cohort["trusted"]), provenance_baseline(flipped, current))
        shifted = copy.deepcopy(prior)
        shifted["affine"][0][3] += 20.0
        add("affine_shift_20mm", verify_pair(shifted, current, cohort["trusted"]), provenance_baseline(shifted, current))
        add("unsupported_measurement", verify_pair(prior, current, cohort["trusted"], registration,
            {"region": "hippocampus", "delta": -1.0}), provenance_baseline(prior, current))
        add("repeatability_unknown", verify_pair(prior, current, cohort["trusted"], registration,
            {"region": "hippocampus", "delta": -0.1, "source": "engineering fixture",
             "independent_reference": True}), provenance_baseline(prior, current))
        # Hypothetical numeric fixture to exercise abstention; its claimed
        # independence flag is not an actual Dallas reference or result.
        add("below_repeatability_fixture", verify_pair(prior, current, cohort["trusted"], registration,
            {"region": "hippocampus", "delta": -0.2, "repeatability_limit": 0.5,
             "source": "engineering fixture", "independent_reference": True}),
            provenance_baseline(prior, current))
        # The no-registration grid comparison comes from real image intensities.
        # Treating it as a substitute for registered evidence is an attack.
        before = registration["before"]
        unregistered_qc = bool(before["foreground_dice"] >= 0.75 and before["foreground_pearson"] >= 0.50)
        add("unregistered_grid", verify_pair(prior, current, cohort["trusted"],
            {"qc_pass": unregistered_qc}), provenance_baseline(prior, current))
    flair = [x for x in cohort["scans"] if x["modality"] == "FLAIR"]
    for prior in flair:
        if prior["session"] != "ses-wave1":
            continue
        current = next(x for x in flair if x["subject"] == prior["subject"] and x["session"] == "ses-wave2")
        t1_current = next(x for x in cohort["scans"] if x["subject"] == prior["subject"]
                          and x["session"] == "ses-wave2" and x["modality"] == "T1")
        add("wrong_sequence", verify_pair(prior, t1_current, cohort["trusted"]), provenance_baseline(prior, t1_current))
        results["flair_protocol_stress"]["n"] += 1
        results["flair_protocol_stress"]["incompatible" if not acquisition_compatible(prior, current) else "compatible"] += 1
    # A true array transformation of one original NIfTI, not merely a header
    # edit. Exact release hash catches this; no anatomical asymmetry check is
    # claimed. The derivative stays under ignored local_data.
    prior, current = pairs[0]
    source_image = nib.load(prior["local_path"])
    reflected = np.asarray(source_image.dataobj)[::-1, :, :].copy()
    derivative = registration_receipt.parent / "verifier_pixel_lr_flip.nii.gz"
    nib.save(nib.Nifti1Image(reflected, source_image.affine, source_image.header), str(derivative))
    modified = copy.deepcopy(prior)
    modified["local_path"] = str(derivative)
    add("pixel_lr_flip", verify_pair(modified, current, cohort["trusted"]), provenance_baseline(modified, current))
    return {"subjects": len(pairs), "scenarios": {k: dict(v) for k, v in sorted(results.items())},
            "interpretation": "Engineering verifier challenge on real Dallas scans; no validated biological change truth."}
