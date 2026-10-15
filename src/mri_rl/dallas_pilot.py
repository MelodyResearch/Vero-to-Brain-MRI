"""Bounded Dallas BIDS real-image CPU receipt. Never commit its output."""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evidence import display_to_voxel, render_volume_evidence
from .volume import load_canonical_volume


def run(images_root: Path, bids_root: Path, output_dir: Path) -> dict:
    participants = {r["participant_id"]: r for r in csv.DictReader(open(bids_root / "participants.tsv", encoding="utf-8"), delimiter="\t")}
    files = sorted(images_root.glob("sub-*/ses-wave*/anat/*.nii.gz"))
    if not files or len(files) > 16:
        raise ValueError("Pilot needs 1–16 bounded BIDS MRI files")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in files:
        rel = path.relative_to(images_root)
        subject, session = rel.parts[:2]
        if subject not in participants or session not in {"ses-wave1", "ses-wave2", "ses-wave3"}:
            raise ValueError(f"Unlinked subject or session: {rel}")
        if "acq-MPRAGE" in path.name and path.name.endswith("_T1w.nii.gz"):
            modality = "T1"
        elif "acq-FLAIR" in path.name and path.name.endswith("_T2w.nii.gz"):
            modality = "FLAIR"
        else:
            raise ValueError(f"Unsupported acquisition: {rel}")
        sidecar = bids_root / rel.with_suffix("").with_suffix(".json")
        if not sidecar.exists():
            raise ValueError(f"Missing acquisition JSON: {rel}")
        acquisition = json.loads(sidecar.read_text(encoding="utf-8"))
        volume = load_canonical_volume(path)
        data = volume["data"]
        evidence = render_volume_evidence(data, volume["affine"], output_dir / subject / session / modality)
        for view in evidence["views"]:
            assert display_to_voxel(view["anchor_pixel_display"], view["plane"], view["slice_index"], data.shape,
                                    view["display_scale"]) == tuple(view["voxel_anchor"])
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        annex_link = bids_root / rel
        if not annex_link.is_symlink():
            raise ValueError(f"Missing source annex link: {rel}")
        annex_key = re.search(r"SHA256E-s(\d+)--([0-9a-f]{64})\.nii\.gz", str(annex_link.readlink()))
        if not annex_key or (int(annex_key.group(1)), annex_key.group(2)) != (path.stat().st_size, sha):
            raise ValueError(f"Source annex size/hash mismatch: {rel}")
        row = participants[subject]
        records.append({"subject": subject, "session": session, "modality": modality,
                        "relative_path": str(rel), "sha256": sha, "bytes": path.stat().st_size,
                        "shape": list(data.shape), "spacing_mm": volume["spacing_mm"],
                        "affine": volume["affine"].tolist(), "orientation": volume["orientation"],
                        "nonzero_voxels": int(np.count_nonzero(data)),
                        "intensity_min_max": [float(data.min()), float(data.max())],
                        "manufacturer": acquisition.get("Manufacturer"),
                        "repetition_time": acquisition.get("RepetitionTime"),
                        "echo_time": acquisition.get("EchoTime"),
                        "mri_age_years": row.get("AgeMRI_W" + session[-1]),
                        "evidence": evidence})
    grouped = defaultdict(dict)
    for record in records:
        key = (record["subject"], record["modality"])
        if record["session"] in grouped[key]:
            raise ValueError(f"Duplicate subject/session/modality: {key}")
        grouped[key][record["session"]] = record
    pairs = []
    for (subject, modality), visits in sorted(grouped.items()):
        if "ses-wave1" not in visits or "ses-wave2" not in visits:
            continue
        interval = float(participants[subject]["MRIW1toW2"])
        first, second = visits["ses-wave1"], visits["ses-wave2"]
        if interval <= 0 or first["sha256"] == second["sha256"]:
            raise ValueError(f"Invalid longitudinal pair: {subject}, {modality}")
        spacing_ratio = [max(a, b) / min(a, b) for a, b in zip(first["spacing_mm"], second["spacing_mm"])]
        pairs.append({"subject": subject, "modality": modality, "first": first["session"],
                      "second": second["session"], "interval_years": interval,
                      "spacing_max_ratio": max(spacing_ratio),
                      "spacing_concordant_10pct": max(spacing_ratio) <= 1.1})
    receipt = {"dataset": "OpenNeuro ds004856", "source_version": "1.3.0",
               "source_commit": "077d0a4e4f32b7c18e3628893a96708ef24e1178",
               "source_url": "https://openneuro.org/datasets/ds004856/versions/1.3.0",
               "subjects": len({r["subject"] for r in records}),
               "sessions": len({(r["subject"], r["session"]) for r in records}),
               "volumes": len(records), "records": records, "longitudinal_pairs": pairs}
    (output_dir / "private_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args.images_root, args.bids_root, args.output_dir)
    print(json.dumps({"subjects": receipt["subjects"], "sessions": receipt["sessions"],
                      "volumes": receipt["volumes"], "T1_pairs": sum(p["modality"] == "T1" for p in receipt["longitudinal_pairs"]),
                      "FLAIR_pairs": sum(p["modality"] == "FLAIR" for p in receipt["longitudinal_pairs"]),
                      "spacing_discordant_pairs": sum(not p["spacing_concordant_10pct"] for p in receipt["longitudinal_pairs"])}, indent=2))


if __name__ == "__main__":
    main()
