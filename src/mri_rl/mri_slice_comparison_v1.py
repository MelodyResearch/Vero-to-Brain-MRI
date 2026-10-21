"""Frozen READ-ONLY slice-input comparison PREPARATION and CPU analyzer, v1. STATUS: NOT_RUN (no model, no GPU authorization).

Question isolated here (DATA INPUT only, never an RL/GSPO gain): for the same parent checkpoint, does a single physical-support
median image (input B, f=0.50) change sequence recognition relative to the single legacy z045 image (input A)? Everything else is
held constant: patient, modality, parent checkpoint, processor, prompt text (bit-exact), normalization, image size, decoder, scorer.
Condition keys are 'checkpoint|input' with checkpoint in parent/trainedA/trainedB (future Direct vs Paired-Prompt arms, never run
here) and input in inA/inB; the input names are deliberately different from the training-arm names to keep the contrasts apart.

Prepared records reuse the shape consumed by paired_z045_matched_f32_eval_v1.run_row (id/row_PRIVATE/image_sha256) so the existing
frozen evaluation kernels can run them after a separate authorization. This module imports no model backend. The `run` action is
dormant and always refuses. Synthetic analyzer fixtures are never model predictions.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

from mri_rl import paired_z045_evaluation_v1 as ev
from mri_rl.answer_likelihood import PREFIX

VERSION = "mri-slice-comparison-v1"
BINDING_SCHEMA = "mri-slice-comparison-binding-v1"
INPUTS = ("inA", "inB")
CHECKPOINTS = ("parent", "trainedA", "trainedB")
READOUTS = ("mean", "sum", "first")
COST_FIELDS = ("greedy_calls", "candidate_score_API_calls", "generated_tokens", "teacherforced_input_positions")
MODALITY_TRUTH = {"flair": "FLAIR", "t1ce": "contrast-enhanced T1"}
GREEDY = {"max_tokens": 256, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": None}
MAX_TF_POSITIONS = 425


class NotAuthorized(RuntimeError):
    pass


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


# ------------------------------------------------------------------ protocol
def caps(rows_per_input=6):
    n = 2 * rows_per_input  # both inputs under one checkpoint
    return {"records": n, "greedy_calls": n, "candidate_score_API_calls": 3 * n, "generated_tokens": 256 * n,
            "teacherforced_input_positions": 3 * n * MAX_TF_POSITIONS, "model_loads_per_checkpoint": 1,
            "swap_inference_calls": 0, "swap_rule": "same-person opposite-modality output within the SAME input reused; zero extra calls"}


def protocol(rows_per_input=6):
    return {
        "version": VERSION, "status": "NOT_RUN", "model_authorization": "NONE", "gpu_authorization": "NONE",
        "blocker": "No GPU/MLX/model authorization; certified base cache v3 is STOP (not PASS) and its permission is not reusable.",
        "inputs": {"inA": "single legacy z045 image", "inB": "single physical-support median (f=0.50) image"},
        "held_constant": ["patient", "modality", "parent_checkpoint", "processor", "prompt_text", "normalization", "image_size", "decoder", "scorer",
                          "image_count"],
        "only_changed_factor": "which single axial plane of the same volume is shown",
        "parent_checkpoint": "Adam52 (original parent) for the prepared data-input contrast",
        "decoder": GREEDY, "candidate_prefix": PREFIX, "label_order": list(ev.LABELS["mri_sequence"]),
        "caps_per_checkpoint": caps(rows_per_input),
        "endpoints": {"strict": "strict-format temperature-0 greedy sequence accuracy; invalid counts wrong",
                      "balanced_accuracy": "mean of FLAIR and contrast-enhanced T1 recall", "recalls": ["FLAIR", "contrast-enhanced T1"],
                      "candidate": "readouts mean (primary) / sum / first reported separately; ties INVALID; correct-minus-wrong margin by class and aggregate",
                      "swap_sensitivity": "same-person opposite-modality image, same input; changed/valid-opposite-class/both-correct",
                      "invalids": "greedy format failures and candidate ties counted, never dropped", "cost": "greedy calls, scorer calls, generated tokens, teacher-forced positions"},
        "effect_kinds": {
            "data_input_effect": "inB minus inA at the SAME checkpoint; a DATA effect only, not an RL gain",
            "training_arm_effect": "trained Direct same-prompt (trainedA) vs Paired-Prompt (trainedB) checkpoints at the SAME fixed input",
            "input_x_training_interaction": "difference of differences; exploratory, never a selected result"},
        "resolution": "3 people / 6 rows per input: one row moves a class recall by 1/3. Engineering preparation, not statistical evidence.",
        "claims_forbidden": ["MRI accuracy improved", "RL gain", "anatomical grounding", "clinical validity"],
        "new16_final100": "never opened or used to choose anything",
        "analysis_frozen_before_outputs": True,
    }


def protocol_digest(rows_per_input=6):
    return hashlib.sha256(json.dumps(protocol(rows_per_input), sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------- records
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pixels(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _typed_images_and_texts(row):
    content = row["prompt"][0]["content"]
    return [c["image"] for c in content if c.get("type") == "image"], [c["text"] for c in content if c.get("type") == "text"]


def rebind_image(row, new_path, new_id):
    """Copy the source row, replacing the image in BOTH the typed prompt entry and the top-level images; text untouched."""
    out = copy.deepcopy(row)
    for c in out["prompt"][0]["content"]:
        if c.get("type") == "image":
            c["image"] = str(new_path)
    out["images"] = [str(new_path)]
    out["id"] = new_id
    typed, _ = _typed_images_and_texts(out)
    require(typed == out["images"] and len(typed) == 1, "Typed prompt image and top-level images differ after rebinding")
    return out


def _check_row(r):
    row = r["old_row_PRIVATE"]
    require(row.get("ability") == "mri_sequence" and r["modality"] in MODALITY_TRUTH and r["truth"] == MODALITY_TRUTH[r["modality"]]
            and row["reward_model"]["ground_truth"] == r["truth"], "Row ability/modality/truth disagree")
    typed, texts = _typed_images_and_texts(row)
    require(len(typed) == 1 and len(row.get("images", [])) == 1 and typed == row["images"] and len(texts) == 1,
            "Source row typed prompt image and top-level images must be one identical image plus one text")
    require(Path(typed[0]).is_file() and Path(r["A_image"]).is_file() and Path(r["B_image"]).is_file(), "Image file missing")
    require(np.array_equal(_pixels(typed[0]), _pixels(r["A_image"])), "Input A is not pixel-identical to the old PNG")
    for key, path in (("A_sha256", r["A_image"]), ("B_sha256", r["B_image"])):
        require(not r.get(key) or r[key] == _sha(path), "Image hash mismatch: " + key)
    secrets = {str(r["opaque_id"]).lower(), str(r["pair_key"]).lower(), str(row.get("extra_info", {}).get("patient_key", "")).lower(),
               Path(r["A_image"]).name.lower(), Path(r["B_image"]).name.lower(), Path(typed[0]).name.lower()}
    secrets = {s for s in secrets if len(s) >= 3}
    visible = texts[0].lower()
    require(not any(s in visible for s in secrets), "Hidden identifier appears in model-visible text")
    return typed[0], texts[0]


def build_records(binding):
    """Verify a PRIVATE binding (3 people x flair/t1ce) and build the matched inA/inB records. No model, no image written."""
    require(binding.get("schema") == BINDING_SCHEMA and binding.get("rows"), "Bad binding schema")
    rows = binding["rows"]
    ids = [r["opaque_id"] for r in rows]
    require(len(set(ids)) == len(ids), "Opaque ids must be unique")
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair_key"], []).append(r["modality"])
    require(all(sorted(v) == ["flair", "t1ce"] for v in pairs.values()), "Each pair_key must hold exactly one flair and one t1ce row")
    records = {a: [] for a in INPUTS}
    meta, swap = {}, {a: {} for a in INPUTS}
    ident = abstain = 0
    for r in rows:
        _check_row(r)
        for arm, key in (("inA", "A_image"), ("inB", "B_image")):
            rid = f"{r['opaque_id']}_{arm}"
            row = rebind_image(r["old_row_PRIVATE"], r[key], rid)
            records[arm].append({"id": rid, "row_PRIVATE": row, "image_sha256": _sha(r[key])})
            meta[rid] = {"pair_key": r["pair_key"], "modality": r["modality"], "arm": arm}
        ident += int(np.array_equal(_pixels(r["A_image"]), _pixels(r["B_image"])))
        abstain += int(bool(r.get("B_abstain")))
    for arm in INPUTS:
        by_pair = {}
        for rec in records[arm]:
            by_pair.setdefault(meta[rec["id"]]["pair_key"], []).append(rec["id"])
        for a, b in by_pair.values():
            swap[arm][a], swap[arm][b] = b, a
    n = len(rows)
    return {"records": records, "swap_map": swap, "meta": meta,
            "report": {"rows_per_input": n, "people": len(pairs), "B_identical_to_A_pixels": ident, "B_abstain_flagged": abstain,
                       "text_bit_exact_across_inputs": True, "both_image_fields_bound": True}}


def prepare(binding, out_dir, public_json):
    out, pub = Path(out_dir), Path(public_json)
    require(not out.exists() or not any(out.iterdir()), "Output directory must be new/empty")
    require(not pub.exists(), "Public output must not pre-exist")
    built = build_records(binding)
    n = built["report"]["rows_per_input"]
    out.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "mri-slice-comparison-inputs-v1", "protocol_sha256": protocol_digest(n), "records": built["records"],
               "swap_map": built["swap_map"], "meta": built["meta"], "report": built["report"]}
    text = json.dumps(payload, indent=1, sort_keys=True)
    (out / "comparison_inputs_PRIVATE.json").write_text(text)
    public = {"schema": "mri-slice-comparison-public-v1", "status": "NOT_RUN", "models_loaded": 0, "gpu_authorization": "NONE",
              "protocol": protocol(n), "protocol_sha256": protocol_digest(n), "records": 2 * n, "report": built["report"],
              "inputs_sha256_16": hashlib.sha256(text.encode()).hexdigest()[:16],
              "note": "Prepared inputs only. No prediction, accuracy or gain exists; analyzer fixtures are SYNTHETIC."}
    pub.write_text(json.dumps(public, indent=1, sort_keys=True) + "\n")
    return {"records": 2 * n, "protocol_sha256": public["protocol_sha256"]}


# ------------------------------------------------------------------ analyzer
def _recall(rows, label, predict):
    group = [r for r in rows if r["truth"] == label]
    return sum(predict(r) == ev.LABELS["mri_sequence"].index(label) for r in group) / len(group) if group else None


def _balanced(a, b):
    return (a + b) / 2 if a is not None and b is not None else None


def condition_metrics(rows, meta):
    labels = ev.LABELS["mri_sequence"]
    require(rows and all(r["task"] == "mri_sequence" for r in rows), "Sequence rows required")
    greedy = lambda r: ev.strict_prediction(r["greedy"]["text"], "mri_sequence")
    fl, t1 = _recall(rows, labels[0], greedy), _recall(rows, labels[1], greedy)
    m = {"rows": len(rows), "strict_accuracy": sum(greedy(r) == labels.index(r["truth"]) for r in rows) / len(rows),
         "flair_recall": fl, "t1_recall": t1, "balanced_accuracy": _balanced(fl, t1), "greedy_invalid": sum(greedy(r) == -1 for r in rows),
         "candidate": {}}
    for readout in READOUTS:
        def scores(r):
            by = {i["label"]: i[readout] for i in r["candidates"]["per_label"]}
            return [by[labels[0]], by[labels[1]]]
        pred = lambda r: ev.candidate_prediction(scores(r))
        cf, ct = _recall(rows, labels[0], pred), _recall(rows, labels[1], pred)
        margins = {l: [] for l in labels}
        for r in rows:
            s, t = scores(r), labels.index(r["truth"])
            margins[r["truth"]].append(s[t] - s[1 - t])
        every = [x for g in margins.values() for x in g]
        m["candidate"][readout] = {"balanced_accuracy": _balanced(cf, ct), "flair_recall": cf, "t1_recall": ct, "invalid_ties": sum(pred(r) == -1 for r in rows),
                                   "margin_correct_minus_wrong": {"FLAIR": sum(margins[labels[0]]) / len(margins[labels[0]]) if margins[labels[0]] else None,
                                                                  "T1": sum(margins[labels[1]]) / len(margins[labels[1]]) if margins[labels[1]] else None,
                                                                  "aggregate": sum(every) / len(every)}}
    by_id = {r["id"]: r for r in rows}
    real, swapped, truth = [], [], []
    for r in rows:
        info = meta[r["id"]]
        partner = next((i for i, x in meta.items() if x["pair_key"] == info["pair_key"] and x["arm"] == info["arm"] and i != r["id"]), None)
        require(partner in by_id, "Swap partner output missing")
        real.append(greedy(r))
        swapped.append(greedy(by_id[partner]))
        truth.append(labels.index(r["truth"]))
    m["swap"] = ev.swap_summary(real, swapped, truth)
    m["costs"] = {k: sum(r["costs"][k] for r in rows) for k in COST_FIELDS}
    return m


def numeric_delta(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        out = {k: numeric_delta(left[k], right[k]) for k in left if k in right}
        return {k: v for k, v in out.items() if v is not None}
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        return left - right
    return None


def _body(m):
    return {k: v for k, v in m.items() if k not in ("costs", "rows")}


def analyze(conditions, meta, *, provenance, receipt_sha256=None):
    """conditions: {'checkpoint|input': [row,...]}. Rows have the run_row shape. provenance must be declared; real outputs need a receipt hash."""
    require(provenance in ("ACTUAL_MODEL_OUTPUT", "SYNTHETIC_FIXTURE"), "Declared provenance required")
    require(provenance != "ACTUAL_MODEL_OUTPUT" or receipt_sha256, "Actual model output needs a receipt sha256")
    metrics = {}
    for key, rows in conditions.items():
        ck, inp = key.split("|")
        require(ck in CHECKPOINTS and inp in INPUTS, "Unknown condition key")
        metrics[key] = condition_metrics(rows, meta)
    M = lambda ck, inp: metrics.get(f"{ck}|{inp}")
    effects = {}
    for ck in CHECKPOINTS:
        if M(ck, "inA") and M(ck, "inB"):
            effects[f"data_input_effect_at_{ck}"] = {"kind": "DATA_INPUT_EFFECT_ONLY_NOT_RL_GAIN", "contrast": f"{ck}: inB minus inA",
                                                     "delta": numeric_delta(_body(M(ck, "inB")), _body(M(ck, "inA")))}
    for inp in INPUTS:
        if M("trainedA", inp) and M("trainedB", inp):
            effects[f"training_arm_effect_trainedB_minus_trainedA_at_{inp}"] = {
                "kind": "TRAINING_ARM_EFFECT_AT_FIXED_INPUT", "delta": numeric_delta(_body(M("trainedB", inp)), _body(M("trainedA", inp)))}
        for arm in ("trainedA", "trainedB"):
            if M("parent", inp) and M(arm, inp):
                effects[f"training_arm_effect_{arm}_minus_parent_at_{inp}"] = {
                    "kind": "TRAINING_ARM_EFFECT_AT_FIXED_INPUT", "delta": numeric_delta(_body(M(arm, inp)), _body(M("parent", inp)))}
    if all(M(a, i) for a in ("trainedA", "trainedB") for i in INPUTS):
        d = lambda a: numeric_delta(_body(M(a, "inB")), _body(M(a, "inA")))
        effects["input_x_training_interaction"] = {"kind": "INTERACTION_DIFFERENCE_OF_DIFFERENCES", "exploratory": True,
                                                   "delta": numeric_delta(d("trainedB"), d("trainedA"))}
    return {"schema": "mri-slice-comparison-analysis-v1", "protocol_sha256": protocol_digest(), "provenance": provenance,
            "model_prediction_claim": provenance == "ACTUAL_MODEL_OUTPUT", "receipt_sha256": receipt_sha256,
            "metrics": metrics, "effects": effects, "MRI_accuracy_improvement_claim": "NOT_ESTABLISHED",
            "limits": protocol()["resolution"], "training_authorized": False}


# ----------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "analyze", "run"))
    ap.add_argument("--binding")
    ap.add_argument("--inputs")
    ap.add_argument("--results")
    ap.add_argument("--private-out")
    ap.add_argument("--public-out")
    ap.add_argument("--provenance")
    ap.add_argument("--receipt-sha256")
    a = ap.parse_args(argv)
    if a.action == "run":
        raise NotAuthorized("NOT_RUN: no GPU/MLX/model authorization and certified cache v3 is STOP; this entry never loads a model.")
    if a.action == "prepare":
        require(a.binding and a.private_out and a.public_out, "--binding --private-out --public-out required")
        res = prepare(json.loads(Path(a.binding).read_text()), a.private_out, a.public_out)
        print(json.dumps({"status": "NOT_RUN", "prepared": res, "models_loaded": 0}, sort_keys=True))
        return 0
    require(a.inputs and a.results and a.public_out, "--inputs --results --public-out required")
    inputs = json.loads(Path(a.inputs).read_text())
    out = analyze(json.loads(Path(a.results).read_text()), inputs["meta"], provenance=a.provenance, receipt_sha256=a.receipt_sha256)
    require(not Path(a.public_out).exists(), "Public output must not pre-exist")
    Path(a.public_out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"provenance": out["provenance"], "effects": sorted(out["effects"])}, sort_keys=True))
    return 0
