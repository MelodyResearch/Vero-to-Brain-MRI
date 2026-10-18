"""Independent MRI-accuracy V2: patient-paired, image-conditioned label discrimination.

No change to the frozen GSPO/SFT protocols. A positive image-pair separation is
not proof of improved held-out accuracy. Data and checkpoints stay private.

The training objective differentiates two images of the SAME patient at the SAME
axial level. It penalizes both wrong-class scores and insufficient separation of
the two image-conditioned class margins. Only label-token log probabilities
contribute gradients; fixed <think>/<answer> formatting contributes no loss.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path

LABELS = ("FLAIR", "contrast-enhanced T1")
MODALITIES = ("flair", "t1ce")
ID_PATTERN = re.compile(r"^(?P<patient>.+)_z(?P<level>\d{3})_(?P<modality>flair|t1ce)_sequence$")
VERSION = "paired-image-label-margin-v2"
CLASS_WEIGHT = 1.0
FUTURE_SENTINEL_TOKEN_ID = 0  # After complete target; never included in scored tokens.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def patient_pairs(rows, *, levels=(45, 91, 136)):
    """Only real, strictly paired train MRI. Never choose by model outcomes."""
    levels = tuple(levels)
    require(len(levels) == len(set(levels)) and all(type(x) is int and 0 <= x <= 999 for x in levels),
            "Unique nonnegative 3-digit axial levels required")
    by_pair = {}
    for row in rows:
        if row.get("ability") != "mri_sequence":
            continue
        info = row["extra_info"]
        require(row["data_source"] == "OpenNeuro_ds007045" and
                info.get("internal_split") == "train" and
                info.get("patient_lineage") == "BIDS_SUBJECT_VERIFIED", "Only verified training sequence rows")
        match = ID_PATTERN.fullmatch(row["id"])
        require(match is not None, "Unrecognized source MRI sequence ID")
        patient = match.group("patient")
        level = int(match.group("level"))
        modality = match.group("modality")
        require(patient == info["patient_key"], "MRI source patient mismatch")
        require(row["reward_model"]["ground_truth"] == LABELS[MODALITIES.index(modality)],
                "MRI image/target label mismatch")
        require(len(row["images"]) == 1 and Path(row["images"][0]).is_file(),
                "Exactly one existing MRI image required")
        require(Path(row["images"][0]).name == row["id"].removesuffix("_sequence") + ".png",
                "MRI image filename does not match frozen image identity")
        require(row["prompt"] == [{"role": "user", "content": [
                {"type": "image", "image": row["images"][0]},
                {"type": "text", "text": info["question"]}]}],
                "Model must not see hidden label or source metadata")
        if level not in levels:
            continue
        key = (patient, level)
        slot = by_pair.setdefault(key, {})
        require(modality not in slot, "Duplicate MRI modality in a pair")
        slot[modality] = row
    require(by_pair, "No image pairs found in the training split")
    out = []
    for (patient, level), members in sorted(by_pair.items()):
        require(set(members) == set(MODALITIES), "Incomplete same-patient, same-slice pair")
        f, t = members[MODALITIES[0]], members[MODALITIES[1]]
        require(f["extra_info"]["question"] == t["extra_info"]["question"],
                "Pair question differs between MRI modalities")
        out.append({"patient_key": patient, "level": level,
                    "FLAIR": f, "contrast-enhanced T1": t})
    return out


def schedule_pairs(pairs, *, seed=7, steps=2):
    """Patient-diverse fixed schedule; levels rotate in subsequent epochs."""
    require(type(seed) is int and seed >= 0 and type(steps) is int and 1 <= steps <= 64,
            "Finite reviewed seed and step count required")
    grouped = {}
    for pair in pairs:
        person, level = pair["patient_key"], pair["level"]
        require(level not in grouped.setdefault(person, {}), "Repeated patient/level pair")
        grouped[person][level] = pair
    require(grouped, "No patients available")
    order = sorted(grouped, key=lambda p: hashlib.sha256(f"{seed}:paired-v2:{p}".encode()).hexdigest())
    result = []
    epoch = 0
    while len(result) < steps:
        for person in order:
            levels = sorted(grouped[person])
            entry = grouped[person][levels[epoch % len(levels)]]
            index = len(result)
            pair_digest = hashlib.sha256(f"{person}:{entry['level']}".encode()).hexdigest()[:12]
            result.append({"id": f"paired-v2-{index:04d}-{pair_digest}",
                           "patient_key": person, "level": entry["level"],
                           "FLAIR": entry["FLAIR"], "contrast-enhanced T1": entry["contrast-enhanced T1"]})
            if len(result) == steps:
                break
        epoch += 1
    return result


def _softplus(x):
    return math.log1p(math.exp(-abs(x))) + max(x, 0.)


def _sigmoid(x):
    return 1. / (1. + math.exp(-x)) if x >= 0 else math.exp(x) / (1. + math.exp(x))


def margin_objective(scores, *, class_weight=CLASS_WEIGHT):
    """Differentiable loss + exact dL/d(score) for four sequential MLX backwards.

    scores[(true_image_label, candidate_label)] is the mean LABEL-token logp.
    L = softplus(-(m_T1-m_FLAIR)) + w/2*(softplus(m_FLAIR)+softplus(-m_T1)).
    The first term cannot be optimized by an image-INDEPENDENT label prior alone.
    """
    require(set(scores) == {(im, ans) for im in LABELS for ans in LABELS},
            "Exactly four image/candidate scores required")
    require(all(isinstance(x, (int, float)) and math.isfinite(x) and x <= 1e-5 for x in scores.values()),
            "Finite log probabilities <= 0 required")
    require(math.isfinite(class_weight) and class_weight >= 0, "Invalid class loss coefficient")
    f, t = LABELS
    m_f = scores[(f, t)] - scores[(f, f)]
    m_t = scores[(t, t)] - scores[(t, f)]
    difference = m_t - m_f
    loss = _softplus(-difference) + class_weight * .5 * (_softplus(m_f) + _softplus(-m_t))
    df = _sigmoid(-difference) + .5 * class_weight * _sigmoid(m_f)
    dt = -_sigmoid(-difference) - .5 * class_weight * _sigmoid(-m_t)
    coeffs = {(f, f): -df, (f, t): df, (t, f): -dt, (t, t): dt}
    require(math.isfinite(loss) and all(math.isfinite(x) for x in coeffs.values()), "Nonfinite objective")
    return {"loss": loss, "flair_margin": m_f, "t1_margin": m_t,
            "paired_separation": difference, "gradient_coefficients": coeffs,
            "flair_correct": m_f < 0, "t1_correct": m_t > 0}


def native_backend_class():
    """Reuse proven persistent Adam/base-cache loader; import MLX only at launch."""
    from mri_rl.mlx_continuous_gspo import NativeGSPOBackend
    from mri_rl.answer_likelihood import partition_target

    class NativePairedAccuracyBackend(NativeGSPOBackend):
        def paired_step(self, pair, expected_image_hashes):
            mx, nn = self.mx, self.nn
            self.model.eval()
            require(all(sha(pair[label]["images"][0]) == expected_image_hashes[label]
                        for label in LABELS), "Frozen paired MRI images changed")

            def label_score(model, prompt_ids, target_ids, image_pixels, kwargs, first, count):
                ids = mx.concatenate((prompt_ids, target_ids[None, :]), axis=1)
                inputs = ids[:, :-1]
                logits = model(inputs, image_pixels, mx.ones_like(inputs), **kwargs).logits.astype(mx.float32)
                true_labels = ids[:, 1:]
                require(logits.shape[1] == true_labels.shape[1], "Image candidate causal index mismatch")
                logps = nn.log_softmax(logits, axis=-1)
                chosen = mx.take_along_axis(logps, true_labels[..., None], axis=-1)[..., 0]
                return mx.mean(chosen[0, first:first+count])

            vg = nn.value_and_grad(self.model, label_score)
            scores, gradients, prompt_counts, target_counts = {}, {}, {}, {}
            # Separate graph per candidate, so no four-branch 8B graph is retained.
            for image_label in LABELS:
                row = pair[image_label]
                image = row["images"][0]
                self.before_unit("paired_v2_image_prepare")
                require(sha(image) == expected_image_hashes[image_label], "Image changed after resource check")
                messages = [{"role": "system", "content": self.contract.system_instruction},
                            {"role": "user", "content": row["extra_info"]["question"]}]
                prompt = self.template(self.processor, self.model.config, messages,
                                       add_generation_prompt=True, num_images=1,
                                       enable_thinking=self.contract.enable_thinking)
                require(prompt.count("<|image_pad|>") == 1 and prompt.endswith(self.contract.assistant_prefix),
                        "Invalid fixed MRI assistant prompt")
                prepared = self.prepare_inputs(self.processor, images=[image], prompts=prompt,
                    image_token_index=getattr(self.model.config, "image_token_index", None),
                    add_special_tokens=self.special_tokens(self.model.config.model_type, self.processor))
                require(sha(image) == expected_image_hashes[image_label], "Image changed during preparation")
                prompt_ids = prepared["input_ids"]
                P = int(prompt_ids.shape[-1])
                tokenizer = self.processor.tokenizer
                partitions = {label: partition_target(tokenizer, label, prompt_length=P) for label in LABELS}
                # Both class candidates get the same TOTAL target length. Trailing
                # future sentinels do not enter the score or any selected gradient.
                max_target = max(len(item["target_ids"]) for item in partitions.values())
                kwargs = {key: value for key, value in prepared.items()
                          if key not in {"input_ids", "attention_mask", "pixel_values"}}
                for answer_label in LABELS:
                    self.before_unit("paired_v2_candidate_gradient")
                    require(sha(image) == expected_image_hashes[image_label], "Image changed before gradient")
                    part = partitions[answer_label]
                    pad = max_target - len(part["target_ids"])
                    future = part["target_ids"] + [FUTURE_SENTINEL_TOKEN_ID] * pad
                    target_ids = mx.array(future, dtype=prompt_ids.dtype)
                    start = part["label_logit_positions"][0]
                    count = len(part["label_ids"])
                    value, grad = vg(self.model, prompt_ids, target_ids, prepared.get("pixel_values"),
                                     kwargs, start, count)
                    mx.eval(value, grad)
                    score = float(value.item())
                    require(math.isfinite(score) and score <= 1e-5, "Invalid conditional label score")
                    scores[(image_label, answer_label)] = score
                    gradients[(image_label, answer_label)] = self.tree_map(mx.stop_gradient, grad)
                    prompt_counts[(image_label, answer_label)] = P
                    target_counts[(image_label, answer_label)] = len(future)
                    mx.clear_cache()
            objective = margin_objective(scores)
            summed = None
            for key in ((f,a) for f in LABELS for a in LABELS):
                weight = objective["gradient_coefficients"][key]
                scaled = self.tree_map(lambda g: weight*g, gradients[key])
                summed = scaled if summed is None else self.tree_map(lambda a,b: a+b, summed, scaled)
                mx.eval(summed)
            del gradients
            norm = math.sqrt(sum(float(mx.sum(g.astype(mx.float32)**2).item())
                                 for _,g in self.tree_flatten(summed)))
            require(math.isfinite(norm) and norm > 0, "No valid paired MRI gradient")
            self.before_unit("paired_v2_optimizer_update")
            require(all(sha(pair[label]["images"][0]) == expected_image_hashes[label]
                        for label in LABELS), "MRI images changed before update")
            self.optimizer.update(self.model, summed)
            mx.eval(self.model.trainable_parameters(), self.optimizer.state)
            mx.clear_cache()
            return {"objective": {k:v for k,v in objective.items() if k != "gradient_coefficients"},
                    "scores": {f"{image_label}|{answer}": scores[(image_label, answer)]
                               for image_label in LABELS for answer in LABELS},
                    "gradient_l2": norm, "prompt_tokens": list(prompt_counts.values()),
                    "candidate_tokens": list(target_counts.values()), "optimizer_step": int(self.optimizer_state()["step"].item())}

        def score_sequence(self, row, expected_image_sha256):
            """Read-only, shape-matched forced-prefix candidate scores for frozen dev."""
            mx, nn = self.mx, self.nn
            self.model.eval()
            image = row["images"][0]
            self.before_unit("paired_v2_dev_score")
            require(sha(image) == expected_image_sha256, "Development MRI image changed")
            messages = [{"role": "system", "content": self.contract.system_instruction},
                        {"role": "user", "content": row["extra_info"]["question"]}]
            prompt = self.template(self.processor, self.model.config, messages,
                                   add_generation_prompt=True, num_images=1,
                                   enable_thinking=self.contract.enable_thinking)
            require(prompt.count("<|image_pad|>") == 1 and prompt.endswith(self.contract.assistant_prefix),
                    "Development prompt must match paired training")
            prepared = self.prepare_inputs(self.processor, images=[image], prompts=prompt,
                image_token_index=getattr(self.model.config, "image_token_index", None),
                add_special_tokens=self.special_tokens(self.model.config.model_type, self.processor))
            prompt_ids = prepared["input_ids"]
            P = int(prompt_ids.shape[-1])
            parts = {a:partition_target(self.processor.tokenizer,a,prompt_length=P) for a in LABELS}
            max_target = max(len(x["target_ids"]) for x in parts.values())
            kwargs = {k:v for k,v in prepared.items() if k not in {"input_ids","attention_mask","pixel_values"}}
            result = {}
            for label,part in parts.items():
                future=part["target_ids"] + [FUTURE_SENTINEL_TOKEN_ID]*(max_target-len(part["target_ids"]))
                ids = mx.concatenate((prompt_ids, mx.array(future,dtype=prompt_ids.dtype)[None,:]),axis=1)
                inputs = ids[:, :-1]
                logits = self.model(inputs,prepared.get("pixel_values"),mx.ones_like(inputs),**kwargs).logits.astype(mx.float32)
                chosen = mx.take_along_axis(nn.log_softmax(logits,axis=-1),ids[:,1:,None],axis=-1)[...,0]
                indices = part["label_logit_positions"]
                value = mx.mean(chosen[0,indices[0]:indices[-1]+1])
                mx.eval(value)
                score = float(value.item())
                require(math.isfinite(score) and score <= 1e-5, "Invalid development candidate score")
                result[label] = score
                mx.clear_cache()
            require(sha(image) == expected_image_sha256, "Development image changed during scoring")
            margin = result[LABELS[1]] - result[LABELS[0]]
            prediction = LABELS[1] if margin>0 else LABELS[0] if margin<0 else "TIE"
            return {"record_id":row["id"],"gold":row["reward_model"]["ground_truth"],
                    "forced_prefix_candidate_logp_mean":result,"t1_minus_flair_margin":margin,
                    "forced_prefix_prediction":prediction,"forced_prefix_correct":prediction==row["reward_model"]["ground_truth"],
                    "statistical_patient_key":row["extra_info"]["patient_key"]}

    return NativePairedAccuracyBackend
