"""V3 paired MRI contrast discrimination: remove image-independent answer bias.

RESEARCH STATUS: new experimental objective; NOT a demonstrated accuracy gain.
This module is isolated from the frozen 18-arm GSPO queue.

For matched same-patient/same-slice images x_F and x_T, define
    m(x) = mean log p(T1 | x, fixed prefix) - mean log p(FLAIR | x, fixed prefix)
    L = 0.5 * (softplus(m(x_F)) + softplus(-m(x_T)))
        + lambda * softplus(m(x_F) - m(x_T)).
The second term is unchanged by any additive shared class prior.  The first
term anchors the absolute binary decision, so this is not a pure-ranking model.

Native memory policy: read-only margins first, then each image's margin
gradient individually with immediate accumulation.  Never retain gradients
or model graphs for both MRI images simultaneously.
"""
from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path

LABELS = ("FLAIR", "contrast-enhanced T1")
CONTRAST_WEIGHT = 1.0
PAIR_IMAGES = 2
STEP_IMAGES = 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sigmoid(value):
    require(math.isfinite(value), "Nonfinite margin")
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return e / (1.0 + e)


def pair_objective(m_flair, m_t1, ops, contrast_weight=CONTRAST_WEIGHT):
    """Differentiable per-pair CE + prior-cancelled softplus ranking loss.

    'ops' is numpy for numerical tests or mlx.core for native execution.
    Neither dev/test data nor any fitted threshold enters this definition.
    """
    require(math.isfinite(float(contrast_weight)) and contrast_weight >= 0,
            "Invalid contrastive loss weight")
    classification = 0.5 * (ops.logaddexp(m_flair, 0.0)
                            + ops.logaddexp(-m_t1, 0.0))
    ranking = ops.logaddexp(m_flair - m_t1, 0.0)
    return classification + contrast_weight * ranking


def margin_coefficients(m_flair, m_t1, contrast_weight=CONTRAST_WEIGHT):
    """Exact scalar partial derivatives dL/dm_F and dL/dm_T.

    These coefficients permit streaming per-image gradient accumulation,
    without holding two 8B vision/language autograd graphs concurrently.
    """
    require(all(math.isfinite(float(v)) for v in (m_flair, m_t1, contrast_weight)),
            "Nonfinite contrastive input")
    require(contrast_weight >= 0, "Negative contrastive weight")
    rank = contrast_weight * sigmoid(m_flair - m_t1)
    return (0.5 * sigmoid(m_flair) + rank,
            -0.5 * sigmoid(-m_t1) - rank)


def validate_step_pairs(examples):
    """Exactly two same-person/same-level FLAIR,T1 pairs, not four labels."""
    require(len(examples) == STEP_IMAGES, "Two MRI pairs per step required")
    patients = []
    for start in (0, 2):
        a, b = examples[start:start + PAIR_IMAGES]
        x, y = a["extra_info"], b["extra_info"]
        require(a["ability"] == b["ability"] == "mri_sequence"
                and x["internal_split"] == y["internal_split"] == "train",
                "Only paired training MRI permitted")
        require(a["reward_model"]["ground_truth"] == LABELS[0]
                and b["reward_model"]["ground_truth"] == LABELS[1],
                "Expected ordered FLAIR,T1 pair")
        require(x["patient_key"] == y["patient_key"] and x["level"] == y["level"]
                and x["question"] == y["question"],
                "Patient/level/question mismatch")
        require(len(a["images"]) == len(b["images"]) == 1
                and a["images"][0] != b["images"][0],
                "Distinct real MRI image paths required")
        patients.append(x["patient_key"])
    # The original V2 pilot may reuse a patient across levels: report that
    # explicitly rather than inventing four independent patients.
    return {"matched_pairs": 2, "distinct_training_people": len(set(patients))}


def native_backend_class(primary):
    """Reuse the V2-certified native constructor; modify ONLY the loss route."""
    from mri_rl import paired_discrimination_pilot_v2 as v2
    from mri_rl import answer_likelihood as al
    NativeBase = v2.native_backend_class(primary)

    class PriorCancelledPairedBackend(NativeBase):
        def _one_image_margin(self, model, prompt_ids, pixels, kwargs, parts):
            mx, nn = self.mx, self.nn
            P = int(prompt_ids.shape[-1])
            scores = []
            for part in parts:
                ids = mx.concatenate((prompt_ids,
                    mx.array(part["target_ids"], dtype=prompt_ids.dtype)[None, :]), axis=1)
                logits = model(ids[:, :-1], pixels, mx.ones_like(ids[:, :-1]),
                               **kwargs).logits
                start = P - 1 + len(part["prefix_ids"])
                n = len(part["label_ids"])
                label_logits = logits[:, start:start + n, :].astype(mx.float32)
                labels = mx.array(part["label_ids"], dtype=prompt_ids.dtype)
                require(int(label_logits.shape[1]) == n,
                        "Candidate label logits do not align")
                logps = nn.log_softmax(label_logits, axis=-1)
                chosen = mx.take_along_axis(logps, labels[None, :, None],
                                            axis=-1)[0, :, 0]
                scores.append(mx.mean(chosen))
            return scores[1] - scores[0]

        def _prepare_one(self, row):
            mx = self.mx
            self.before_unit("prior_cancelled_mri_image")
            require(v2.sha(row["images"][0]) == self.expected_images[row["id"]],
                    "Frozen MRI pixels changed")
            messages = [
                {"role": "system", "content": self.contract.system_instruction},
                {"role": "user", "content": row["extra_info"]["question"]},
            ]
            prompt = self.template(self.processor, self.model.config, messages,
                                   add_generation_prompt=True, num_images=1,
                                   enable_thinking=self.contract.enable_thinking)
            require(prompt.count("<|image_pad|>") == 1
                    and prompt.endswith(self.contract.assistant_prefix),
                    "Native multimodal prompt mismatch")
            prepared = self.prepare_inputs(
                self.processor, images=row["images"], prompts=prompt,
                image_token_index=getattr(self.model.config, "image_token_index", None),
                add_special_tokens=self.special_tokens(
                    self.model.config.model_type, self.processor))
            require(v2.sha(row["images"][0]) == self.expected_images[row["id"]],
                    "MRI changed during processor call")
            ids = prepared["input_ids"]
            P = int(ids.shape[-1])
            require(P == self.pilot_plan["input_tokens"][row["id"]]
                    and bool(mx.all(prepared["attention_mask"] == 1).item()),
                    "Native measured prompt/mask mismatch")
            parts = [al.partition_target(self.processor.tokenizer, label,
                                         prompt_length=P) for label in LABELS]
            grid = prepared["image_grid_thw"].tolist()
            merge = int(self.processor.image_processor.merge_size)
            require(merge > 0 and prepared.get("pixel_values") is not None,
                    "Missing vision features")
            visual = sum(t * h * w // (merge * merge) for t, h, w in grid)
            for label, part in zip(LABELS, parts):
                require(part["target_ids"] == self.expected_target_ids[row["id"]][label],
                        "Target token drift")
                v2.validate_candidate_alignment(
                    self.processor.tokenizer, prompt, ids[0].tolist(),
                    al.PREFIX + label + al.SUFFIX, part,
                    self.model.config.image_token_index, visual)
            kwargs = {k: v for k, v in prepared.items()
                      if k not in {"input_ids", "attention_mask", "pixel_values"}}
            return prepared, ids, parts, kwargs

        def sft_step(self, examples, work):
            report = validate_step_pairs(examples)
            require(len(self.tree_flatten(self.model.trainable_parameters())) == 504,
                    "Unexpected language-LoRA trainable tensor count")
            mx, nn = self.mx, self.nn
            self.model.eval()
            all_margins = []
            prompt_costs, target_costs = [], []
            # Pass 1: measure four margins without retaining model graphs.
            for row in examples:
                prepared, ids, parts, kwargs = self._prepare_one(row)
                margin = mx.stop_gradient(self._one_image_margin(
                    self.model, ids, prepared["pixel_values"], kwargs, parts))
                mx.eval(margin)
                m = float(margin.item())
                require(math.isfinite(m), "Nonfinite frozen-image margin")
                all_margins.append(m)
                prompt_costs.append(2 * int(ids.shape[-1]))
                target_costs.append(sum(len(p["target_ids"]) for p in parts))
                del margin, prepared, ids, parts, kwargs
                gc.collect(); mx.clear_cache()

            coeffs, loss_terms, pair_separations = [], [], []
            for offset in (0, 2):
                mf, mt = all_margins[offset:offset + 2]
                pair_loss = float(pair_objective(mf, mt, math_ops,
                                                 CONTRAST_WEIGHT))
                c_f, c_t = margin_coefficients(mf, mt, CONTRAST_WEIGHT)
                coeffs.extend((c_f / 2.0, c_t / 2.0))  # two-pair mean
                loss_terms.extend((pair_loss / 2.0, pair_loss / 2.0))
                pair_separations.append(mt - mf)

            # Pass 2: one image graph at a time; no simultaneous pair graphs.
            vg = nn.value_and_grad(self.model, self._one_image_margin)
            summed = None
            for row, scalar in zip(examples, coeffs):
                prepared, ids, parts, kwargs = self._prepare_one(row)
                value, grad = vg(self.model, ids, prepared["pixel_values"],
                                 kwargs, parts)
                mx.eval(value, grad)
                require(math.isfinite(float(value.item()))
                        and abs(float(value.item()) - all_margins[examples.index(row)]) <= 1e-4,
                        "Margin changed before optimizer step")
                weighted = self.tree_map(lambda g: g * scalar, grad)
                summed = (weighted if summed is None else
                          self.tree_map(lambda a, b: a + b, summed, weighted))
                mx.eval(summed)
                del prepared, ids, parts, kwargs, value, grad, weighted
                gc.collect(); mx.clear_cache()

            norm = math.sqrt(sum(float(mx.sum(g.astype(mx.float32) ** 2).item())
                                 for _, g in self.tree_flatten(summed)))
            require(math.isfinite(norm) and norm > 0,
                    "No nonzero prior-cancelled pair gradient")
            self.before_unit("prior_cancelled_optimizer_update")
            self.optimizer.update(self.model, summed)
            mx.eval(self.model.trainable_parameters(), self.optimizer.state)
            del summed
            mx.clear_cache()
            return {"loss_terms": loss_terms, "gradient_l2": norm,
                    "prompt_tokens": prompt_costs, "target_tokens": target_costs,
                    "pair_separations_pre_update": pair_separations,
                    "matched_training_pairs": report["matched_pairs"],
                    "distinct_training_people": report["distinct_training_people"]}
    return PriorCancelledPairedBackend


# math.logaddexp was added relatively recently; this wrapper is CPU-only.
class _MathOps:
    @staticmethod
    def logaddexp(a, b):
        x = max(a, b)
        return x + math.log(math.exp(a - x) + math.exp(b - x))


math_ops = _MathOps()


def run_pilot(*, primary, plan, output, adapter, model_path, contract,
              cache, before_unit, authorize, feature_proof,
              feature_proof_sha256, resume=False, max_new_steps=2):
    """Separate two-update V3 run. All V2 model/data admission stays intact.

    Differences from V2: objective/source binding and twice as many native
    passes. Token ledger follows V2's recorded candidate input budget, NOT
    equal physical FLOPs; compare runtime/compute separately.
    """
    from mri_rl import paired_discrimination_pilot_v2 as v2
    sft, al, entry = v2.primary_modules(primary)
    require(v2.sha(adapter) == "b316db3d5eadf2e1fc0cbe80ccc93c7f7405858769f7d32b39a137dea3d6c572"
            and v2.sha(Path(adapter).with_name("adapter_config.json"))
            == "032becabb7cbb6d2af6f3451eb127e5f944551d42eb51eaee3e8d75b839b2f6a",
            "Only common-B28 initializer permitted")
    require(plan["objective_source_sha256"] == v2.sha(v2.__file__)
            and all(v2.sha(Path(primary) / "src/mri_rl" / n) == x
                    for n, x in plan["sources"].items()),
            "Frozen V2 candidate/source plan changed")
    require(plan["input_tokens"] is not None
            and feature_proof and v2.sha(feature_proof) == feature_proof_sha256,
            "Verified native vision feature/input evidence required")
    require(plan["maximum_updates"] == 2 and max_new_steps in (1, 2)
            and callable(authorize) and callable(before_unit),
            "Finite independent research scope and resources required")
    require(all(v2.sha(row["images"][0]) == plan["image_sha256"][row["id"]]
                for row in plan["rows_PRIVATE"]),
            "Frozen training images changed")
    # Preserve common model, cache and prompt bindings from V2's authoritative preflight.
    from mri_rl.experiment_contract import load_contract
    from mri_rl.likelihood_cache import verify_cache_identity
    baseline_path = (Path(primary) /
                     "local_data/c_two_task_pool_v1/u7_visual_controls_v1/probe_plan_PRIVATE.json")
    require(v2.sha(baseline_path)
            == "732260275ca977ccce1b59ea5856701f77a54b0150899acfc17aee2a6484b0ab",
            "Shared checkpoint/prompt reference changed")
    baseline = json.loads(baseline_path.read_text())
    require(str(model_path) == baseline["model_path"]
            and contract.identity()
            == load_contract(Path(baseline["prompt_contract_path"])).identity(),
            "Common model/prompt mismatch")
    certificate = Path(primary) / "results/EXACT_QUANT_CACHE_CERTIFICATE.json"
    cert = json.loads(certificate.read_text())
    artifacts = {str(Path(model_path) / n): v2.sha(Path(model_path) / n)
                 for n in set(cert["non_weight_files_identical"]) | {"config.json"}}
    require(cache and verify_cache_identity(
        cache["dir"], certificate,
        "8fcd2b0bd1122d78153ed8e2114204cdbe7fb095bb7a6273005731f907928160",
        artifacts) == baseline["derived_cache"],
        "Certified base model changed")

    from mri_rl.two_task_training import TokenLedger
    rows = plan["rows_PRIVATE"]
    steps = sft.step_rows(rows, plan["schedule"])
    # The same selected image pairs, native processor, and candidate strings as V2.
    for step in steps:
        validate_step_pairs(step["examples"])
    inputs = lambda r: 2 * plan["input_tokens"][r["id"]]
    targets = plan["target_token_cost"]
    cap = sum(sft.step_cost(step, inputs, targets) for step in steps)
    ledger = TokenLedger({"budget": {"text_visual_token_cap": cap,
                                    "maximum_attempted_groups": 2}})
    Native = native_backend_class(primary)

    def factory(starting, config):
        backend = Native(starting, config, model_path=model_path,
                         contract=contract, seed=7, before_unit=before_unit,
                         cache=cache)
        backend.pilot_plan = plan
        return backend

    queue_lock = (Path(primary) /
                  "local_data/c_two_task_pool_v1/tier_c_v1/remaining20_bundle_v1/remaining20.lock")
    with queue_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return sft.run_sft(
            output=output, steps=steps,
            bindings={
                "objective": "prior_cancelled_mri_pair_v3",
                "objective_source_sha256": v2.sha(__file__),
                "v2_frozen_plan_sha256": v2.identity(plan),
                "feature_proof_sha256": feature_proof_sha256,
                "contrast_weight": CONTRAST_WEIGHT,
                "extra_native_forward_passes_per_example": 2,
            },
            backend_factory=factory, authorize=authorize,
            ledger=ledger, input_tokens=inputs, target_tokens=targets,
            resume=resume, starting_adapter=adapter,
            max_new_steps=max_new_steps, expected_target_ids=plan["target_ids"])
