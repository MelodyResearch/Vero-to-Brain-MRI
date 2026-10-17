"""Same-C-data supervised control (Claude-authored) from the common B28 start.

Teacher-forced NLL on the SAME admitted C76 training rows, natural-proportional exposure (epochs of permutations of the
whole pool, so every row is seen equally often), the approved answer/format target only (no invented reasoning), same
native prompt/system instruction as the RL arms, the same persistent Adam profile (1e-5, betas .9/.999, eps 1e-8, no bias
correction, constant LR) and a processed input+target token cap equal to the RL budget B. One update per 4 examples
(gradient averaged, microbatch 1). Atomic boundaries/resume/ledger reuse continuous_rl helpers (adapter, Adam, RNG,
cursor, hash chain). Costs are teacher-forcing costs, NOT generation+backprop FLOPs; reported as measured.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from mri_rl import continuous_rl as cr
from mri_rl.continuous_rl import canonical, digest, require, write_json
from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
from mri_rl.two_task_training import TokenLedger, validate_training_row

EXAMPLES_PER_STEP = 4
TARGET_CAP = 64


def sft_target(row):
    return "<think>Visible MRI evidence.</think><answer>\\boxed{%s}</answer>" % row['reward_model']['ground_truth']


def target_token_counts(rows, tokenizer_json):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_json))
    counts = {r['id']: len(tok.encode(sft_target(r), add_special_tokens=False).ids) for r in rows}
    require(all(0 < n <= TARGET_CAP for n in counts.values()), 'SFT target outside the 1-64 token budget')
    return counts


def build_steps(rows, seed, token_cap, input_tokens, target_tokens, max_steps=256):
    """Deterministic natural-proportional step list; stops when the next whole step cannot fit the cap."""
    for r in rows:
        validate_training_row(r)
    rng = np.random.default_rng(seed)
    queue, steps, used = [], [], 0
    while len(steps) < max_steps:
        while len(queue) < EXAMPLES_PER_STEP:
            queue += [int(i) for i in rng.permutation(len(rows))]
        chunk = [queue.pop(0) for _ in range(EXAMPLES_PER_STEP)]
        cost = sum(input_tokens(rows[i]) + target_tokens[rows[i]['id']] for i in chunk)
        if used + cost > token_cap:
            break
        used += cost
        steps.append(chunk)
    return steps, used


def step_rows(rows, steps):
    out = []
    for n, chunk in enumerate(steps):
        ex = [rows[i] for i in chunk]
        out.append({'id': f'sft-step-{n:04d}', 'ability': 'sft_step', 'data_source': 'C_pool_sft', 'images': [ex[0]['images'][0]],
                    'extra_info': {'reward_type': 'sft_step', 'source_ids': [e['id'] for e in ex]}, 'examples': ex})
    return out


def _validate_step(step):
    require(step.get('ability') == 'sft_step' and len(step['examples']) == EXAMPLES_PER_STEP, 'Malformed SFT step')
    for e in step['examples']:
        validate_training_row(e)


def step_cost(step, input_tokens, target_tokens):
    return sum(input_tokens(e) + target_tokens[e['id']] for e in step['examples'])


def run_sft(*, output, steps, bindings, backend_factory, authorize, ledger, input_tokens, target_tokens,
            resume=False, starting_adapter=None):
    """One resident run over the whole SFT budget; mirrors continuous_rl.run's boundary protocol."""
    require(callable(authorize) and ledger is not None, 'Authorizer and ledger required')
    require(len(steps) <= ledger.plan['budget']['maximum_attempted_groups'] + 58, 'Unexpected SFT step count')
    image_hashes = {e['id']: digest(e['images'][0]) for s in steps for e in s['examples']}
    config = {'profile': 'tier_c_sft_v1', 'examples_per_step': EXAMPLES_PER_STEP, 'target_cap': TARGET_CAP,
              'optimizer': {'name': 'Adam', 'learning_rate': 1e-5, 'betas': [0.9, 0.999], 'eps': 1e-8, 'bias_correction': False}}
    binding = cr._binding(steps, {**bindings, 'example_image_sha256': hashlib.sha256(canonical(image_hashes).encode()).hexdigest(),
                                  'ledger_plan_sha256': hashlib.sha256(canonical(ledger.plan).encode()).hexdigest()},
                          config, _validate_step)
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError('Existing run requires explicit resume; preserve evidence')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another SFT writer is active') from None
        current = cr.inspect_boundaries(output, binding)
        require(not resume or current is not None, 'No complete boundary to resume')
        cursor = current['completed_groups'] if current else 0
        if cursor == len(steps):
            return {'status': 'SCHEDULE_COMPLETE', 'completed_steps': cursor, 'ledger': ledger.state()}
        adapter = current['path'] / 'adapter.safetensors' if current else starting_adapter
        authorize({'phase': 'load', 'binding': binding, 'cursor': cursor})
        backend = backend_factory(adapter, config)
        try:
            if current:
                expected = {'run': binding, 'adapter_sha256': digest(adapter), 'trainable_signature': backend.trainable_signature()}
                state, rng, _ = load_optimizer_checkpoint(current['path'] / 'optimizer', expected_bindings=expected,
                                                          load_arrays=backend.load_arrays)
                backend.restore(state, rng)
                prior = current.get('ledger') or {}
                ledger.used, ledger.charged_groups = prior.get('used_tokens', 0), prior.get('charged_groups', 0)
            else:
                require(cr._step(backend) == 0, 'Fresh run requires initial optimizer')
                work = Path(tempfile.mkdtemp(prefix='.pending-', dir=output))
                current = cr._publish(output, work, backend, binding, 0, 0, None,
                                      {'updated': False, 'record_id': None, 'schedule_record_id': None,
                                       'ledger': ledger.state()})
            stopped = None
            for index in range(cursor, len(steps)):
                step = steps[index]
                if not ledger.can_start_cost(step_cost(step, input_tokens, target_tokens)):
                    stopped = 'TOKEN_BUDGET_EXHAUSTED'; break
                authorize({'phase': 'step', 'cursor': index})
                for e in step['examples']:
                    require(digest(e['images'][0]) == image_hashes[e['id']], 'Frozen image changed before the step')
                before_step = cr._step(backend)
                require(before_step == current['optimizer_updates'], 'Optimizer counter lost continuity')
                work = Path(tempfile.mkdtemp(prefix='.pending-', dir=output))
                before_state = cr.state_fingerprint(backend, backend.optimizer_state(), work / 'optimizer_before.safetensors')
                evidence = backend.sft_step(step['examples'], work)
                prompts, targets = evidence['prompt_tokens'], evidence['target_tokens']
                require(len(prompts) == len(targets) == EXAMPLES_PER_STEP and
                        all(p == input_tokens(e) and 0 < t <= TARGET_CAP and t == target_tokens[e['id']]
                            for p, t, e in zip(prompts, targets, step['examples'])),
                        'Native token counts differ from the frozen measurements')
                norm, losses = evidence['gradient_l2'], evidence['loss_terms']
                require(math.isfinite(norm) and norm > 0 and len(losses) == EXAMPLES_PER_STEP and all(math.isfinite(x) for x in losses),
                        'SFT step lacks finite loss and gradient evidence')
                require(cr._step(backend) == before_step + 1, 'Optimizer step count mismatch')
                write_json(work / 'rollout_PRIVATE.json', {'source_ids': step['extra_info']['source_ids'],
                                                           'prompt_tokens': prompts, 'target_tokens': targets})
                after_state = cr.state_fingerprint(backend, backend.optimizer_state(), work / 'optimizer_after.safetensors')
                check = work / 'verify_adapter.safetensors'; backend.save_adapter(check)
                changed = digest(check) != current['files_sha256']['adapter.safetensors']; check.unlink()
                require(changed, 'SFT step did not change the adapter')
                ledger.used += sum(prompts) + sum(targets); ledger.charged_groups += 1
                current = cr._publish(output, work, backend, binding, index + 1, before_step + 1,
                                      digest(current['path'] / 'receipt.json'),
                                      {'updated': True, 'record_id': step['id'], 'schedule_record_id': step['id'],
                                       'loss_terms': losses, 'gradient_l2': norm,
                                       'optimizer_before_fingerprint': before_state, 'optimizer_after_fingerprint': after_state,
                                       'ledger': ledger.state()})
            status = stopped or ('SCHEDULE_COMPLETE' if current['completed_groups'] == len(steps) else 'BOUNDED_STAGE_COMPLETE')
            return {'status': status, 'completed_steps': current['completed_groups'],
                    'optimizer_updates': current['optimizer_updates'], 'ledger': ledger.state(), 'science_claim': False}
        finally:
            backend.close()


def native_backend_class():
    """Native backend (lazy MLX): reuses the GSPO backend's certified-cache load, adapter attach, Adam and state I/O."""
    from mri_rl.mlx_continuous_gspo import NativeGSPOBackend

    class NativeSFTBackend(NativeGSPOBackend):
        def sft_step(self, examples, work):
            mx, nn = self.mx, self.nn
            def loss_fn(model, prompt_ids, target_ids, pixels, kwargs):
                ids = mx.concatenate((prompt_ids, target_ids[None, :]), axis=1)
                inputs = ids[:, :-1]
                logits = model(inputs, pixels, mx.ones_like(inputs), **kwargs).logits.astype(mx.float32)
                labels = ids[:, 1:]
                require(logits.shape[1] == labels.shape[1], 'SFT logits/labels length mismatch')
                chosen = mx.take_along_axis(nn.log_softmax(logits, axis=-1), labels[..., None], axis=-1)[..., 0]
                return -mx.mean(chosen[0, prompt_ids.shape[-1] - 1:])
            vg = nn.value_and_grad(self.model, loss_fn)
            self.model.eval(); grad_sum = None; losses = []; prompts = []; targets = []
            for row in examples:
                self.before_unit('sft_example')
                messages = [{'role': 'system', 'content': self.contract.system_instruction},
                            {'role': 'user', 'content': row['extra_info']['question']}]
                prompt = self.template(self.processor, self.model.config, messages, add_generation_prompt=True, num_images=1,
                                       enable_thinking=self.contract.enable_thinking)
                require(prompt.count('<|image_pad|>') == 1 and prompt.endswith(self.contract.assistant_prefix),
                        'Native image placeholder/assistant prefix mismatch')
                prepared = self.prepare_inputs(self.processor, images=[row['images'][0]], prompts=prompt,
                    image_token_index=getattr(self.model.config, 'image_token_index', None),
                    add_special_tokens=self.special_tokens(self.model.config.model_type, self.processor))
                prompt_ids = prepared['input_ids']
                target_ids = mx.array(self.processor.tokenizer.encode(sft_target(row), add_special_tokens=False), dtype=prompt_ids.dtype)
                kwargs = {k: v for k, v in prepared.items() if k not in {'input_ids', 'attention_mask', 'pixel_values'}}
                loss, grad = vg(self.model, prompt_ids, target_ids, prepared.get('pixel_values'), kwargs)
                mx.eval(loss, grad)
                losses.append(float(loss.item())); prompts.append(int(prompt_ids.shape[-1])); targets.append(int(target_ids.size))
                scaled = self.tree_map(lambda x: x / len(examples), grad)
                grad_sum = scaled if grad_sum is None else self.tree_map(lambda a, b: a + b, grad_sum, scaled)
                mx.eval(grad_sum); mx.clear_cache()
            norm = math.sqrt(sum(float(mx.sum(g.astype(mx.float32) ** 2).item()) for _, g in self.tree_flatten(grad_sum)))
            self.before_unit('optimizer_update')
            self.optimizer.update(self.model, grad_sum)
            mx.eval(self.model.trainable_parameters(), self.optimizer.state); mx.clear_cache()
            return {'loss_terms': losses, 'gradient_l2': norm, 'prompt_tokens': prompts, 'target_tokens': targets}
    return NativeSFTBackend
