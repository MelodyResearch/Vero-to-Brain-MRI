"""CPU-only closed-B2 arithmetic analysis; no native launch entry point.

Consumes the original closed diagnostic, writes aggregates/hashes only, and
does not emit saved token IDs, patient paths or tokenwise likelihood vectors.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W/'src'))
from mri_rl.rollout_precision_v2 import cpu_bf16_normalization_counterexample


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def hook_source_evidence(path):
    """Require the inspected ordinary-AR hook/normalization order in real source."""
    tree = ast.parse(Path(path).read_text())
    generate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_step')
    step = next(n for n in generate.body if isinstance(n, ast.FunctionDef) and n.name == '_step')
    appends = [n for n in ast.walk(generate) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and isinstance(n.func.value, ast.Name) and n.func.value.id == 'processors' and n.func.attr == 'extend'
               and len(n.args) == 1 and isinstance(n.args[0], ast.Name) and n.args[0].id == 'logits_processors']
    require(len(appends) == 1, 'Installed ordinary AR does not append caller logits_processors as inspected')
    body = next(n for n in step.body if isinstance(n, ast.With)).body
    hook = next(n for n in body if isinstance(n, ast.If) and any(isinstance(x, ast.For) and isinstance(x.target, ast.Name)
                and x.target.id == 'processor' for x in n.body))
    loop = next(n for n in hook.body if isinstance(n, ast.For))
    assigned = next(n for n in loop.body if isinstance(n, ast.Assign))
    require(ast.unparse(assigned) == 'logits = processor(tokens, logits)', 'Processor return no longer directly replaces logits')
    normalization = next(n for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'logprobs' for t in n.targets))
    require(ast.unparse(normalization.value) == 'logits - mx.logsumexp(logits, axis=-1, keepdims=True)', 'Installed AR normalization changed')
    require(body.index(hook) < body.index(normalization), 'Hook is after normalization')
    between = body[body.index(hook)+1:body.index(normalization)]
    require(not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == 'logits'
                    for statement in between for n in ast.walk(statement)), 'Logits reassigned after upcast hook')
    return {'source_sha256': sha(path), 'custom_processor_append_line': appends[0].lineno,
            'processor_return_line': assigned.lineno, 'normalization_line': normalization.lineno,
            'processor_guard': ast.unparse(hook.test),
            'verdict': 'SOURCE_SUPPORTS_FINAL_UPCAST_BEFORE_ORDINARY_AR_NORMALIZATION',
            'native_hook_execution': 'NOT_RUN',
            'scope': 'Ordinary generate_step._step with nonempty y and upcast final in custom processors; no batch/speculative/fused-path guarantee.'}


def difference(left, right):
    require(len(left) == len(right) > 0 and all(math.isfinite(x) for x in [*left, *right]), 'Saved likelihood vectors invalid')
    delta = [abs(a-b) for a, b in zip(left, right)]
    return {'tokens': len(left), 'max_abs_delta': max(delta), 'mean_abs_delta': math.fsum(delta)/len(delta)}


def analyze(record_path, expected_sha, scope_path):
    require(sha(record_path) == expected_sha, 'Exact closed B2 record digest required')
    scope = load(scope_path); record = load(record_path)
    require(scope['schema'] == 'native-gspo-path-probe-v1' and scope['budget']['optimizer_updates'] == 0, 'Unexpected diagnostic scope')
    run = Path(record_path).parent.parent
    require(str(run.resolve()) == scope['output_directory'], 'Record is not from bound v3 output')
    for path, expected in scope['source_sha256'].items(): require(sha(path) == expected, 'Frozen executed source changed')
    ar = next(Path(path) for path in scope['source_sha256'] if path.endswith('/mlx_vlm/generate/ar.py'))
    integrity = load(run/'all_parameter_integrity_SAFE.json'); release = load(run/'release_SAFE.json')
    group_integrity = load(Path(record_path).parent/'exit_integrity_PRIVATE.json'); stopped = load(run/'STOP_PRIVATE.json')
    require(integrity['all_parameters_unchanged'] and integrity['tensor_count'] == 1942 and
            group_integrity['weights_unchanged'] and group_integrity['optimizer_Python_MLX_RNG_unchanged'] and
            group_integrity['weights_before'] == group_integrity['weights_after'] and group_integrity['state_before'] == group_integrity['state_after'],
            'Closed all-model/Adam/Python/MLX RNG integrity missing')
    require(release['release_verified'] and release['model_is_none'] and release['active_MLX_bytes'] < 128*1024**2 and
            stopped['completed_groups'] == 0 and stopped['optimizer_updates'] == 0 and
            stopped['error'] == 'B2 numerical parity failed; stop without threshold change', 'Expected closed strict B2 STOP required')
    kv = record['incremental_KV_conditional_logps']; native = kv['native_dtype_selected_logps']; stable = kv['fp32_stable_selected_logps']
    teacher = record['full_teacherforce_conditional_logps']
    native_teacher, stable_teacher, native_stable = difference(native, teacher), difference(stable, teacher), difference(native, stable)
    for actual, key in [(native_teacher, 'comparison'), (stable_teacher, 'stable_KV_vs_full_teacherforce'), (native_stable, 'native_KV_vs_stable_KV')]:
        require(all(abs(actual[k]-record[key][k]) <= 1e-12 for k in ('max_abs_delta', 'mean_abs_delta')), 'Recomputed saved-vector aggregate differs')
    require(all(d == {'logits': 'mlx.core.bfloat16', 'native_logp': 'mlx.core.bfloat16', 'stable_logp': 'mlx.core.float32'}
                for d in kv['per_token_dtypes']), 'Expected observed BF16/F32 split changed')
    hook = hook_source_evidence(ar)
    helper = W/'src/mri_rl/rollout_precision_v2.py'
    tests = W/'tests/test_rollout_precision_v2.py'
    return {'schema': 'rollout-fp32-normalization-fix-v2', 'status': 'CPU_FIX_IMPLEMENTED_AND_TESTED_NATIVE_HOOK_NOT_RUN',
        'research_tag': 'ENGINEERING_UNBLOCK', 'evidence_tier': 'OBSERVED_SAVED_LOGIT_BRANCH_AGGREGATES_AND_CPU_FIXTURE',
        'mission_alignment': {'main_question': 'Qualify policy logprob precision before interpreting Vero MRI GSPO path diagnostics.',
            'claim_falsifier': 'BF16 normalization introduces error beyond the model-logit quantization; float32 arithmetic should reduce the same-logit error. Residual KV/full error falsifies full parity.',
            'isolation': 'Future final logits-processor upcast only; v3 source, checkpoints, weights and completed STOP preserved.',
            'provenance': 'Closed first saved 17-token TRAIN B2 response, same-forward native/F32 branches and original source hashes.',
            'baseline': 'Same-logit BF16 normalization, float32 normalization and separate full teacherforce path.',
            'resources': 'CPU source/NumPy arithmetic only, zero native imports, model loads, generation, backward or optimizer updates.'},
        'source_hook_evidence': hook,
        'closed_diagnostic': {'responses_observed': 1, 'tokens_observed': len(native), 'completed_groups': 0,
            'native_KV_vs_full_TF': native_teacher, 'stable_F32_KV_vs_full_TF': stable_teacher,
            'native_KV_vs_same_logits_stable_F32_KV': native_stable,
            'frozen_max_abs_tolerance': record['comparison']['tolerance'],
            'native_KV_full_gate': 'FAIL', 'stable_KV_full_gate': 'FAIL',
            'all_model_tensors_unchanged': 1942, 'Adam_Python_MLX_RNG_unchanged': True,
            'active_MLX_bytes_after_release': release['active_MLX_bytes'], 'release_verified': True,
            'historical_generation_distribution': record['historical_generation_distribution'],
            'same_logits_F32_branch': 'ACTUAL_SAVED_ARITHMETIC_CONTROL; future-hook native execution NOT_RUN'},
        'cpu_BF16_rounding_counterexample': cpu_bf16_normalization_counterexample(),
        'future_change': {'implementation': 'Append FP32LogitsProcessor last in existing AR logits_processors; cast logits only, leave existing ar normalization and sampling code in place.',
            'native_execution': 'NOT_RUN', 'new_model_forwards': 0, 'new_generations': 0, 'new_backwards': 0, 'new_optimizer_updates': 0},
        'interpretation': 'Observed same-logit F32 arithmetic removes a substantial normalization discrepancy. It does not remove the saved residual KV/full max error, which still exceeds .002; B3 remains unrun.',
        'limitations': ['CPU BF16 round-to-nearest-even model is an arithmetic counterexample, not a bitwise reproduction of an MLX reduction kernel.',
            'FP32 upcast cannot recover precision already lost in model logits; the residual KV/full path mismatch remains unresolved.',
            'No future generation with this hook was executed and no historical rollout distribution was recovered.',
            'No MRI accuracy, grounding, training efficacy, full native parity or scientific generalization claim.'],
        'evidence_sha256': {'closed_B2_record_PRIVATE': sha(record_path), 'executed_v3_scope_PRIVATE': sha(scope_path),
            'original_run_STOP_PRIVATE': sha(run/'STOP_PRIVATE.json'), 'all_parameter_integrity_SAFE': sha(run/'all_parameter_integrity_SAFE.json'),
            'group_exit_integrity_PRIVATE': sha(Path(record_path).parent/'exit_integrity_PRIVATE.json'),
            'original_release_SAFE': sha(run/'release_SAFE.json'), 'executed_v3_helper': scope['source_sha256'][str(W/'src/mri_rl/native_gspo_path_probe_v1.py')],
            'executed_v3_runner': scope['source_sha256'][str(W/'scripts/run_native_gspo_path_probe_v1.py')],
            'new_precision_helper': sha(helper), 'new_CPU_tests': sha(tests), 'analysis_CLI': sha(__file__)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--record-sha256', required=True)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); result = analyze(a.record, a.record_sha256, a.scope)
    require(a.output.resolve().is_relative_to(W/'results'), 'Aggregate results destination required')
    content = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    if a.output.exists(): require(a.output.read_bytes() == content, 'Preserve existing result')
    else: a.output.write_bytes(content)
    print(json.dumps({'status': result['status'], 'aggregate_sha256': sha(a.output), 'native_model_forwards': 0}))


if __name__ == '__main__': main()
