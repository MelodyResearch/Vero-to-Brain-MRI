"""Analyze only externally hash-bound, complete matched F32 saved outputs.

No future completion spec or approval is manufactured by this program. A fresh
independent completion-spec review is required after the actual native exit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from mri_rl.paired_z045_continuation_entry_v1 import scope_digest
from mri_rl.paired_z045_evaluation_v1 import require
from mri_rl.paired_z045_saved_evaluation_v1 import compare_saved

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def authenticated(spec_path, spec_sha, review_path, review_sha):
    require(digest(spec_path) == spec_sha and digest(review_path) == review_sha,
            'External completion-spec/review hash changed')
    spec = json.loads(Path(spec_path).read_text())
    review = json.loads(Path(review_path).read_text())
    require(spec.get('schema') == 'paired-z045-actual-saved-completion-v1' and
            review.get('verdict') in ('PASS', 'PASS_WITH_LIMITATIONS') and
            review.get('completion_spec_sha256') == spec_sha and
            review.get('reviewer_kind') == 'isolated_codex' and review.get('reviewer_id'),
            'Fresh independent actual-completion review required')
    required_sources = ('scripts/analyze_paired_z045_saved_v1.py',
        'src/mri_rl/paired_z045_saved_evaluation_v1.py',
        'src/mri_rl/paired_z045_evaluation_v1.py',
        'src/mri_rl/paired_z045_matched_f32_eval_v1.py',
        'src/mri_rl/vero_mri_reward.py')
    require(all(spec['analysis_source_sha256'].get(name) == digest(ROOT / name)
                for name in required_sources), 'Reviewed analysis/native sources changed')
    loaded = {}
    required = {'evaluation', 'native_scope', 'complete', 'external_exit', 'closed_stdout',
                'parent_release', 'terminal_release', 'parent_integrity', 'terminal_integrity',
                'parent_restored', 'terminal_restored'}
    require(set(spec['inputs']) == required, 'All external witnesses and state audits required')
    for key, bound in spec['inputs'].items():
        require(isinstance(bound.get('sha256'), str) and len(bound['sha256']) == 64 and
                digest(bound['path']) == bound['sha256'], 'Actual input hash mismatch: ' + key)
        if key != 'closed_stdout':
            loaded[key] = json.loads(Path(bound['path']).read_text())
    witness = loaded['external_exit']
    require(witness.get('exit_code') == 0 and type(witness['exit_code']) is int and
            witness.get('closed_stdout_sha256') == spec['inputs']['closed_stdout']['sha256'] and
            witness.get('complete_sha256') == spec['inputs']['complete']['sha256'],
            'Actual exit0 and closed-log/output digests required')
    scope = loaded['native_scope']; complete = loaded['complete']
    require(scope.get('schema') == 'paired-z045-matched-F32-evaluator-v1' and
            complete.get('status') == 'ACTUAL_MATCHED_F32_PARENT_TERMINAL_EVALUATION_COMPLETE' and
            complete.get('scope_sha256') == scope_digest(scope) and complete.get('optimizer_updates') == 0,
            'Exact complete native scope/zero-update witness required')
    require(spec['inputs']['evaluation']['sha256'] == scope['evaluation_sha256'],
            'Evaluation plan differs from actual native scope')
    for name in ('src/mri_rl/paired_z045_matched_f32_eval_v1.py', 'src/mri_rl/vero_mri_reward.py'):
        require(scope['source_sha256'].get(str(ROOT / name)) == digest(ROOT / name),
                'Actual native evaluator/strict parser source changed')
    for stage in ('parent', 'terminal'):
        release = loaded[stage + '_release']; integrity = loaded[stage + '_integrity']
        restored = loaded[stage + '_restored']
        require(release.get('release_verified') is True and release.get('model_is_none') is True and
                type(release.get('active_MLX_bytes')) is int and 0 <= release['active_MLX_bytes'] < 128*1024**2,
                'Actual resource release not verified')
        require(integrity.get('unchanged') is True and integrity['parameters_before'] == integrity['parameters_after'] and
                len(integrity['parameters_before']) == 1942 and integrity['Adam_before'] == integrity['Adam_after'],
                'Full unchanged native parameter/Adam audit required')
        require(restored.get('unchanged') is True and restored['initial'] == restored['restored'],
                'Actual RNG/state restoration audit required')
    outputs = complete['checkpoints']
    require(len(outputs) == 2 and [v['checkpoint_label'] for v in outputs] == ['parent', 'terminal'],
            'Exactly parent and terminal required')
    for result in outputs:
        require(result['checkpoint_receipt_sha256'] == scope[result['checkpoint_label']]['receipt_sha256'] and
                result['numerical_contract_sha256'] == scope['numerical_contract_sha256'],
                'Checkpoint/numerical provenance differs')
    # The independent completion review must bind actual training cost evidence,
    # not an estimate; these aggregate values do not authorize another update.
    cost = spec['actual_training_cost']
    require(review.get('actual_training_cost_verified') is True,
            'Measured training cost requires independent saved-state verification')
    result = compare_saved(loaded['evaluation'], *outputs, training_cost=cost)
    result['authentication'] = {'completion_spec_sha256': spec_sha, 'review_sha256': review_sha,
        'native_complete_sha256': spec['inputs']['complete']['sha256'],
        'native_scope_canonical_sha256': scope_digest(scope)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completion-spec', type=Path, required=True)
    p.add_argument('--expected-spec-sha256', required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--expected-review-sha256', required=True)
    p.add_argument('--aggregate-output', type=Path, required=True)
    a = p.parse_args()
    result = authenticated(a.completion_spec, a.expected_spec_sha256,
                           a.review, a.expected_review_sha256)
    with a.aggregate_output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    print(json.dumps({'status': 'AUTHENTICATED_SAVED_ARITHMETIC_COMPLETE',
                      'aggregate_sha256': digest(a.aggregate_output), 'new_model_loads': 0,
                      'new_optimizer_updates': 0}))


if __name__ == '__main__':
    main()
