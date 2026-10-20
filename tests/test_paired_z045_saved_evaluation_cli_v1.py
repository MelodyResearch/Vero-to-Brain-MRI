import importlib.util
import json
from pathlib import Path
import pytest
from test_paired_z045_saved_evaluation_v1 import fixtures


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/analyze_paired_z045_saved_v1.py'
    spec = importlib.util.spec_from_file_location('saved_cli', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def bundle(tmp_path, mutation=None):
    m = module(); plan, parent, terminal = fixtures()
    def save(name, obj):
        p = tmp_path / (name + '.json'); p.write_text(json.dumps(obj, sort_keys=True))
        return {'path': str(p), 'sha256': m.digest(p)}
    scope = {'schema': 'paired-z045-matched-F32-evaluator-v1',
             'parent': {'receipt_sha256': 'parent'}, 'terminal': {'receipt_sha256': 'terminal'},
             'numerical_contract_sha256': 'same-qualified-contract',
             'source_sha256': {str(m.ROOT / n): m.digest(m.ROOT / n) for n in (
                 'src/mri_rl/paired_z045_matched_f32_eval_v1.py', 'src/mri_rl/vero_mri_reward.py')}}
    if mutation == 'native-parser':
        scope['source_sha256'][str(m.ROOT / 'src/mri_rl/vero_mri_reward.py')] = 'changed'
    inputs = {'evaluation': save('plan', plan)}
    scope['evaluation_sha256'] = inputs['evaluation']['sha256']
    inputs['native_scope'] = save('scope', scope)
    complete = {'status': 'ACTUAL_MATCHED_F32_PARENT_TERMINAL_EVALUATION_COMPLETE',
                'scope_sha256': m.scope_digest(scope), 'optimizer_updates': 0,
                'checkpoints': [parent, terminal]}
    inputs['complete'] = save('complete', complete)
    stdout = tmp_path / 'closed.log'; stdout.write_text('Synthetic fixture, not native evidence.\n')
    inputs['closed_stdout'] = {'path': str(stdout), 'sha256': m.digest(stdout)}
    exit_witness = {'exit_code': 1 if mutation == 'exit' else 0,
        'closed_stdout_sha256': inputs['closed_stdout']['sha256'],
        'complete_sha256': inputs['complete']['sha256']}
    inputs['external_exit'] = save('exit', exit_witness)
    for stage in ('parent', 'terminal'):
        inputs[stage + '_release'] = save(stage + '_release', {
            'release_verified': True, 'model_is_none': True,
            'active_MLX_bytes': 200*1024**2 if mutation == 'release' else 1000})
        before = {str(i): 'synthetic-hash' for i in range(1942)}
        after = dict(before)
        if mutation == 'weights': after['0'] = 'changed'
        inputs[stage + '_integrity'] = save(stage + '_integrity', {
            'unchanged': True, 'parameters_before': before, 'parameters_after': after,
            'Adam_before': {'step': 60}, 'Adam_after': {'step': 60}})
        inputs[stage + '_restored'] = save(stage + '_restored', {
            'unchanged': True, 'initial': {'rng': 'initial'},
            'restored': {'rng': 'different' if mutation == 'rng' else 'initial'}})
    names = ('scripts/analyze_paired_z045_saved_v1.py',
        'src/mri_rl/paired_z045_saved_evaluation_v1.py',
        'src/mri_rl/paired_z045_evaluation_v1.py',
        'src/mri_rl/paired_z045_matched_f32_eval_v1.py',
        'src/mri_rl/vero_mri_reward.py')
    spec = {'schema': 'paired-z045-actual-saved-completion-v1', 'inputs': inputs,
            'analysis_source_sha256': {n: m.digest(m.ROOT / n) for n in names},
            'actual_training_cost': {'charged_tokens': 12000, 'seconds': 240., 'updates': 4}}
    if mutation == 'source': spec['analysis_source_sha256'][names[0]] = 'wrong'
    s = save('spec', spec)
    review = {'verdict': 'PASS_WITH_LIMITATIONS', 'completion_spec_sha256': s['sha256'],
              'reviewer_kind': 'isolated_codex', 'reviewer_id': 'synthetic-reviewer',
              'actual_training_cost_verified': mutation != 'unverified-cost'}
    r = save('review', review)
    return m, s, r


def test_actual_witness_gate_precedes_saved_arithmetic(tmp_path):
    m, s, r = bundle(tmp_path)
    result = m.authenticated(s['path'], s['sha256'], r['path'], r['sha256'])
    assert result['delta']['T1_recall'] == 1.
    assert result['generalization_proven'] is False


@pytest.mark.parametrize('mutation', ['exit', 'release', 'weights', 'rng', 'source', 'unverified-cost', 'native-parser'])
def test_bound_but_invalid_native_witnesses_stop(tmp_path, mutation):
    m, s, r = bundle(tmp_path, mutation)
    with pytest.raises(ValueError):
        m.authenticated(s['path'], s['sha256'], r['path'], r['sha256'])


def test_external_expected_hash_is_not_derived_from_untrusted_spec(tmp_path):
    m, s, r = bundle(tmp_path)
    with pytest.raises(ValueError):
        m.authenticated(s['path'], '0'*64, r['path'], r['sha256'])
