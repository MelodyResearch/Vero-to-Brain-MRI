import copy
import pytest
from mri_rl import paired_z045_saved_evaluation_v1 as p
from mri_rl.paired_z045_evaluation_v1 import protocol, LABELS


def fixtures():
    plan = protocol()
    plan['records_PRIVATE'] = []
    outputs = []
    for task, people, levels in [('mri_sequence', 4, 3), ('mri_visit_correspondence', 4, 2)]:
        for person in range(people):
            for level in range(levels):
                for truth, label in enumerate(LABELS[task]):
                    key = f'{task}-{person}-{level}-{truth}'
                    extra = {'answer': label, 'patient_key': f'p{person}',
                             'statistical_cluster_key': f'c{person}'}
                    row = {'id': key, 'ability': task, 'extra_info': extra,
                           'reward_model': {'ground_truth': label}}
                    plan['records_PRIVATE'].append({'id': key, 'condition': 'real_image',
                                                    'image_sha256': key + '-pixels', 'row_PRIVATE': row})
    for person in range(4):
        for truth in (0, 1):
            original = f'mri_sequence-{person}-0-{truth}'
            reused = f'mri_sequence-{person}-0-{1-truth}'
            plan['records_PRIVATE'].append({'id': original + '-swap', 'condition': 'same_patient_same_level_opposite_modality',
                'original_real_id_PRIVATE': original, 'reuse_real_id_PRIVATE': reused,
                'image_sha256': reused + '-pixels'})
    for stage in ('parent', 'terminal'):
        out = {'checkpoint_label': stage, 'checkpoint_receipt_sha256': stage,
               'numerical_contract_sha256': 'same-qualified-contract',
               'model_parameters_Adam_unchanged': True, 'RNG_restored': True,
               'rows': [], 'aliases': {}, 'wall_seconds': 100.}
        for ref in plan['records_PRIVATE']:
            if ref['condition'] != 'real_image':
                out['aliases'][ref['id']] = {k: ref[k] for k in (
                    'original_real_id_PRIVATE', 'reuse_real_id_PRIVATE', 'image_sha256')}
                continue
            source = ref['row_PRIVATE'];task = source['ability'];labels = list(LABELS[task])
            truth = labels.index(source['extra_info']['answer'])
            pred = truth if stage == 'terminal' and task == 'mri_sequence' else 0
            scores = [-1. if i == pred else -2. for i in (0, 1)]
            out['rows'].append({'id': ref['id'], 'task': task, 'truth': labels[truth],
                'image_sha256': ref['image_sha256'], 'prompt_sha256': 'same-prompt',
                'greedy': {'text': '<think>MRI contrast</think><answer>\\boxed{' + labels[pred] + '}</answer>', 'generation_tokens': 10},
                'candidates': {'labels': labels, 'per_label': [
                    {'label': labels[i], 'token_count': 1, 'mean': scores[i], 'sum': scores[i],
                     'first': scores[i], 'label_logps_PRIVATE': [scores[i]]} for i in (0, 1)],
                    'prefix_oracle': {'max_abs_error': 0., 'tolerance': .002}},
                'costs': {'greedy_calls': 1, 'candidate_score_API_calls': 3,
                          'generated_tokens': 10, 'teacherforced_input_positions': 300}})
        out['actual_calls'] = {'greedy_calls': 40, 'candidate_score_API_calls': 120,
                              'generated_tokens': 400, 'teacherforced_input_positions': 12000}
        outputs.append(out)
    return plan, *outputs


def compare(plan, parent, child):
    return p.compare_saved(plan, parent, child, {'charged_tokens': 12000, 'seconds': 240., 'updates': 4})


def test_known_balanced_gain_and_aliases_do_not_add_cases():
    x = compare(*fixtures())
    assert x['parent']['mri_sequence']['strict_greedy']['accuracy'] == .5
    assert x['terminal']['mri_sequence']['strict_greedy']['recall'] == [1., 1.]
    assert x['parent']['equal_task_macro_accuracy'] == .5
    assert x['terminal']['equal_task_macro_accuracy'] == .75
    assert x['terminal']['image_swap']['greedy']['both_images_correct'] == 8
    assert x['terminal']['image_swap']['new_inference_calls'] == 0
    assert x['terminal']['mri_sequence']['strict_greedy']['n'] == 24
    assert x['decision']['decision'] == 'ENGINEERING_GO_REQUIRES_NEW_AUTHORIZATION'
    assert x['training_cost']['tokens_per_update'] == 3000
    assert x['generalization_proven'] is False


def test_incomplete_think_counts_invalid_in_denominator_not_salvaged():
    plan, parent, child = fixtures()
    child['rows'][0]['greedy']['text'] = '<think>unfinished'
    x = compare(plan, parent, child)
    seq = x['terminal']['mri_sequence']['strict_greedy']
    assert seq['invalid'] == 1 and seq['n'] == 24
    assert seq['recall'][0] == 11/12
    assert x['decision']['decision'] == 'STOP_SAME_CONFIGURATION'


def test_candidate_tie_invalid_and_no_joint_go():
    plan, parent, child = fixtures()
    for row in child['rows']:
        if row['task'] == 'mri_sequence':
            for score in row['candidates']['per_label']:
                score.update(mean=-1., sum=-1., first=-1., label_logps_PRIVATE=[-1.])
    x = compare(plan, parent, child)
    assert x['terminal']['mri_sequence']['candidate']['mean']['invalid'] == 24
    assert x['decision']['decision'] == 'STOP_SAME_CONFIGURATION'


@pytest.mark.parametrize('mutation', ['duplicate', 'image', 'oracle', 'sum', 'cost', 'alias', 'contract', 'state'])
def test_malformed_or_mismatched_evidence_is_rejected(mutation):
    plan, parent, child = fixtures()
    if mutation == 'duplicate': child['rows'][1] = copy.deepcopy(child['rows'][0])
    if mutation == 'image': child['rows'][0]['image_sha256'] = 'different'
    if mutation == 'oracle': child['rows'][0]['candidates']['prefix_oracle']['max_abs_error'] = .003
    if mutation == 'sum': child['rows'][0]['candidates']['per_label'][0]['sum'] = 0.
    if mutation == 'cost': child['rows'][0]['costs']['teacherforced_input_positions'] = 1276
    if mutation == 'alias': next(iter(child['aliases'].values()))['reuse_real_id_PRIVATE'] = 'new-input'
    if mutation == 'contract': child['numerical_contract_sha256'] = 'BF16-other'
    if mutation == 'state': child['model_parameters_Adam_unchanged'] = False
    with pytest.raises(ValueError): compare(plan, parent, child)


def test_bootstrap_cluster_order_is_independent_of_file_order():
    plan, parent, child = fixtures()
    expected = compare(plan, parent, child)
    parent['rows'].reverse();child['rows'].reverse()
    actual = compare(plan, parent, child)
    assert actual['parent']['mri_sequence']['strict_cluster_uncertainty'] == expected['parent']['mri_sequence']['strict_cluster_uncertainty']
