"""CPU schedule tests on synthetic rows. Not MRI evidence, not learning."""
import copy
import pytest
from mri_rl import exposure_rounds_v1 as ex

SITES = ['BA', 'BO', 'FO', 'JS', 'KA', 'ME', 'NSK', 'TO']


def pool56():
    rows, sites = [], {}
    for i in range(28):
        person = f'sub-{i:02d}'
        sites[person] = SITES[i % 8]
        for label, tag in (('FLAIR', 'f'), ('contrast-enhanced T1', 't')):
            rows.append({'id': f'{person}_{tag}', 'ability': 'mri_sequence', 'images': [f'/x/{person}{tag}.png'],
                         'extra_info': {'patient_key': person, 'internal_split': 'train', 'question': 'q'}, 'reward_model': {'ground_truth': label}})
    return rows, sites


def dallas48():
    return [{'id': f'd{i:02d}', 'ability': 'mri_visit_correspondence', 'images': [f'/x/d{i}.png'],
             'extra_info': {'internal_split': 'train', 'question': 'dq'}, 'reward_model': {'ground_truth': 'AB'[i % 2]}} for i in range(48)]


def plan(rounds=8):
    rows, sites = pool56()
    return ex.build_round_plan(rows, dallas48(), sites, rounds=rounds)


def test_every_selected_patient_gets_both_modalities_inside_one_atomic_block():
    p = plan(8)
    assert len(p['rounds']) == 8
    for r in p['rounds']:
        seq = r['blocks'][0]
        assert seq['kind'] == 'SEQ_PAIR' and len(seq['rows']) == 2
        assert {row['reward_model']['ground_truth'] for row in seq['rows']} == {'FLAIR', 'contrast-enhanced T1'}
        assert len({row['extra_info']['patient_key'] for row in seq['rows']}) == 1
    report = ex.exposure_report(p)
    assert report['patients_exposed'] == 8 and report['patients_with_both_modalities'] == 8 and report['patients_with_one_modality'] == 0
    assert report['class_rows'] == {'FLAIR': 8, 'contrast-enhanced T1': 8} and report['sequence_rows_repeated'] == 0


def test_task_balance_is_explicit_two_sequence_groups_and_two_dallas_groups_per_round():
    p = plan(8)
    report = ex.exposure_report(p)
    assert report['k4_groups'] == {'mri_sequence': 16, 'mri_visit_correspondence': 16}
    assert report['task_share'] == {'mri_sequence': .5, 'mri_visit_correspondence': .5}
    assert report['dallas_rows_unique'] == 16 and report['dallas_class_rows'] == {'A': 8, 'B': 8}


def test_plan_is_deterministic_and_identical_for_both_arms():
    assert plan(8)['plan_sha256'] == plan(8)['plan_sha256']
    a, b = ex.arm_view(plan(8), 'A'), ex.arm_view(plan(8), 'B')
    assert a['plan_sha256'] == b['plan_sha256'] and a['grouping'] == 'same_prompt' and b['grouping'] == 'paired_prompt'
    assert a['rounds'] == b['rounds']          # only the grouping label differs


def test_update_cap_truncates_whole_rounds_never_a_pair():
    p = plan(8)
    cut = ex.truncate_to_update_cap(p, max_updates=5)       # 2 updates per round -> 2 whole rounds
    assert len(cut['rounds']) == 2 and ex.exposure_report(cut)['patients_with_one_modality'] == 0
    with pytest.raises(ValueError):
        ex.truncate_to_update_cap(p, max_updates=1)         # would split the first round: refused, not silently half-run


def test_prefix_is_never_reported_as_all56_trained():
    r = ex.exposure_report(plan(8))
    assert r['all_56_rows_exposed'] is False and r['coverage_claim'] == 'PREFIX_8_OF_28_PATIENTS'
    full = ex.exposure_report(plan(28))
    assert full['all_56_rows_exposed'] is True and full['coverage_claim'] == 'ALL_28_PATIENTS_56_ROWS'


def test_original_sampler_comparison_reports_the_historical_pair_deficit():
    rows, _ = pool56()
    historical = []                                         # one modality per person: the old 28-row pool behaviour
    for row in rows[::2][:16]:
        historical.append(row)
    cmp = ex.compare_with_original(ex.exposure_report(plan(8)), ex.original_pool_exposure(historical))
    assert cmp['original']['patients_with_both_modalities'] == 0 and cmp['new']['patients_with_both_modalities'] == 8


def test_site_coverage_and_duplicate_guards():
    rows, sites = pool56()
    bad = copy.deepcopy(rows)
    bad[1]['reward_model']['ground_truth'] = 'FLAIR'        # person without a T1
    with pytest.raises(ValueError):
        ex.build_round_plan(bad, dallas48(), sites, rounds=4)
    with pytest.raises(ValueError):
        ex.build_round_plan(rows, dallas48(), sites, rounds=29)
    assert len(ex.exposure_report(plan(8))['sites_covered']) >= 7


def test_token_reservation_is_explicit_and_conditional_on_measured_prompt_lengths():
    t = ex.describe_costs(plan(8), prompt_lengths={'mri_sequence': 213, 'mri_visit_correspondence': 405})
    # 16 seq groups x4 x(213+256) + 16 Dallas groups x4 x(405+256)
    assert t['generation_charged_tokens'] == 16 * 4 * (213 + 256) + 16 * 4 * (405 + 256)
    assert t['max_returned_response_tokens'] == 32 * 4 * 256 and t['updates_max_per_arm'] == 16
