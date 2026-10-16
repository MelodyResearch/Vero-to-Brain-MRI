"""Synthetic fixtures; no MRI forward results."""
import importlib.util
from pathlib import Path
import pytest
from test_analyze_answer_likelihood import fixture


def module():
    p=Path(__file__).parents[1]/'scripts/analyze_single_checkpoint_likelihood.py'
    spec=importlib.util.spec_from_file_location('single_likelihood',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_partial_fixed_pairs_and_trailing_row_are_separate(fixture):
    c,o,ids=fixture;m=module()
    public,private=m.summarize(c,o['A28'][:4],'A28')
    assert public['complete_patient_pairs']==1 and public['trailing_unpaired_images']==1
    assert public['all_saved_images']['images']==3
    assert public['complete_pairs']['images']==2
    assert public['trailing_images']['class_counts']=={'FLAIR':1,'contrast-enhanced T1':0}
    assert public['no_image']['unique_inputs']==1
    assert public['pair_separation']['mean']['mean']==pytest.approx(.6)
    assert public['cross_checkpoint_comparison']=='NOT_RUN'
    assert public['status']=='PARTIAL_FIXED_PREFIX'


def test_missing_noimage_is_explicit_and_no_margin_imputation(fixture):
    c,o,ids=fixture;m=module()
    public,_=m.summarize(c,o['A28'][:2],'A28')
    assert public['no_image']['status']=='NOT_AVAILABLE'
    assert public['complete_pairs']['mean']['signed_real_minus_no_image_margin'] is None


def test_only_first_complete_pairs_never_outcome_selected(fixture):
    c,o,ids=fixture;m=module()
    with pytest.raises(ValueError):m.summarize(c,[o['A28'][0],o['A28'][3]],'A28')
    public,_=m.summarize(c,o['A28'],'A28')
    assert public['status']=='FULL_SINGLE_CHECKPOINT' and public['complete_patient_pairs']==2
    assert public['trailing_unpaired_images']==0


def test_strict_format_and_unknown_termination_preserved(fixture):
    c,o,ids=fixture;m=module();o['A28'][0]['response']='FLAIR'
    public,_=m.summarize(c,o['A28'][:3],'A28')
    assert public['complete_pairs']['greedy']['full_format']==1
    assert public['complete_pairs']['greedy']['strict_correct']==0
    assert public['complete_pairs']['greedy']['termination']=={'unknown':2,'stop':0,'length':0}
    assert public['complete_pairs']['greedy']['confusion']['FLAIR']['INVALID_OR_UNPARSED']==1


def test_public_has_no_identifiers_or_raw_text(fixture):
    import json
    c,o,ids=fixture;m=module();public,private=m.summarize(c,o['A28'],'A28')
    assert 'private-person' not in json.dumps(public) and '/private/' not in json.dumps(public)
    assert 'private-person' in json.dumps(private)

@pytest.mark.parametrize('damage',[None,'identity','profile','checkpoint','output','old_scope','prefix_hash','budget_ids'])
def test_single_scope_loading_checks_provenance(fixture,tmp_path,damage):
    import json
    from test_analyze_answer_likelihood import scoped_files
    _,contract,paths,scopes=scoped_files(fixture,tmp_path)
    m=module()
    raw=paths['A28'].read_bytes();lines=raw.splitlines(keepends=True)
    scope=json.loads(scopes['A28'].read_text())
    scope.update(completed_records=1,max_new_records=4,
        existing_output_sha256=__import__('hashlib').sha256(lines[0]).hexdigest(),
        new_record_ids=[json.loads(line)['record_id'] for line in lines[1:]])
    scopes['A28'].write_text(json.dumps(scope))
    if damage=='identity':
        rows=[json.loads(x) for x in paths['A28'].read_text().splitlines()];rows[0]['run_identity']='bad'
        paths['A28'].write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif damage:
        scope=json.loads(scopes['A28'].read_text())
        if damage=='profile':scope['temperature']=1
        elif damage=='checkpoint':scope['checkpoint']='B28'
        elif damage=='old_scope':scope['new_record_ids']=scope['new_record_ids'][:1]
        elif damage=='prefix_hash':scope['existing_output_sha256']='bad'
        elif damage=='budget_ids':scope['new_record_ids'][0]='not-frozen'
        else:scope['output']=str(tmp_path/'other.jsonl')
        scopes['A28'].write_text(json.dumps(scope))
    if damage:
        with pytest.raises(ValueError):m.load(contract,paths['A28'],scopes['A28'])
    else:
        c,rows,model,evidence=m.load(contract,paths['A28'],scopes['A28'])
        assert model=='A28' and len(rows)==5 and len(evidence)==3
