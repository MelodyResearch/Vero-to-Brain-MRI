"""CPU analyzer counterexamples; no model outputs or synthetic research results."""
import copy
import importlib.util
from pathlib import Path
import pytest


def module():
    p=Path(__file__).parents[1]/'scripts/analyze_c40_visual_probe_results.py'
    assert p.is_file(),'Frozen-probe analyzer missing'
    s=importlib.util.spec_from_file_location('probe_analysis_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def maps(m,prior=0.,image_shift=0.):
    records={};pairs=[]
    for task in m.TASKS:
        suffix='t1_minus_flair' if task=='mri_sequence' else 'same_minus_different'
        def row(key,condition,margin,target=None):
            return {'record_id':key,'source_record_id':key,'category':task,'condition':condition,'visible_target_PRIVATE':target,
                'comparisons':{'mean_'+suffix:margin,'raw_'+suffix:margin},'prefix_oracle_first_logps':[-10.,-10.+margin]}
        labels=m.LABELS[task];records[task+'null']=row(task+'null','no_image',prior)
        for i in range(2):
            neg,pos=task+str(i)+'neg',task+str(i)+'pos'
            records[neg]=row(neg,'real_image',prior-1-image_shift,labels[0]);records[pos]=row(pos,'real_image',prior+1+image_shift,labels[1])
            pairs.append({'task':task,'negative_id':neg,'positive_id':pos,'unit':str(i) if task=='mri_sequence' else 'one_cycle'})
    return records,pairs


def test_common_prior_shift_is_not_image_discrimination_improvement():
    m=module();base,pairs=maps(m);shifted,_=maps(m,prior=2.)
    result,_=m.compare_checkpoints({'B28':base,'U_raw_seed7':shifted},pairs)
    for task in m.TASKS:
        for kind in m.SCORES:
            x=result['U7_minus_B28'][task][kind]
            assert x['delta_no_image_margin']==pytest.approx(2.)
            assert x['delta_unit_averaged_paired_separation']==pytest.approx(0.)
            assert x['delta_unit_averaged_real_minus_no_image_effect']==pytest.approx(0.)


def test_task_relevant_image_separation_change_is_distinguishable():
    m=module();base,pairs=maps(m);changed,_=maps(m,image_shift=.5)
    result,private=m.compare_checkpoints({'B28':base,'U_raw_seed7':changed},pairs)
    for task in m.TASKS:
        x=result['U7_minus_B28'][task]['mean']
        assert x['delta_no_image_margin']==0
        assert x['delta_unit_averaged_paired_separation']==pytest.approx(1.)
        assert result['checkpoints']['B28'][task]['mean']['independent_units']==(2 if task=='mri_sequence' else 1)
    assert private['B28']['images_PRIVATE']


def numeric_fixture():
    from tests.test_c40_visual_probe import numeric,Tok
    from mri_rl.c40_visual_probe import task_comparisons
    m=module();records=[];rows=[]
    for task in m.TASKS:
        for i in range(5):
            r={'record_id':task+str(i),'source_record_id':task+str(i),'category':task,'condition':'no_image' if i==0 else 'real_image',
                'candidate_labels':list(m.LABELS[task]),'image':None if i==0 else 'CPU.png','image_sha256':None if i==0 else 'a'*64,
                'visible_target_PRIVATE':None if i==0 else m.LABELS[task][i%2]}
            records.append(r);x={**numeric(task),**r,'response':'CPU','checkpoint':'B28','adapter_sha256':'b'*64,'run_identity':'frozen',
                'weight_updates':0,'new_greedy_generation':i==0,'candidate_order':list(m.LABELS[task])}
            x['comparisons']=task_comparisons(task,x['comparisons']);rows.append(x)
    scope={'run_identity':'frozen','adapter':{'sha256':'b'*64}}
    return m,records,rows,scope,Tok()


@pytest.mark.parametrize('corruption',['missing','duplicate','identity','record','numeric','route'])
def test_invalid_coverage_identity_source_or_numeric_evidence_rejected(corruption):
    m,records,rows,scope,tok=numeric_fixture()
    assert len(m.validate_records(rows,records,scope,'B28',tok))==10
    bad=copy.deepcopy(rows)
    if corruption=='missing':bad.pop()
    elif corruption=='duplicate':bad[-1]=copy.deepcopy(bad[0])
    elif corruption=='identity':bad[0]['run_identity']='other'
    elif corruption=='record':bad[1]['image_sha256']='f'*64
    elif corruption=='numeric':bad[0]['candidate_scores'][m.LABELS[bad[0]['category']][0]]['label_logp_mean']+=.01
    else:bad[0]['candidate_labels'].reverse()
    with pytest.raises(ValueError):m.validate_records(bad,records,scope,'B28',tok)


def test_external_file_digest_tamper_rejected(tmp_path):
    m=module();p=tmp_path/'out';p.write_text('ORIGINAL');h=m.file_sha(p)
    m.require_hash(p,h,'fixture');p.write_text('CHANGED')
    with pytest.raises(ValueError):m.require_hash(p,h,'fixture')


def test_full_file_authentication_binds_scope_checkpoint_cache_and_identity(tmp_path,monkeypatch):
    import json
    m,records,rows,_,tok=numeric_fixture();sources={'CPU_SOURCE_FIXTURE':'frozen'}
    monkeypatch.setattr(m.probe,'source_hashes',lambda:sources)
    plan={'checkpoints':{'B28':{'sha256':'b'*64}},'derived_cache':{'CPU_CACHE':'fixed'},'records_PRIVATE':records,
        'likelihood':{'prefix':'PREFIX','suffix':'SUFFIX'}}
    output=tmp_path/'completed.jsonl';scopepath=tmp_path/'scope.json'
    scope={'kind':'C40_visual_likelihood_v1','execution_tier':'B','checkpoint':'B28','plan_sha256':m.probe.PLAN_SHA,
        'sources':sources,'adapter':plan['checkpoints']['B28'],'derived_cache':plan['derived_cache'],'weight_updates':0,
        'original21training_ready':True,'output':str(output.resolve()),'fixed_prefix':'PREFIX','fixed_suffix':'SUFFIX',
        'candidate_order':{t:list(m.LABELS[t]) for t in m.TASKS},'prefix_tolerance':.002,'completed_records':0,
        'new_record_ids':[r['record_id'] for r in records],'forwards_per_record':3,'max_new_forwards':30,'existing_output_sha256':None}
    scope['run_identity']=m.canonical_sha({'plan':m.probe.PLAN_SHA,'checkpoint':'B28','adapter':scope['adapter'],'sources':sources})
    for row in rows:row['run_identity']=scope['run_identity']
    output.write_text(''.join(json.dumps(r)+'\n' for r in rows));scopepath.write_text(json.dumps(scope))
    assert len(m.authenticate_output(plan,scopepath,m.file_sha(scopepath),output,m.file_sha(output),'B28',tok))==10
    for changed in ({'adapter':{'sha256':'a'*64}},{'derived_cache':{'CPU_CACHE':'changed'}},{'run_identity':'changed'}):
        scopepath.write_text(json.dumps({**scope,**changed}))
        with pytest.raises(ValueError):m.authenticate_output(plan,scopepath,m.file_sha(scopepath),output,m.file_sha(output),'B28',tok)
