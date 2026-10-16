"""Synthetic likelihood receipts only; never MRI/model results."""
import copy
import importlib.util
from pathlib import Path
import pytest
from mri_rl.answer_likelihood import LABELS,PREFIX,SUFFIX,partition_target,summarize_tokens,compare_candidates


def module():
    path=Path(__file__).parents[1]/'scripts/analyze_answer_likelihood.py'
    assert path.exists(),'Likelihood aggregate analyzer missing'
    spec=importlib.util.spec_from_file_location('likelihood_analysis',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.fixture
def fixture():
    class Tok:
        def encode(self,text,add_special_tokens=False):return list(text.encode())
    records=[]
    for i in range(2):
        for j,label in enumerate(LABELS):
            records.append({'record_id':f'private-person-{i}-{j}','patient_key':f'private-person-{i}',
              'level':45,'visible_label':label,'image':f'/private/pixel-{i}-{j}.png','image_sha256':f'pixel-{i}-{j}',
              'prompt':['same-question',f'image-{i}-{j}'],'direct_A_sequence_exposure':j==0,'direct_B_sequence_exposure':i==j})
    records.insert(2,{'record_id':'no_image_unique','patient_key':None,'level':None,'visible_label':None,
       'image':None,'image_sha256':None,'prompt':['same-question'],'direct_A_sequence_exposure':False,'direct_B_sequence_exposure':False})
    for record in records:
        content=[] if record['image'] is None else [{'type':'image','image':record['image']}]
        record['prompt']=[{'role':'system','content':'Synthetic fixture instruction'},
            {'role':'user','content':content+[{'type':'text','text':'Synthetic fixture question'}]}]
    c={'patients':2,'level':45,'records':records,'version':'training_answer_likelihood_v1','labels':list(LABELS),
       'fixed_prefix':PREFIX,'fixed_suffix':SUFFIX,'weight_updates':0,'heldout_used':False,
       'checkpoints':{k:{'sha256':k+'-adapter'} for k in ['A0','A28','B28']},'generation':{'max_tokens':256,'temperature':0}}
    outputs={}
    for model in ['A0','A28','B28']:
        rows=[]
        for r in records:
            is_t1=r['visible_label']==LABELS[1]
            # Image dependence improves for B only on T1 images, with unchanged greedy text.
            margin=(-.2 if not is_t1 else .4)+(0.3 if model=='B28' and is_t1 else 0.)
            if r['image'] is None:margin=-.8
            values=[-2.,-2.+margin];scores={};promptids=[5,151655,6] if r['image'] else [5,6]
            for label,value in zip(LABELS,values):
                part=partition_target(Tok(),label,prompt_length=len(promptids))
                lp=[-.2]*len(part['prefix_ids'])+[value]*len(part['label_ids'])+[-.4]*len(part['suffix_ids'])
                scores[label]={**summarize_tokens(lp,part),'partition':part,'target_token_logps':lp,'first_label_candidate_logps':values}
            rows.append({**r,'checkpoint':model,'adapter_sha256':model+'-adapter','run_identity':model+'-identity',
              'prompt_ids':promptids,'prepared_metadata':{'pixels':r['image_sha256']},'rendered_prompt_sha256':str(r['prompt']),
              'image_token_count':int(r['image'] is not None),'add_special_tokens':False,
              'candidate_scores':scores,'comparisons':compare_candidates(scores),'prefix_oracle_first_logps':values,'prefix_max_disagreement':0.,
              'response':'<think>Fixture.</think><answer>\\boxed{FLAIR}</answer>',
              'generation_tokens':16,'prompt_tokens':len(promptids),'total_tokens':len(promptids)+16,
              'finish_reason':None,'truncated':None,'weight_updates':0})
        outputs[model]=rows
    return c,outputs,{k:k+'-identity' for k in outputs}


def test_true_margins_pair_separation_noimage_and_paired_change(fixture):
    m=module();c,o,identities=fixture
    public,private=m.analyze(c,o,identities)
    b=public['checkpoints']['B28'];a=public['checkpoints']['A28']
    assert b['images']['per_class'][LABELS[1]]['mean']['margin']['mean']==pytest.approx(.7)
    assert b['images']['mean']['signed_margin']['mean']==pytest.approx(.45)
    assert b['pairs']['mean']['separation']['mean']==pytest.approx(.9)
    assert b['no_image']['unique_inputs']==1
    assert b['images']['greedy']['strict_correct']==a['images']['greedy']['strict_correct']==2
    assert public['paired_changes']['B28_minus_A28']['mean']['signed_margin_delta']['mean']==pytest.approx(.15)
    assert public['paired_changes']['B28_minus_A28']['mean']['greedy_unchanged_but_margin_changed']==2
    assert len(private['pairs'])==2


@pytest.mark.parametrize('damage',['missing','duplicate','identity','image','prompt','numeric'])
def test_invalid_or_incomplete_inputs_fail_closed(fixture,damage):
    m=module();c,o,identities=fixture
    if damage=='missing':o['B28'].pop()
    if damage=='duplicate':o['B28'][-1]=copy.deepcopy(o['B28'][0])
    if damage=='identity':o['B28'][0]['run_identity']='wrong'
    if damage=='image':o['B28'][0]['image_sha256']='wrong'
    if damage=='prompt':o['B28'][0]['prompt_ids'][0]=7
    if damage=='numeric':o['B28'][0]['candidate_scores'][LABELS[0]]['label_logp_mean']=float('nan')
    with pytest.raises(ValueError):m.analyze(c,o,identities)


def test_swaps_follow_displayed_truth_and_are_bijective_derivations(fixture):
    m=module();c,o,identities=fixture
    public,private=m.analyze(c,o,identities)
    swap=public['checkpoints']['B28']['same_patient_swap']
    assert swap['new_forwards']==0 and swap['new_patients']==0
    assert swap['mean']['correct_direction_flips']==4 and swap['mean']['both_correct_pairs']==2
    assert swap['greedy']['correct_direction_flips']==0
    mapping=private['mappings']['other_patient_opposite']
    assert len({x['replacement_record_id'] for x in mapping})==4
    assert all(x['origin_patient']!=x['replacement_patient'] and x['origin_label']!=x['replacement_label'] for x in mapping)


def test_invalid_greedy_has_own_bucket_and_unknown_finish_not_stop(fixture):
    m=module();c,o,identities=fixture
    o['B28'][0]['response']='not a strict answer'
    public,_=m.analyze(c,o,identities)
    greedy=public['checkpoints']['B28']['images']['greedy']
    assert greedy['confusion'][LABELS[0]]['INVALID_OR_UNPARSED']==1
    assert greedy['full_format']==3 and greedy['strict_correct']==1
    assert greedy['termination']=={'unknown':4,'stop':0,'length':0}


def test_public_whitelist_contains_no_record_patient_path_or_response(fixture):
    import json
    m=module();c,o,identities=fixture
    public,private=m.analyze(c,o,identities)
    text=json.dumps(public)
    assert 'private-person' not in text and '/private/' not in text and '<think>' not in text
    assert 'private-person' in json.dumps(private)


def test_partial_replacement_preparation_is_still_matched(fixture):
    m=module();c,o,ids=fixture
    o['B28'][-1]['prepared_metadata']={'different':'pixels'}
    with pytest.raises(ValueError,match='configuration'):
        m.analyze(c,o,ids,pair_count=1)


def test_patient_means_use_patient_keys_not_adjacent_rows(fixture):
    m=module();c,o,ids=fixture
    row=o['B28'][-1];target=row['candidate_scores'][LABELS[1]];part=target['partition']
    n=len(part['prefix_ids']);a=len(part['label_ids'])
    target['target_token_logps'][n:n+a]=[-.7]*a
    target.update(summarize_tokens(target['target_token_logps'],part))
    row['prefix_oracle_first_logps'][1]=-.7
    for label in LABELS:row['candidate_scores'][label]['first_label_candidate_logps'][1]=-.7
    row['comparisons']=compare_candidates(row['candidate_scores'])
    order=[0,3,2,1,4]
    c['records']=[c['records'][i] for i in order]
    for model in o:o[model]=[o[model][i] for i in order]
    public,_=m.analyze(c,o,ids)
    result=public['paired_changes']['B28_minus_A28']['mean']['within_patient_mean_signed_delta']
    assert result['min']==pytest.approx(.15) and result['max']==pytest.approx(.45)
    assert result['sd_descriptive']==pytest.approx(.15)


def scoped_files(fixture,tmp_path,profile=None):
    import json
    from mri_rl.experiment_contract import canonical_hash
    m=module();c,outputs,_=fixture
    contract=tmp_path/'contract.json';contract.write_text(json.dumps(c))
    paths={};scopes={}
    for model in outputs:
        path=tmp_path/(model+'.jsonl');scopepath=tmp_path/(model+'-scope.json')
        s={'checkpoint':model,'contract_sha256':m.sha(contract),'adapter':c['checkpoints'][model],
           'output':str(path),'source_hashes':{'fixture':'hash'},'backend_sources':{'fixture':'hash'},
           'temperature':0,'max_generation_tokens':256,'lora_rank':4,'lora_alpha':8,'quantization_bits':4,'group_size':64,'battery_exception':None}
        s.update(profile or {})
        s['run_identity']=canonical_hash({'contract':m.sha(contract),'checkpoint':model,'sources':s['source_hashes'],'backend':s['backend_sources'],'battery_diagnostic':False})
        for row in outputs[model]:row['run_identity']=s['run_identity']
        path.write_text(''.join(json.dumps(row)+'\n' for row in outputs[model]));scopepath.write_text(json.dumps(s))
        paths[model]=path;scopes[model]=scopepath
    return m,contract,paths,scopes


def test_matching_but_wrong_profiles_rejected(fixture,tmp_path):
    m,c,paths,scopes=scoped_files(fixture,tmp_path,{'temperature':1,'max_generation_tokens':512,
        'lora_rank':8,'lora_alpha':16,'quantization_bits':8,'group_size':128})
    with pytest.raises(ValueError,match='profile'):m.load_scoped(c,paths,scopes)


def test_output_mutation_during_validation_rejected(fixture,tmp_path,monkeypatch):
    m,c,paths,scopes=scoped_files(fixture,tmp_path)
    original=m.completed_prefix
    def changing(path,*args,**kwargs):
        result=original(path,*args,**kwargs)
        path.write_bytes(path.read_bytes().replace(b'\n',b' \n'))
        return result
    monkeypatch.setattr(m,'completed_prefix',changing)
    with pytest.raises(ValueError,match='changed'):m.load_scoped(c,paths,scopes)


def test_derived_controls_reject_different_questions(fixture):
    m=module();c,o,ids=fixture
    c['records'][-1]['prompt'][1]['content'][-1]['text']='Different question'
    for model in o:o[model][-1]['prompt']=c['records'][-1]['prompt']
    with pytest.raises(ValueError,match='question|prompt'):m.analyze(c,o,ids)
