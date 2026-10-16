import json
import pytest

def numeric_fixture():
    from mri_rl.answer_likelihood import LABELS,partition_target,summarize_tokens,compare_candidates
    class Tok:
        def encode(self,s,add_special_tokens=False):return list(s.encode())
    scores={}
    for label in LABELS:
        p=partition_target(Tok(),label,prompt_length=1);logps=[-1.]*len(p['target_ids'])
        scores[label]={**summarize_tokens(logps,p),'partition':p,'target_token_logps':logps,'first_label_candidate_logps':[-1.,-1.]}
    return {'prompt_ids':[1],'candidate_scores':scores,'comparisons':compare_candidates(scores),
            'prefix_oracle_first_logps':[-1.,-1.],'prefix_max_disagreement':0.}

def test_resume_rejects_changed_identity_and_torn_prefix(tmp_path):
    from mri_rl.paced_answer_likelihood import completed_prefix
    records=[{'record_id':'x','image_sha256':'pixel'}]
    p=tmp_path/'out.jsonl'
    valid={'record_id':'x','image_sha256':'pixel','run_identity':'identity','response':'answer',**numeric_fixture()}
    p.write_text(json.dumps(valid)+'\n')
    assert completed_prefix(p,records,'identity')==1
    with pytest.raises(ValueError):completed_prefix(p,records,'other')
    p.write_text(json.dumps(valid))
    with pytest.raises(ValueError,match='Torn'):completed_prefix(p,records,'identity')

def test_resume_rejects_impossible_numeric_receipt(tmp_path):
    from mri_rl.paced_answer_likelihood import completed_prefix
    p=tmp_path/'bad.jsonl'
    p.write_text(json.dumps({'record_id':'x','image_sha256':'p','run_identity':'i','response':'a',
        'comparisons':{'mean_t1_minus_flair':float('nan')},'candidate_scores':{'FLAIR':{'label_logp_mean':999}}})+'\n')
    with pytest.raises(ValueError,match='numeric'):
        completed_prefix(p,[{'record_id':'x','image_sha256':'p'}],'i')

@pytest.mark.parametrize('field', ['summary','logps','oracle','index'])
def test_numeric_replay_detects_mutated_evidence(field):
    from mri_rl.paced_answer_likelihood import validate_numeric_record
    x=numeric_fixture();validate_numeric_record(x)
    if field=='summary':x['candidate_scores']['FLAIR']['label_logp_mean']+=.1
    if field=='logps':x['candidate_scores']['FLAIR']['target_token_logps'][0]=float('nan')
    if field=='oracle':x['prefix_oracle_first_logps'][0]+=.1
    if field=='index':x['candidate_scores']['FLAIR']['partition']['label_logit_positions'][0]+=1
    with pytest.raises(ValueError,match='numeric'):validate_numeric_record(x)

def test_review_is_bound_to_exact_next_scope(tmp_path):
    from mri_rl.paced_answer_likelihood import verify_review
    from mri_rl.frozen_eval import sha256
    p=tmp_path/'approval.json';scope={'new_record_ids':['a'],'max_new_records':1}
    p.write_text(json.dumps({'kind':'answer_likelihood_v1','verdict':'PASS_WITH_LIMITATIONS','reviewer':'independent','scope':scope}))
    verify_review(p,sha256(p),scope)
    with pytest.raises(ValueError):verify_review(p,sha256(p),{**scope,'new_record_ids':['b']})
    with pytest.raises(ValueError):verify_review(p,'bad',scope)
