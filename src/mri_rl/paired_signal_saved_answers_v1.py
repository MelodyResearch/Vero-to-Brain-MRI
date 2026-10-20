"""Saved-vocabulary TRAIN diagnostic; never imports MLX or changes a scorer.

Actual generated thinking histories are NOT a shared canonical label prefix.
For an ungenerated multi-token class only a rigorous logp<=0 upper bound is
available; no missing conditional score is invented.
"""
import hashlib
import json
from pathlib import Path
import re
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def score_answer_context(logits,start,actual_label,correct_label):
    a=np.asarray(logits)
    if a.ndim!=2 or not actual_label or not correct_label or not 0<=start<=a.shape[0]-len(actual_label):raise ValueError('Actual response-aligned label rows required')
    rows=a[start:start+len(actual_label)].astype(np.float64)
    if not np.isfinite(rows).all():raise ValueError('Nonfinite saved logits')
    ids=[*actual_label,*correct_label]
    if any(type(v) is not int or not 0<=v<a.shape[1] for v in ids):raise ValueError('Vocabulary IDs required')
    z=np.logaddexp.reduce(rows,axis=1)
    wrong=float(np.mean([rows[i,t]-z[i] for i,t in enumerate(actual_label)]))
    first=float(rows[0,correct_label[0]]-z[0]);firstmargin=float(rows[0,correct_label[0]]-rows[0,actual_label[0]])
    exact=firstmargin if len(actual_label)==len(correct_label)==1 else None
    return {'first_token_margin':firstmargin,'actual_wrong_label_mean_logp':wrong,
        'correct_mean_logp_upper_bound':first/len(correct_label),
        'correct_minus_wrong_mean_upper_bound':first/len(correct_label)-wrong,
        'correct_minus_wrong_complete_label_margin':exact,'common_canonical_prefix_measured':False}

def analyze(old_worktree):
    from tokenizers import Tokenizer
    from mri_rl.vero_mri_reward import _answer_and_format
    old=Path(old_worktree).resolve();base=old/'local_data/paired_z045_f32_qualification_v1'
    scope=base/'scope_IMAGE_BOUND_FINAL_PRIVATE.json';complete=base/'actual_once_v1_PRIVATE/complete_PRIVATE.json'
    expected={'scope':'865aeda5a772c9a917370bacddfd84309ac51ac225cd52732e0ee0e2f56f3511',
              'complete':'ea8c354b99f17409e8c2fdeaa9f572f2188995159da02d26302c62029387d69d'}
    if digest(scope)!=expected['scope'] or digest(complete)!=expected['complete']:raise ValueError('Frozen actual evidence changed')
    spec=old/'local_data/paired_z045_continuation_v4/spec_PREPARATION_ONLY_PRIVATE.json'
    if digest(spec)!='ddd53814a73ac13f5e819f9f3835f579f5bed712d352cf5f4c7b5680028b8993':raise ValueError('Original runtime spec changed')
    tokenizer=Path(json.loads(spec.read_text())['model'])/'tokenizer.json';tok=Tokenizer.from_file(str(tokenizer));enc=lambda x:tok.encode(x,add_special_tokens=False).ids
    rows=json.loads(scope.read_text())['rows_PRIVATE'];inventory={str(p):digest(p) for p in [scope,complete,spec,tokenizer]};groups=[]
    for g,row in enumerate(rows):
        folder=base/f'actual_once_v1_PRIVATE/group_{g}_PRIVATE';path=folder/'rollout_PRIVATE.json';inventory[str(path)]=digest(path)
        rollout=json.loads(path.read_text());truth=row['reward_model']['ground_truth'];records=[]
        if len(rollout['responses'])!=4:raise ValueError('Exact four saved responses required')
        for i,response in enumerate(rollout['responses']):
            ids=response['token_ids'];text=tok.decode(ids,skip_special_tokens=False)
            if enc(text)!=ids:raise ValueError('Saved token decode/encode not exact')
            spans=list(re.finditer(r'\\boxed\{([^{}]+)\}',text))
            if len(spans)!=1:raise ValueError('One actual boxed class required')
            span=spans[0];label=span.group(1);start=len(enc(text[:span.start(1)]));actual=enc(label)
            labels=('FLAIR','contrast-enhanced T1') if g==0 else ('same','different')
            if label not in labels or truth not in labels or label==truth:raise ValueError('Expected fixed historical incorrect canonical label')
            if ids[:start]!=enc(text[:span.start(1)]) or ids[start:start+len(actual)]!=actual:raise ValueError('Actual label token boundary differs')
            p=folder/f'response_{i}_full_TF_vocab_PRIVATE.npy';inventory[str(p)]=digest(p);logits=np.load(p,mmap_mode='r')
            if logits.ndim==3 and logits.shape[0]==1:logits=logits[0]
            result=score_answer_context(logits,start,actual,enc(truth));del logits
            records.append({'response_index':i,'generated_correct':False,'normal_EOS':ids[-1] in (151643,151645),
                'returned_tokens':len(ids),'format_valid':_answer_and_format(response['text'])[1]==1.,**result})
        groups.append({'task':row['ability'],'responses':records})
    if not all(r['normal_EOS'] for g in groups for r in g['responses']):raise ValueError('Unexpected historical EOS')
    public={'schema':'paired-signal-saved-answer-context-v1','status':'SAVED_TRAIN_ANALYSIS_COMPLETED','groups':groups,
        'scope_sha256':expected['scope'],'actual_complete_sha256':expected['complete'],'tokenizer_sha256':inventory[str(tokenizer)],
        'inventory_sha256':hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'canonical_common_prefix_scores':'NOT_MEASURED','image_swap_and_null':'NOT_MEASURED',
        'conclusion':'Wrong class dominates at each actual generated answer history; common-prefix scores and image dependence remain unresolved.',
        'limitations':'TRAIN two fixed contexts, four stochastic outputs each, no independent accuracy estimate. Multi-token correct-label value is an upper bound, not a measured alternative continuation.',
        'new_model_calls':0,'training_runs':0,'optimizer_updates':0}
    return public,{'file_sha256_PRIVATE':inventory,'public_result':public}

if __name__=='__main__':
    import argparse,sys
    p=argparse.ArgumentParser();p.add_argument('--old-worktree',required=True);p.add_argument('--public-output',required=True);p.add_argument('--private-output',required=True);a=p.parse_args()
    result,private=analyze(a.old_worktree)
    if any(k=='mlx' or k.startswith('mlx.') for k in sys.modules):raise RuntimeError('CPU-only analysis imported MLX')
    for path,value in [(a.public_output,result),(a.private_output,private)]:
        out=Path(path);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':result['status'],'new_model_calls':0,'optimizer_updates':0}))
