"""Offline single-checkpoint fixed-prefix diagnostic; never a checkpoint comparison.

Run only after the model stage stops. Exact immutable prefix adoption is verified.
All saved images are accounted for; pair metrics use only complete frozen pairs.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

_spec=importlib.util.spec_from_file_location('paired_likelihood_analysis',Path(__file__).with_name('analyze_answer_likelihood.py'))
core=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(core)
from mri_rl.likelihood_adoption import verify_adoption
from mri_rl.experiment_contract import canonical_hash
from mri_rl.likelihood_adoption import segment_identity


def stats(values):return core.stats(values) if values else None


def summarize(contract,rows,checkpoint):
    require=core.require;labels=core.LABELS;frozen=contract['records']
    require(0<len(rows)<=len(frozen),'Empty or oversized saved prefix')
    require(len(frozen)==2*contract['patients']+1 and len({r['record_id'] for r in frozen})==len(frozen),'Invalid frozen inventory')
    for actual,expected in zip(rows,frozen):
        require(all(actual.get(k)==v for k,v in expected.items()),'Rows are not exact frozen prefix')
        require(actual['checkpoint']==checkpoint and actual['adapter_sha256']==contract['checkpoints'][checkpoint]['sha256'] and actual['weight_updates']==0,'Checkpoint mismatch')
        core.validate_numeric_record(actual)
        require(actual.get('finish_reason') in (None,'stop','length') and type(actual.get('generation_tokens')) is int and
                0<=actual['generation_tokens']<=256 and actual.get('prompt_tokens')==len(actual['prompt_ids']) and
                actual.get('truncated')==(actual['finish_reason']=='length' if actual['finish_reason'] else None),'Generation measurement mismatch')
    for actual in rows:
        for i,label in enumerate(labels):
            score=actual['candidate_scores'][label];first=score['target_token_logps'][len(score['partition']['prefix_ids'])]
            require(abs(first-actual['prefix_oracle_first_logps'][i])<=.002,'First token oracle mismatch')
    frozen_images=[r for r in frozen if r['image'] is not None]
    references=[r for r in frozen if r['image'] is None]
    require(len(references)==1 and references[0]['record_id']=='no_image_unique','Unique no-image contract required')
    require(all(core.text_prompt(r)==core.text_prompt(references[0]) for r in frozen_images),'Question/prompt mismatch')
    order=list(dict.fromkeys(r['patient_key'] for r in frozen_images))
    require(len(order)==contract['patients'],'Patient inventory mismatch')
    expected_pairs={p:[r for r in frozen_images if r['patient_key']==p] for p in order}
    require(all(len(pair)==2 and {r['visible_label'] for r in pair}==set(labels) for pair in expected_pairs.values()),'Invalid patient pair')
    index={r['record_id']:r for r in rows};metrics={rid:core.row_metrics(r) for rid,r in index.items()}
    complete=[p for p in order if all(r['record_id'] in index for r in expected_pairs[p])]
    require(complete==order[:len(complete)],'Complete pairs are not fixed patient prefix')
    images=[r for r in rows if r['image'] is not None]
    paired=[r for r in images if r['patient_key'] in complete]
    trailing=[r for r in images if r['patient_key'] not in complete]
    require(len(trailing)<=1,'Unexpected noncontiguous partial pair inventory')
    reference=metrics.get(references[0]['record_id'])
    def aggregate(subset):
        confusion={label:{prediction:0 for prediction in (*labels,core.INVALID)} for label in labels}
        termination={'unknown':0,'stop':0,'length':0};full=correct=0
        result={'images':len(subset),'class_counts':{label:sum(r['visible_label']==label for r in subset) for label in labels}}
        for r in subset:
            g=metrics[r['record_id']]['greedy'];confusion[r['visible_label']][g['strict_parsed']]+=1
            full+=g['full_format'];correct+=g['strict_parsed']==r['visible_label'];termination[g['finish'] or 'unknown']+=1
        result['greedy']={'full_format':full,'strict_correct':correct,'confusion':confusion,'termination':termination}
        for kind in core.KINDS:
            values=[metrics[r['record_id']][kind] for r in subset]
            signs=[1 if r['visible_label']==labels[1] else -1 for r in subset]
            result[kind]={'margin':stats([v['margin'] for v in values]),
                'signed_margin':stats([sign*v['margin'] for sign,v in zip(signs,values)]),
                'preferred_correct':sum(v['preferred']==r['visible_label'] for r,v in zip(subset,values)),
                'per_class_margin':{label:stats([metrics[r['record_id']][kind]['margin'] for r in subset if r['visible_label']==label]) for label in labels},
                'signed_real_minus_no_image_margin':stats([sign*(v['margin']-reference[kind]['margin']) for sign,v in zip(signs,values)]) if reference else None,
                'true_label_logp_real_minus_no_image':stats([v['scores'][labels.index(r['visible_label'])]-reference[kind]['scores'][labels.index(r['visible_label'])] for r,v in zip(subset,values)]) if reference else None}
        return result
    separation={kind:[] for kind in core.KINDS}
    for p in complete:
        pair={r['visible_label']:r for r in expected_pairs[p]}
        for kind in core.KINDS:
            separation[kind].append(metrics[pair[labels[1]]['record_id']][kind]['margin']-metrics[pair[labels[0]]['record_id']][kind]['margin'])
    public={'status':'FULL_SINGLE_CHECKPOINT' if len(rows)==len(frozen) else 'PARTIAL_FIXED_PREFIX',
        'checkpoint':checkpoint,'saved_records':len(rows),'planned_records':len(frozen),'complete_patient_pairs':len(complete),
        'trailing_unpaired_images':len(trailing),'all_saved_images':aggregate(images),'complete_pairs':aggregate(paired),
        'trailing_images':aggregate(trailing),'pair_separation':{k:stats(v) for k,v in separation.items()},
        'no_image':{'status':'AVAILABLE' if reference else 'NOT_AVAILABLE','unique_inputs':int(reference is not None),
            'margins':{k:reference[k]['margin'] for k in core.KINDS} if reference else None},
        'cross_checkpoint_comparison':'NOT_RUN','scope':'TRAINING_PATIENT_SINGLE_CHECKPOINT_DIAGNOSTIC',
        'caveats':['Fixed saved prefix; stopping can depend on resources, not a random sample.',
            'Training patients with prior panel exposure; no independent-patient generalization.',
            'Raw, mean-token and first-token scores are different estimands.',
            'Single no-image reference; paired images are not independent patients.',
            'No causal attribution to SFT or label balance without matched checkpoints.']}
    return public,{'complete_patient_keys':complete,'trailing_record_ids':[r['record_id'] for r in trailing],
                   'row_metrics':metrics,'pair_separation':separation}


def load(contract_path,output_path,scope_path):
    scope_bytes=scope_path.read_bytes();scope_doc=json.loads(scope_bytes);scope=scope_doc.get('scope',scope_doc)
    paths=[contract_path,output_path,scope_path];adoption=None
    if scope.get('adopted_prefix'):
        bridge=Path(scope['adopted_prefix']['path']);d=json.loads(bridge.read_text())
        paths.extend([bridge,*[Path(d[k]) for k in ('prefix_path','approval_path','audit_path')]])
    evidence={p:core.sha(p) for p in paths}
    core.require(hashlib.sha256(scope_bytes).hexdigest()==evidence[scope_path],'Scope changed during load')
    raw=output_path.read_bytes();c=json.loads(contract_path.read_bytes())
    core.require(c['version']=='training_answer_likelihood_v1' and c['heldout_used'] is False and c['weight_updates']==0 and
        c['labels']==list(core.LABELS) and c['fixed_prefix']==core.PREFIX and c['fixed_suffix']==core.SUFFIX and
        c['generation']=={'max_tokens':256,'temperature':0},'Frozen contract mismatch')
    core.require(all(scope.get(k)==v for k,v in core.FIXED_PROFILE.items()),'Fixed profile mismatch')
    model=scope['checkpoint'];core.require(model in core.MODELS and scope['contract_sha256']==evidence[contract_path] and
        scope['adapter']==c['checkpoints'][model] and Path(scope['output']).resolve()==output_path.resolve(),'Scope/output mismatch')
    identity=segment_identity(scope)
    core.require(identity==scope['run_identity'],'Identity mismatch')
    if scope.get('adopted_prefix'):
        adoption=verify_adoption(Path(scope['adopted_prefix']['path']),scope)
        core.require(adoption['binding']==scope['adopted_prefix'],'Adoption binding mismatch')
    core.completed_prefix(output_path,c['records'],identity,adoption=adoption)
    rows=[json.loads(line) for line in raw.splitlines()]
    start=scope.get('completed_records');budget=scope.get('max_new_records');allowed=scope.get('new_record_ids')
    finite=scope.get('finite_remaining_stage') is True
    core.require(not finite or (scope.get('resident') is False and scope.get('battery_exception') is not None), 'Invalid finite stage')
    core.require(type(start) is int and 0<=start<=len(c['records']) and type(budget) is int and 1<=budget<=(57 if finite else 8),
        'Invalid approved cursor/budget')
    expected=[r['record_id'] for r in c['records'][start:start+budget]]
    core.require(allowed==expected and start<=len(rows)<=start+len(expected),'Output extends beyond approved scope')
    prior=b''.join(raw.splitlines(keepends=True)[:start])
    prior_hash=hashlib.sha256(prior).hexdigest()
    core.require(scope.get('existing_output_sha256')==prior_hash or
        (start==0 and scope.get('existing_output_sha256') is None),'Approved existing prefix hash mismatch')
    core.require(all(core.sha(p)==h for p,h in evidence.items()),'Evidence changed during load')
    return c,rows,model,evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['contract','output','scope','public-output','private-output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();private_root=Path(__file__).resolve().parents[1]/'local_data'
    core.require(all(p.resolve().is_relative_to(private_root) for p in [args.contract,args.output,args.scope,args.private_output]),'Source and row outputs must remain private')
    core.require(not args.public_output.exists() and not args.private_output.exists(),'Never overwrite analysis')
    c,rows,model,evidence=load(args.contract,args.output,args.scope)
    public,private=summarize(c,rows,model)
    core.require(all(core.sha(p)==h for p,h in evidence.items()),'Evidence changed before publication')
    public['evidence_sha256']={'dependencies':sorted(set(evidence.values())),
        'analyzer':core.sha(__file__),'shared_metrics':core.sha(Path(__file__).with_name('analyze_answer_likelihood.py'))}
    for path,value in [(args.private_output,private),(args.public_output,public)]:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':public['status'],'saved_records':len(rows),'complete_pairs':public['complete_patient_pairs']}))

if __name__=='__main__':main()
