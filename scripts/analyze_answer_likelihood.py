"""CPU-only, fail-closed analysis of the frozen training likelihood diagnostic.

Default requires the full 28-pair cohort and one no-image input at each checkpoint.
A declared --pair-count N analyzes only the fixed first N frozen pairs, explicitly
labeled partial. Missing scores are never imputed or selected by outcome.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from mri_rl.answer_likelihood import LABELS,PREFIX,SUFFIX
from mri_rl.paced_answer_likelihood import validate_numeric_record,completed_prefix
from mri_rl.experiment_contract import canonical_hash
from mri_rl.vero_mri_reward import _answer_and_format

MODELS=('A0','A28','B28')
KINDS=('raw','mean','first_token')
INVALID='INVALID_OR_UNPARSED'
FIXED_PROFILE={'temperature':0,'max_generation_tokens':256,'lora_rank':4,
               'lora_alpha':8,'quantization_bits':4,'group_size':64}


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def stats(values):
    require(bool(values) and all(math.isfinite(v) for v in values),'Missing/nonfinite observations')
    return {'n':len(values),'mean':statistics.mean(values),'median':statistics.median(values),
            'min':min(values),'max':max(values),'sd_descriptive':statistics.pstdev(values)}


def text_prompt(row):
    prompt=row['prompt']
    require(isinstance(prompt,list) and len(prompt)==2 and all(isinstance(x,dict) for x in prompt) and prompt[0].get('role')=='system' and
            isinstance(prompt[0].get('content'),str) and prompt[1].get('role')=='user' and
            isinstance(prompt[1].get('content'),list),'Invalid frozen prompt structure')
    content=prompt[1]['content'];require(all(isinstance(x,dict) for x in content),'Invalid prompt content')
    images=[x for x in content if x.get('type')=='image']
    texts=[x for x in content if x.get('type')=='text']
    require(len(texts)==1 and isinstance(texts[0].get('text'),str) and len(images)+len(texts)==len(content) and
            len(images)==int(row['image'] is not None) and
            (not images or images[0].get('image')==row['image']),'Frozen prompt image/question mismatch')
    return [prompt[0],{'role':'user','content':texts}]


def preference(scores):
    delta=scores[1]-scores[0];peak=max(scores)
    exps=[math.exp(x-peak) for x in scores];prob=[x/sum(exps) for x in exps]
    return {'margin':delta,'preferred':LABELS[1] if delta>0 else LABELS[0] if delta<0 else 'TIE',
        'candidate_preferences':prob,'entropy':-sum(p*math.log(p) for p in prob if p>0)}


def row_metrics(row):
    result={}
    for kind in KINDS:
        scores=(row['prefix_oracle_first_logps'] if kind=='first_token' else
            [row['candidate_scores'][label]['label_logp_sum' if kind=='raw' else 'label_logp_mean'] for label in LABELS])
        result[kind]={**preference(scores),'scores':list(scores)}
    answer,fmt=_answer_and_format(row['response'])
    parsed=next((label for label in LABELS if answer.lower()==label.lower()),INVALID)
    result['greedy']={'parsed':parsed,'format_credit':fmt,'full_format':fmt==1.,
                      'strict_parsed':parsed if fmt==1. else INVALID,'finish':row.get('finish_reason')}
    return result


def metrics_for(rows,metrics,noimage):
    output={'inputs':len(rows),'per_class':{}}
    for kind in KINDS:
        entries=[metrics[row['record_id']][kind] for row in rows]
        signed=[(1 if row['visible_label']==LABELS[1] else -1)*entry['margin'] for row,entry in zip(rows,entries)]
        output[kind]={'margin':stats([e['margin'] for e in entries]),'signed_margin':stats(signed),
          'preferred_correct':sum(e['preferred']==r['visible_label'] for r,e in zip(rows,entries)),
          'ties':sum(e['preferred']=='TIE' for e in entries),'entropy_nats':stats([e['entropy'] for e in entries]),
          'real_minus_no_image_margin':stats([e['margin']-noimage[kind]['margin'] for e in entries]),
          'signed_real_minus_no_image_margin':stats([(1 if r['visible_label']==LABELS[1] else -1)*(e['margin']-noimage[kind]['margin']) for r,e in zip(rows,entries)]),
          'true_label_logp_real_minus_no_image':stats([e['scores'][LABELS.index(r['visible_label'])]-noimage[kind]['scores'][LABELS.index(r['visible_label'])] for r,e in zip(rows,entries)])}
    for label in LABELS:
        subset=[r for r in rows if r['visible_label']==label]
        output['per_class'][label]={kind:{'margin':stats([metrics[r['record_id']][kind]['margin'] for r in subset]),
            'signed_margin':stats([(1 if label==LABELS[1] else -1)*metrics[r['record_id']][kind]['margin'] for r in subset])} for kind in KINDS}
    confusion={label:{p:0 for p in (*LABELS,INVALID)} for label in LABELS}
    parsedconf={label:{p:0 for p in (*LABELS,INVALID)} for label in LABELS}
    term={'unknown':0,'stop':0,'length':0};full=correct=parsedcorrect=0
    for r in rows:
        g=metrics[r['record_id']]['greedy'];truth=r['visible_label']
        confusion[truth][g['strict_parsed']]+=1;parsedconf[truth][g['parsed']]+=1
        full+=g['full_format'];correct+=g['strict_parsed']==truth;parsedcorrect+=g['parsed']==truth
        term[g['finish'] or 'unknown']+=1
    output['greedy']={'full_format':full,'strict_correct':correct,'confusion':confusion,
        'parsed_answer_correct_diagnostic':parsedcorrect,'parsed_confusion_diagnostic':parsedconf,'termination':term}
    fields={'prefix':'prefix_nll_mean','answer':'label_nll_mean','suffix':'suffix_nll_mean',
            'fixed':'fixed_nll_mean','whole_target':'whole_target_nll_mean'}
    output['nll_true_label']={name:stats([r['candidate_scores'][r['visible_label']][field] for r in rows]) for name,field in fields.items()}
    output['nll_by_candidate']={label:{name:stats([r['candidate_scores'][label][field] for r in rows]) for name,field in fields.items()} for label in LABELS}
    output['label_token_counts']={label:sorted({r['candidate_scores'][label]['label_tokens'] for r in rows}) for label in LABELS}
    output['raw_mean_preference_disagreements']=sum(metrics[r['record_id']]['raw']['preferred']!=metrics[r['record_id']]['mean']['preferred'] for r in rows)
    return output


def analyze(contract,outputs,identities,*,pair_count=None):
    require(set(outputs)==set(identities)==set(MODELS),'All A0/A28/B28 outputs and identities required')
    require(contract['version']=='training_answer_likelihood_v1' and contract['heldout_used'] is False and
            contract['weight_updates']==0 and contract['labels']==list(LABELS) and
            contract['fixed_prefix']==PREFIX and contract['fixed_suffix']==SUFFIX and
            contract['generation']=={'max_tokens':256,'temperature':0},'Unexpected frozen diagnostic contract')
    frozen=contract['records'];n=contract['patients'];count=n if pair_count is None else pair_count
    require(type(count) is int and 1<=count<=n,'Invalid prespecified pair count')
    require(len(frozen)==2*n+1 and len({r['record_id'] for r in frozen})==len(frozen),'Invalid frozen input inventory')
    images=[r for r in frozen if r['image'] is not None];none=[r for r in frozen if r['image'] is None]
    require(len(none)==1 and none[0]['record_id']=='no_image_unique' and none[0]['patient_key'] is None,'One unique no-image reference required')
    reference_prompt=text_prompt(none[0])
    require(all(text_prompt(r)==reference_prompt for r in images),'Derived controls require identical system/question prompt')
    patient_order=list(dict.fromkeys(r['patient_key'] for r in images));require(len(patient_order)==n,'Patient count differs')
    pairmap={p:{r['visible_label']:r for r in images if r['patient_key']==p} for p in patient_order}
    for p,pair in pairmap.items():
        require(set(pair)==set(LABELS) and sum(r['patient_key']==p for r in images)==2 and
                all(r['level']==contract['level'] for r in pair.values()),'Incomplete matched patient pair')
    selectedpatients=patient_order[:count];selected=[r for r in images if r['patient_key'] in selectedpatients]
    selectedids={r['record_id'] for r in selected}|{none[0]['record_id']}
    indexed={};metrics={}
    for model in MODELS:
        actual=outputs[model]
        require(0<len(actual)<=len(frozen) and len({r['record_id'] for r in actual})==len(actual),'Missing/duplicate output records')
        expected_identities=identities[model]
        if isinstance(expected_identities,list):require(len(expected_identities)==len(actual),'Wrong segment identity count')
        for index,(row,expected) in enumerate(zip(actual,frozen)):
            require(all(row.get(k)==v for k,v in expected.items()),'Frozen image/label/prompt/record differs')
            require(row.get('run_identity')==(expected_identities[index] if isinstance(expected_identities,list) else expected_identities) and row.get('checkpoint')==model and
                    row.get('adapter_sha256')==contract['checkpoints'][model]['sha256'] and row.get('weight_updates')==0,
                    'Checkpoint/run identity mismatch')
            validate_numeric_record(row)
            require(isinstance(row.get('response'),str) and row.get('finish_reason') in (None,'stop','length') and
                    type(row.get('generation_tokens')) is int and 0<=row['generation_tokens']<=256 and
                    row.get('prompt_tokens')==len(row['prompt_ids']) and
                    row.get('truncated')==(row['finish_reason']=='length' if row['finish_reason'] else None),
                    'Invalid generation measurements')
            for i,label in enumerate(LABELS):
                part=row['candidate_scores'][label]['partition'];first=row['candidate_scores'][label]['target_token_logps'][len(part['prefix_ids'])]
                require(abs(first-row['prefix_oracle_first_logps'][i])<=.002,'Selected first-token likelihood disagrees with oracle')
        indexed[model]={r['record_id']:r for r in actual}
        require(selectedids<=set(indexed[model]),'Incomplete common fixed patient-pair cohort or no-image reference')
        metrics[model]={rid:row_metrics(row) for rid,row in indexed[model].items()}
    # Partial diagnostics may consume additional replacement inputs. Validate
    # every common completed input, including rows outside the primary subset.
    commonids=set.intersection(*(set(indexed[model]) for model in MODELS))
    for rid in commonids:
        base=indexed['A0'][rid]
        for model in ('A28','B28'):
            other=indexed[model][rid]
            for key in ('prompt_ids','prepared_metadata','rendered_prompt_sha256','image_token_count','add_special_tokens'):
                require(base.get(key)==other.get(key),'Cross-checkpoint prompt/pixel configuration mismatch')
            require(all(base['candidate_scores'][label]['partition']==other['candidate_scores'][label]['partition'] for label in LABELS),
                    'Cross-checkpoint target tokenization mismatch')
    public={'schema':'training-answer-likelihood-analysis-v1','scope':'TRAINING_ONLY_EXPLORATORY_DIAGNOSTIC',
        'status':'COMPLETE_FIXED_COHORT' if count==n else 'EXPLICIT_FIXED_PREFIX_PARTIAL_DIAGNOSTIC',
        'frozen_patients':n,'analyzed_patient_pairs':count,'image_inputs_per_checkpoint':2*count,
        'unique_no_image_inputs_per_checkpoint':1,'level':contract['level'],'heldout_generalization':False,
        'new_training_updates':0,'cross_model_review':'NOT_VERIFIED','checkpoints':{},'paired_changes':{},
        'interpretation_limits':['Raw sums retain label-length effects; mean token scores are not calibrated class posteriors.',
          'First-token scores compare only initial label tokens at the common prefix.',
          'Candidate entropy is relative to two candidates, not all possible model responses.',
          'All statistics are descriptive; two image rows and swapped directions are not independent patients.',
          'Fixed prefix/suffix NLL does not establish gradient dominance; no gradient-contribution experiment is inferred.',
          'Checkpoint comparisons use one underlying model family, not a cross-model scientific review.']}
    public['exposure']={'direct_sequence':{model:{label:sum(bool(r['direct_'+model+'_sequence_exposure']) for r in selected if r['visible_label']==label) for label in LABELS} for model in ('A','B')},
        'prior_panel_pixels':'All selected z45 frames had paired-panel exposure; direct sequence-label supervision is a distinct fact.' if contract['level']==45 else
          'Cross-slice panel exposure requires its separate source audit; no unseen-pixel count inferred here.'}
    private={'pairs':[{'patient_key':p,'record_ids':[pairmap[p][label]['record_id'] for label in LABELS]} for p in selectedpatients],
             'mappings':{},'row_metrics':metrics}
    mappings={}
    for kind in ('same_patient_swap','other_patient_opposite'):
        mapping=[]
        for r in selected:
            p=r['patient_key'];replacement_patient=p if kind=='same_patient_swap' else patient_order[(patient_order.index(p)+1)%n]
            target=pairmap[replacement_patient][LABELS[1-LABELS.index(r['visible_label'])]]
            require(kind=='same_patient_swap' or replacement_patient!=p,'Other-patient control needs at least two patients')
            mapping.append({'origin_record_id':r['record_id'],'replacement_record_id':target['record_id'],
                'origin_patient':p,'replacement_patient':replacement_patient,'origin_label':r['visible_label'],'replacement_label':target['visible_label']})
        require(len({x['replacement_record_id'] for x in mapping})==len(mapping),'Derived replacement mapping is not one-to-one')
        private['mappings'][kind]=mapping;mappings[kind]=mapping
    for model in MODELS:
        met=metrics[model];rows=[indexed[model][r['record_id']] for r in selected];no=met[none[0]['record_id']]
        result={'images':metrics_for(rows,met,no),'no_image':{'unique_inputs':1,**no},'pairs':{}}
        for kind in KINDS:
            separations=[met[pairmap[p][LABELS[1]]['record_id']][kind]['margin']-met[pairmap[p][LABELS[0]]['record_id']][kind]['margin'] for p in selectedpatients]
            both=sum(all(met[pairmap[p][label]['record_id']][kind]['preferred']==label for label in LABELS) for p in selectedpatients)
            result['pairs'][kind]={'separation':stats(separations),'positive_separation_pairs':sum(x>0 for x in separations),'both_correct_pairs':both}
        for control,mapping in mappings.items():
            if not all(x['replacement_record_id'] in indexed[other] for other in MODELS for x in mapping):
                result[control]={'status':'NOT_READY_INCOMPLETE_FIXED_REPLACEMENT_INPUTS','new_forwards':0,'new_patients':0};continue
            targetrows=[indexed[model][x['replacement_record_id']] for x in mapping]
            result[control]={'status':'DERIVED_EXACT_VISIBLE_INPUT_PERMUTATION','new_forwards':0,'new_patients':0,
                'directions':len(mapping),'unique_origin_patient_pairs':count,'displayed_label_scoring':metrics_for(targetrows,met,no)}
            for kind in KINDS:
                correct=sum(met[x['origin_record_id']][kind]['preferred']==x['origin_label'] and
                            met[x['replacement_record_id']][kind]['preferred']==x['replacement_label'] for x in mapping)
                shifts=[(1 if x['replacement_label']==LABELS[1] else -1)*(met[x['replacement_record_id']][kind]['margin']-met[x['origin_record_id']][kind]['margin']) for x in mapping]
                result[control][kind]={'correct_direction_flips':correct,'directional_signed_margin_change':stats(shifts)}
                if control=='same_patient_swap':result[control][kind]['both_correct_pairs']=correct//2
            result[control]['greedy']={'correct_direction_flips':sum(met[x['origin_record_id']]['greedy']['strict_parsed']==x['origin_label'] and
                met[x['replacement_record_id']]['greedy']['strict_parsed']==x['replacement_label'] for x in mapping)}
        public['checkpoints'][model]=result
    for before,after in [('A0','A28'),('A28','B28'),('A0','B28')]:
        change={};a=metrics[before];b=metrics[after]
        for kind in KINDS:
            deltas=[b[r['record_id']][kind]['margin']-a[r['record_id']][kind]['margin'] for r in selected]
            signed=[(1 if r['visible_label']==LABELS[1] else -1)*delta for r,delta in zip(selected,deltas)]
            pairmean=[statistics.mean(value for r,value in zip(selected,signed) if r['patient_key']==patient)
                      for patient in selectedpatients]
            change[kind]={'margin_delta':stats(deltas),'signed_margin_delta':stats(signed),'within_patient_mean_signed_delta':stats(pairmean),
              'per_class':{label:stats([d for r,d in zip(selected,deltas) if r['visible_label']==label]) for label in LABELS},
              'greedy_unchanged_but_margin_changed':sum(a[r['record_id']]['greedy']['strict_parsed']==b[r['record_id']]['greedy']['strict_parsed'] and abs(d)>1e-10 for r,d in zip(selected,deltas)),
              'no_image_margin_delta':b['no_image_unique'][kind]['margin']-a['no_image_unique'][kind]['margin']}
        change['greedy']={'strict_answer_changed':sum(a[r['record_id']]['greedy']['strict_parsed']!=b[r['record_id']]['greedy']['strict_parsed'] for r in selected)}
        public['paired_changes'][after+'_minus_'+before]=change
    return public,private


def validate_scope_extent(scope,raw,records):
    """Bind saved prefix to this review's cursor and exact permitted next slice."""
    start=scope.get('completed_records');budget=scope.get('max_new_records');allowed=scope.get('new_record_ids')
    require(type(start) is int and 0<=start<=len(records) and type(budget) is int and 1<=budget<=8,
        'Invalid approved cursor/budget')
    expected=[r['record_id'] for r in records[start:start+budget]]
    lines=raw.splitlines(keepends=True)
    require(allowed==expected and start<=len(lines)<=start+len(expected),'Output extends beyond approved scope')
    prior_hash=hashlib.sha256(b''.join(lines[:start])).hexdigest()
    require(scope.get('existing_output_sha256')==prior_hash or
        (start==0 and scope.get('existing_output_sha256') is None),'Approved existing prefix hash mismatch')


def load_scoped(contract_path,outputs,scopes):
    paths=[contract_path,*outputs.values(),*scopes.values()]
    require(all(path.exists() for path in paths),'Missing likelihood output/contract/scope')
    snapshots={path:path.read_bytes() for path in paths}
    hashes={path:hashlib.sha256(raw).hexdigest() for path,raw in snapshots.items()}
    c=json.loads(snapshots[contract_path]);contract_hash=hashes[contract_path];identities={};data={};profiles=[]
    for model in MODELS:
        scope_doc=json.loads(snapshots[scopes[model]]);s=scope_doc.get('scope',scope_doc)
        require(all(s.get(key)==value for key,value in FIXED_PROFILE.items()),'Execution profile differs from frozen diagnostic')
        require(s['checkpoint']==model and s['contract_sha256']==contract_hash and
                s['adapter']==c['checkpoints'][model] and Path(s['output']).resolve()==outputs[model].resolve(),
                'Reviewed scope does not bind these contract/output/checkpoint bytes')
        identity=canonical_hash({'contract':contract_hash,'checkpoint':model,'sources':s['source_hashes'],
            'backend':s['backend_sources'],'battery_diagnostic':s.get('battery_exception') is not None})
        require(identity==s['run_identity'],'Scope run identity cannot be reconstructed')
        profiles.append({k:s[k] for k in ('source_hashes','backend_sources','temperature','max_generation_tokens','lora_rank','lora_alpha','quantization_bits','group_size')})
        adoption=None
        if s.get('adopted_prefix'):
            from mri_rl.likelihood_adoption import verify_adoption
            bridge=Path(s['adopted_prefix']['path'])
            adoption=verify_adoption(bridge,s)
            require(adoption['binding']==s['adopted_prefix'],'Adoption scope digest/profile mismatch')
            hashes[bridge]=s['adopted_prefix']['sha256']
        completed_prefix(outputs[model],c['records'],identity,adoption=adoption)
        validate_scope_extent(s,snapshots[outputs[model]],c['records'])
        require(outputs[model].exists(),'Missing output: '+model)
        data[model]=[json.loads(line) for line in snapshots[outputs[model]].splitlines()]
        identities[model]=(adoption['identities']+[identity]*(len(data[model])-len(adoption['identities']))) if adoption else identity
    require(all(profile==profiles[0] for profile in profiles),'Nonmatching model/prompt implementation profiles')
    require(all(sha(path)==value for path,value in hashes.items()),'Likelihood evidence changed during validation')
    return c,data,identities


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    for model in MODELS:
        p.add_argument('--'+model.lower(),type=Path,required=True);p.add_argument('--'+model.lower()+'-scope',type=Path,required=True)
    p.add_argument('--pair-count',type=int);p.add_argument('--public-output',type=Path,required=True);p.add_argument('--private-output',type=Path,required=True)
    args=p.parse_args();private_root=Path(__file__).resolve().parents[1]/'local_data'
    require(args.contract.resolve().is_relative_to(private_root) and args.private_output.resolve().is_relative_to(private_root),
            'Source-linked contract and row-level output must remain in ignored local_data')
    require(not args.public_output.exists() and not args.private_output.exists(),'Never overwrite prior analysis')
    paths={m:getattr(args,m.lower()) for m in MODELS};scopes={m:getattr(args,m.lower()+'_scope') for m in MODELS}
    require(all(path.resolve().is_relative_to(private_root) for path in [*paths.values(),*scopes.values()]),'Private outputs/scopes must remain ignored')
    evidence_paths=[args.contract,*paths.values(),*scopes.values()]
    for scope_path in scopes.values():
        document=json.loads(scope_path.read_text());scope=document.get('scope',document)
        if scope.get('adopted_prefix'):
            bridge=Path(scope['adopted_prefix']['path'])
            require(bridge.resolve().is_relative_to(private_root),'Adoption bridge must remain private')
            bridge_doc=json.loads(bridge.read_text())
            children=[Path(bridge_doc[key]) for key in ('prefix_path','approval_path','audit_path')]
            require(all(path.resolve().is_relative_to(private_root) for path in children),'Adoption evidence must remain private')
            evidence_paths.extend([bridge,*children])
    evidence={path:sha(path) for path in evidence_paths}
    c,rows,identities=load_scoped(args.contract,paths,scopes);require(c['patients']==28,'Production analysis requires frozen 28-patient design')
    public,private=analyze(c,rows,identities,pair_count=args.pair_count)
    require(all(sha(path)==value for path,value in evidence.items()),'Likelihood evidence changed before publication')
    public['evidence_sha256']={'contract':evidence[args.contract],'analyzer':sha(__file__),
        'outputs':{m:evidence[path] for m,path in paths.items()},'scopes':{m:evidence[path] for m,path in scopes.items()},
        'adoption_dependencies':sorted({evidence[path] for path in evidence_paths if path not in [args.contract,*paths.values(),*scopes.values()]})}
    # Constructed public object is a whitelist: no source rows, IDs, paths or text copied.
    for path,value in [(args.private_output,private),(args.public_output,public)]:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':public['status'],'patient_pairs':public['analyzed_patient_pairs'],'public_sha256':sha(args.public_output)}))

if __name__=='__main__':main()
