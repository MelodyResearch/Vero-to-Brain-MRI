"""Conditional CPU configs for six calibrated arms; never approves or launches models.

Consumes an exact existing raw release bundle and complete common-B28 calibration.
A separately supplied independent raw-quantile proof digest is mandatory. The
existing tier_c_entry, samplers, native backend, budget and queue remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
from mri_rl import tier_c_entry as te,tier_c_calibration as cal,tier_c_queue as queue
from mri_rl.continuous_rl import digest,require,write_json,sync_dir
from mri_rl.tier_c_calibration_entry import source_hashes as calibration_sources

ROOT=Path(__file__).resolve().parents[2]
PROOF_SCHEMA='common-B28-calibration-independent-quantiles-v1'

def sha(value):return te._sha(value)
def load(path):return json.loads(Path(path).read_text())

def conditions():
    return {'schema':'conditional-calibrated-arm-configs-v1','status':'WAITING_FOR_ACTUAL_CALIBRATION_AND_INDEPENDENT_PROOF',
        'prospective_stage_order':[s['id'] for s in queue.plan_stages() if s.get('needs_calibration')],
        'stages':[],'models_launched':0,'approvals_issued':0,'zero_IQR':'BLOCK_NO_FLOOR',
        'required_proof_schema':PROOF_SCHEMA,'required_proof_fields':['reviewer_kind','reviewer_id','verdict',
            'calibration_plan_sha256','journal_binding_sha256','final_row_receipt_sha256','statistics_sha256',
            'complete_receipt_sha256','ordered_raw_76_verified','raw_quantiles_independently_recomputed'],
        'execution':'NOT_RUN; this module only prepares unapproved scopes and launchers',
        'terminal_verifier_requirement':'Use exact calibration argument in prepare_arm; raw-only terminal preparation is invalid.'}


def authenticate_calibration(run,plan_path,proof_path,expected_proof_sha256,raw_plan,rows=None):
    require(Path(proof_path).is_file() and digest(proof_path)==expected_proof_sha256,'Independent raw-quantile proof digest missing or changed')
    proof=load(proof_path)
    require(proof.get('schema')==PROOF_SCHEMA and proof.get('verdict') in {'PASS','PASS_WITH_LIMITATIONS'} and
            proof.get('reviewer_kind') in {'isolated_codex','claude_code'} and proof.get('reviewer_id') and
            proof.get('ordered_raw_76_verified') is True and proof.get('raw_quantiles_independently_recomputed') is True,
            'Independent calibration proof does not establish ordered actual raw76 and quantiles')
    run=Path(run);plan=load(plan_path);binding=load(run/'binding_PRIVATE.json');complete=load(run/'calibration_complete_SAFE.json')
    require(plan['schema']=='common-B28-training-calibration-v1' and plan['source_sha256']==calibration_sources(), 'Calibration plan/source changed')
    require(plan['common_bindings']==raw_plan['common_bindings'] and plan['manifest_sha256']==raw_plan['manifest_sha256'] and
            plan['processor_sha256']==raw_plan['processor_sha256'] and plan['measurements_sha256']==raw_plan['measurements_sha256'],
            'Calibration differs from common raw-arm weights/data/processor')
    require(rows is not None and len(rows)==76 and [r['id'] for r in rows]==plan['row_ids'] and
            binding['row_ids']==plan['row_ids'] and binding['row_sha256']==plan['row_sha256'] and
            binding['images']==plan['image_sha256'] and binding['prompts_sha256']==sha(plan['prompt_signatures']) and
            binding['caller']['plan_sha256']==sha(plan) and binding['caller']['starting_adapter_sha256']==raw_plan['common_bindings']['starting_adapter_sha256'] and
            binding['group_size']==4 and binding['max_tokens']==256 and binding['token_cap']==179440 and
            binding['helper_sha256']==digest(cal.__file__) and binding['weight_updates']==0,'Calibration run binding differs from frozen76 protocol')
    records,ledger,last,anchor,_=cal._inspect(run,binding,rows,plan['prompt_signatures'])
    require(len(records)==76 and complete['status']=='CALIBRATION_COMPLETE' and complete['complete'] is True and
            complete['rows']==complete['expected_rows']==76 and complete['weight_updates']==0 and
            complete['plan_sha256']==sha(plan) and complete['journal_binding_sha256']==digest(run/'binding_PRIVATE.json') and
            complete['final_row_receipt_sha256']==last and complete['token_cap']==179440,'No authenticated complete actual76 calibration')
    stats=cal.stats_from_accuracies({t:[r['mean_accuracy'] for r in records if r['task']==t] for t in cal.ROUTES})
    require(stats==complete['stats'] and sum(ledger[k] for k in ('measured_tokens','imputed_unknown_tokens'))==complete['cost_tokens']<=179440 and
            ledger['measured_tokens']==complete['measured_tokens'] and ledger['imputed_unknown_tokens']==complete['imputed_unknown_tokens'], 'Raw calibration statistics/cost ledger differs')
    expected={'calibration_plan_sha256':sha(plan),'journal_binding_sha256':digest(run/'binding_PRIVATE.json'),
        'final_row_receipt_sha256':last,'statistics_sha256':sha(stats),'complete_receipt_sha256':digest(run/'calibration_complete_SAFE.json')}
    require(all(proof.get(k)==v for k,v in expected.items()), 'Independent proof belongs to another calibration/raw statistics')
    blocked=any(s['blocked'] for s in stats.values())
    require(complete['calibrated_arms_unblocked']==(not blocked),'Zero-IQR status differs from raw statistics')
    if blocked:return {'blocked':True,'stats':stats,'proof_sha256':expected_proof_sha256,**expected}
    path=run/'calibration_for_arms_PRIVATE.json';value=cal.calibration_for_arms(stats)
    require(path.is_file() and load(path)==value and digest(path)==complete['calibration_file_sha256'],'Frozen calibration file differs from raw quantiles')
    return {'blocked':False,'value':value,'path':str(path.resolve()),'file_sha256':digest(path),'stats':stats,
        'proof_sha256':expected_proof_sha256,**expected}


def stage_configs(plan,rows,measurements,calibration,output,runtime):
    require(set(calibration)==set(cal.ROUTES) and all(math.isfinite(v['p50']) and 0<=v['p50']<=1 and math.isfinite(v['iqr']) and
            0<v['iqr']<=1 for v in calibration.values()),'Complete finite nonzero-IQR calibration required; no floor')
    require(plan['seeds']==[7,11,13] and plan['group_size']==4 and plan['max_tokens']==256 and
            plan['budget']['text_visual_token_cap']==169216 and plan['budget']['maximum_attempted_groups']==198,'Frozen calibrated arm budget changed')
    specs=[]
    for st in queue.plan_stages():
        if not st.get('needs_calibration'):continue
        p=te.prepare_arm(plan,st['arm'],st['seed'],rows,measurements['rows'],calibration)
        raw=te.prepare_arm(plan,st['arm'].replace('calibrated','raw'),st['seed'],rows,measurements['rows'])
        ids=[r['id'] for r in p['schedule']]
        require(ids==[r['id'] for r in raw['schedule']],'Calibration must not change existing P/U sampler realization')
        out=str((Path(output)/f'{st["arm"]}_seed{st["seed"]}'/'run').resolve())
        specs.append({**st,'output':out,'scope':te.review_scope(p,output=out,runtime=runtime),'schedule_ids':ids,
            'schedule_binding':p['schedule_binding'],'realized':p['realized']})
    require(len(specs)==6,'Exact six calibrated arms required')
    return specs


def launch_argv(base,spec,calibration_path):
    argv=list(base)
    require(argv[argv.index('-m')+1]=='mri_rl.tier_c_entry' and 'train' in argv and '--calibration' not in argv and '--review-verdict' not in argv,'Unexpected raw launch template')
    for flag,value in (('--arm',spec['arm']),('--seed',str(spec['seed'])),('--output',spec['output'])):
        require(argv.count(flag)==1,'Ambiguous launch flag');argv[argv.index(flag)+1]=value
    return argv+['--calibration',str(calibration_path)]


def build_bundle(raw_manifest,expected_raw_sha256,calibration_run,calibration_plan,proof,expected_proof_sha256,output):
    require(digest(raw_manifest)==expected_raw_sha256,'Raw bundle identity changed');raw=load(raw_manifest)
    require(raw['source_sha256']==te.source_hashes() and digest(raw['queue_module_path'])==raw['queue_module_sha256'],'Frozen GSPO/queue source changed')
    template=next(s for s in raw['stages'] if s['arm']=='P_raw' and s['seed']==7)
    require(digest(template['launcher'])==template['launcher_sha256'],'Raw launcher changed')
    line=[l for l in Path(template['launcher']).read_text().splitlines() if l.startswith('exec ')]
    require(len(line)==1,'Expected one exact raw launcher command')
    parsed=shlex.split(line[0][5:]);base=parsed[:parsed.index('--review-verdict')]
    require(base[0]==raw['lexical_python'],'Lexical virtualenv interpreter changed')
    plan=load(template['plan']);measurements=load(template['measurements'])
    require(digest(template['plan'])==raw['plan_file_sha256'] and sha(plan)==raw['plan_sha256'] and
            digest(template['train'])==plan['manifest_sha256'],'Existing plan/data bytes changed')
    rows=[json.loads(l) for l in Path(template['train']).read_text().splitlines() if l.strip()]
    auth=authenticate_calibration(calibration_run,calibration_plan,proof,expected_proof_sha256,plan,rows)
    output=Path(output);require(output.resolve().is_relative_to(ROOT/'local_data') and not output.exists(),'Fresh private bundle output required')
    manifest={**conditions(),'schema':'calibrated-arm-config-bundle-v1','source_sha256':te.source_hashes(),
        'builder_sha256':digest(__file__),'raw_manifest_sha256':expected_raw_sha256,'calibration':auth,
        'plan_file_sha256':digest(template['plan']),'plan_sha256':sha(plan),'runtime':raw['runtime']}
    if auth['blocked']:
        manifest['status']='BLOCKED_ZERO_IQR_NO_FLOOR';output.mkdir(parents=True);write_json(output/'manifest_PRIVATE.json',manifest);sync_dir(output);return manifest
    from mri_rl.experiment_contract import load_contract
    arg=lambda flag:Path(base[base.index(flag)+1])
    runtime=te.verify_runtime_artifacts(plan,measurements,arg('--adapter'),arg('--contract'),load_contract)
    evidence=base[base.index('--dynamic-evidence')+1:]
    require(evidence and all(not x.startswith('--') for x in evidence),'Expected frozen dynamic-evidence tail')
    runtime['dynamic_evidence_sha256']={Path(f).name:digest(f) for f in evidence}
    require(runtime==raw['runtime'],'Runtime/prompt/adapter/dynamic evidence differs from existing raw arms')
    specs=stage_configs(plan,rows,measurements,auth['value'],output,runtime)
    output.mkdir(parents=True)
    for spec in specs:
        d=Path(spec['output']).parent;d.mkdir();require(not Path(spec['output']).exists(),'Calibrated native output already exists')
        write_json(d/'scope_NOT_APPROVED_PRIVATE.json',spec['scope'])
        write_json(d/'schedule_PRIVATE.json',{'ids':spec['schedule_ids'],'binding':spec['schedule_binding'],'realized':spec['realized']})
        argv=launch_argv(base,spec,auth['path']);script=d/'launch_after_independent_approval.sh'
        text='#!/bin/sh\nset -eu\n[ "$#" -eq 2 ] || exit 78\ncd '+shlex.quote(str(ROOT))+'\nexport PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1\nexec '+shlex.join(argv)+' --review-verdict "$1" --expected-review-sha256 "$2"\n'
        script.write_text(text);script.chmod(0o700)
        spec.update({'launcher':str(script),'launcher_sha256':digest(script),'scope_sha256':sha(spec['scope']),
            'scope_file_sha256':digest(d/'scope_NOT_APPROVED_PRIVATE.json'),'calibration_file_sha256':auth['file_sha256']})
    manifest.update({'status':'SIX_CONFIGS_NOT_APPROVED_NOT_EXECUTED','stages':specs,'stage_order':[s['id'] for s in specs]})
    write_json(output/'manifest_PRIVATE.json',manifest);sync_dir(output);return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('conditions','build'))
    for k in ('raw-manifest','calibration-run','calibration-plan','proof','output'):p.add_argument('--'+k,type=Path)
    p.add_argument('--expected-raw-sha256');p.add_argument('--expected-proof-sha256');a=p.parse_args()
    if a.command=='conditions':print(json.dumps(conditions(),sort_keys=True));return
    require(all(getattr(a,n) for n in ('raw_manifest','calibration_run','calibration_plan','proof','output','expected_raw_sha256','expected_proof_sha256')),'Exact actual calibration/proof/private bundle arguments required')
    value=build_bundle(a.raw_manifest,a.expected_raw_sha256,a.calibration_run,a.calibration_plan,a.proof,a.expected_proof_sha256,a.output)
    print(json.dumps({'status':value['status'],'configs':len(value['stages']),'models_launched':0,'approvals_issued':0}))

if __name__=='__main__':main()
