"""CPU inspection or fully gated future F32 GSPO+supervised auxiliary continuation.

No model or native import at inspection. Present preparation does not authorize
training. Old read-only, inactive-backward and battery exceptions never unlock
this entry. Only an exact future numerical proof, human permission and review do.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from mri_rl import class_margin_aux_v2 as integration
from mri_rl import class_margin_aux_v2_driver as driver
from mri_rl import paired_z045_continuation_entry_v1 as inherited
from mri_rl.continuous_rl import require
from mri_rl.paired_z045_matched_f32_eval_v1 import token_plan,score_input_positions
from mri_rl.answer_likelihood import PREFIX

ROOT=Path(__file__).resolve().parents[2]


def _json(path):return json.loads(Path(path).read_text())


def inputs(spec):
    for key,expected in spec['file_sha256'].items():require(inherited.digest(spec[key])==expected,'Frozen specification input changed: '+key)
    runtime=_json(spec['runtime_spec'])
    for key,expected in runtime['file_sha256'].items():require(inherited.digest(runtime[key])==expected,'Frozen runtime input changed: '+key)
    return runtime


def same_image_bytes(original,companion,expected):
    require(inherited.digest(original)==inherited.digest(companion)==expected,'Main/companion authenticated PNG bytes differ')
    return True


def schedule_and_pairs(spec,runtime):
    preparation=_json(spec['paired_preparation']);rows=preparation['main_original_schedule_PRIVATE'][:4]
    ref=_json(runtime['reference_schedule']);ref=ref['schedule'] if isinstance(ref,dict) else ref
    require(len(rows)==4 and [row['ability'] for row in rows]==['mri_sequence','mri_visit_correspondence','mri_sequence','mri_visit_correspondence'],'Original slots8:12 must remain2sequence2Dallas')
    for row,original in zip(rows,ref[:4]):
        require({k:v for k,v in row.items() if k!='id'}=={k:v for k,v in original.items() if k!='id'},'Original28+48 uniform nextsource occurrence changed')
    pool=[json.loads(line) for line in Path(runtime['sequence56']).read_text().splitlines() if line.strip()]
    normalized={row['id']:row for row in preparation['normalized_pair_records_PRIVATE']};pairs={};expected={}
    inventory=_json(runtime['original_image_inventory']);receipt=_json(runtime['original_pool_receipt'])
    require(receipt['private_image_inventory_sha256']==inherited.digest(runtime['original_image_inventory']),'Original image inventory not receipt authenticated')
    for row,route in zip(rows,preparation['routing_PRIVATE'][:4]):
        if row['ability']=='mri_visit_correspondence':
            require(route['pair_ids'] is None and route['main']['record_id']==row['id'],'Dallas must have no auxiliary companion')
            expected[row['id']]=inventory['image_sha256_by_record'][row['source_record_id']];continue
        matches=[]
        for id in route['pair_ids']:
            metadata=normalized[id]
            found=[candidate for candidate in pool if candidate['extra_info']['patient_key']==row['extra_info']['patient_key']==metadata['patient'] and
                   candidate['reward_model']['ground_truth']==metadata['label'] and candidate['rc_data_v1']['image_sha256']==metadata['image_sha256']]
            require(len(found)==1 and metadata['level']==45 and metadata['question']==row['extra_info']['question'],'Exact original TRAIN F/T companion missing')
            candidate=found[0];expected[candidate['id']]=metadata['image_sha256'];matches.append(candidate)
        matches.sort(key=lambda candidate:('FLAIR','contrast-enhanced T1').index(candidate['reward_model']['ground_truth']))
        require(len(matches)==2 and {r['reward_model']['ground_truth'] for r in matches}=={'FLAIR','contrast-enhanced T1'},'Canonical F/T pair required')
        main=[r for r in matches if r['reward_model']['ground_truth']==row['reward_model']['ground_truth']]
        require(len(main)==1,'Auxiliary main class missing');same_image_bytes(row['images'][0],main[0]['images'][0],expected[main[0]['id']])
        expected[row['id']]=expected[main[0]['id']];pairs[row['id']]=matches
    inherited.verify_images(rows+[r for pair in pairs.values() for r in pair],expected)
    return rows,pairs,expected


def prepare(spec):
    runtime=inputs(spec);parent=inherited.parent_metadata(runtime['parent_checkpoint'],runtime['parent_receipt_sha256'])
    rows,pairs,images=schedule_and_pairs(spec,runtime)
    configuration={**driver.old.PROFILE,'group_size':4,'max_tokens':256}
    require(configuration==parent['original_binding']['configuration'],'Inherited reward/LR/LoRA/K4 configuration changed')
    calibration=_json(runtime['calibration']);arm=parent['original_binding']['experiment']['parent_experiment']
    require(arm['arm']=={'name':'U_calibrated','reward':'medgrpo_eq3','sampling':'uniform'} and arm['seed']==13 and inherited.scope_digest(calibration)==arm['calibration_sha256'],'Original task-routed calibration differs')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(Path(runtime['model'])/'tokenizer.json'))
    plan=token_plan('mri_sequence',PREFIX,lambda text:tokenizer.encode(text,add_special_tokens=False).ids)
    aux_per_sequence=2*sum(score_input_positions(213,len(plan['prefix']),len(ids)) for ids in plan['labels'])
    require(aux_per_sequence==906,'Frozen canonical auxiliary token costs changed')
    sources=inherited.source_closure(runtime)
    names=('class_margin_aux_v2.py','class_margin_aux_v2_driver.py','class_margin_aux_v2_entry.py','paired_classification_signal_v2.py','paired_class_margin_probe_v2.py','paired_interaction_signal_v1.py','paired_z045_matched_f32_eval_v1.py')
    for name in names:sources[str(ROOT/'src/mri_rl'/name)]=inherited.digest(ROOT/'src/mri_rl'/name)
    for path in (ROOT/'scripts/run_class_margin_aux_v2.py',ROOT/'tests/test_class_margin_aux_v2.py'):sources[str(path)]=inherited.digest(path)
    artifacts={str(spec[key]):expected for key,expected in spec['file_sha256'].items()}
    artifacts.update({str(runtime[key]):expected for key,expected in runtime['file_sha256'].items()})
    output=Path(spec['output']).resolve();require(output.is_relative_to(ROOT/'local_data/class_margin_aux_v2'),'Fresh isolated private output required')
    require(spec['maximum_seconds']==1200,'Exact1200s preparation envelope required')
    return {'schema':'class-margin-aux-v2-preparation','implementation_status':integration.STATUS,'scope_kind':'PREPARATION_ONLY_NOT_TRAIN_AUTHORIZATION',
        'training_authorized':False,'native_approved':False,'model_loads_this_preparation':0,'optimizer_updates_this_preparation':0,
        'parent':parent,'runtime_spec_sha256':inherited.digest(spec['runtime_spec']),'schedule_PRIVATE':rows,'companions_PRIVATE':pairs,
        'expected_image_sha256':images,'aux_token_plan_PRIVATE':plan,'configuration':configuration,'source_sha256':sources,'bound_artifact_sha256':artifacts,
        'derived_cache':inherited.cache_metadata(runtime)['expected_binding'],'output':str(output),
        'budget':{'maximum_attempted_groups':4,'maximum_optimizer_updates':2,'group_size':4,'max_tokens':256,
                  'main_rollout_text_visual_token_cap':9040,'auxiliary_TF_positions_cap':1812,'combined_rollout_plus_auxiliary_cap':10852,
                  'maximum_generated_tokens':4096,'full_TF_forwards':40,'backward_calls':20,'teacherforced_positions':19860,
                  'generation_LM_calls_including_lookahead':4128,'vision_embeddings':56,'maximum_seconds':1200,'simultaneous_models':1,'simultaneous_image_graphs':1},
        'bindings':{'stage':'class_margin_aux_v2','spec_sha256':inherited.stage_spec_binding(spec),'paired_preparation_sha256':inherited.digest(spec['paired_preparation']),
                    'objective':'unchanged admitted GSPO + .1*(softplus(-mF)+softplus(-mT))/2, mean-label margins only; explicit supervised auxiliary',
                    'sampling':'original28sequence+48Dallas uniform seed13 nextslots8:12; paired opposite modality auxiliary only; no main pool expansion',
                    'evaluation_sealed_manifest_sha256':spec.get('evaluation_sealed_manifest_sha256','PENDING')},
        'numerical_prerequisites':{'status':'PENDING_INDEPENDENT_COMPATIBILITY_ADMISSION','active_F32_GSPO_backward':'NOT_QUALIFIED_BY_OLD_INACTIVE_PROBE',
            'auxiliary_VJP':'CPU_TESTED_ONLY_AWAIT_READONLY_MARGIN_ACTUAL','joint_one_Adam':'CPU_FIXTURE_ONLY_NATIVE_UNMEASURED','full_resume_integrity':'CPU_FIXTURE_ONLY','no_BF16_fallback':True,'admission':'Independent compatibility certificate binds exact training sources/budget and CPU transaction proof, actual readonly margin artifact/source/context+review; nativejoint/onpolicy remains explicitly unmeasured. Future human+review must accept limitations, every admitted group dynamically gates selected rollout/scorer.002, owncurrent1e-5, finite/nonzero gradients before step.'},
        'counting_contract':'Main rollout charged<=9040 includes returned tokens/EOS; auxiliaryTF<=1812 separately; all fullTF API inputpositions<=19860 includes GSPO scorer. AR unreturned lookahead and checkpoint internal recomputation are additional work, not charged returnedresponse tokens; actual completed LM evidence counts reported. No claim10852 is total compute.',
        'forward_reuse':'Per admitted GSPO group four detached own-old forwards reused for rollout alignment, plus four fresh gradient forwards; redundant old/current standalone check removed but original loss AST and inside-gradient1e-5 guard unchanged.',
        'state_contract':'Exact Adam60fullstate/global129; serial gradients then1persistentAdam. Base1438 unchanged/all1942 fingerprints;504LoRA allowed once per actualstep. Rollback pre-group Adam/RNG/params on uncommitted failure; completed rename never replayed. Complete pending rollout+postgeneration RNG reused, incomplete pending STOP/no automatic retry.',
        'resources':'Original global training AC/PacedGate/nootherheavy unchanged; no inherited localbattery exception. Absolute1200s operation-start deadline acrossresume; atomicgroup+release mayoverrun.',
        'mission_alignment':{'evidence_tier':'CPU_IMPLEMENTED_NOT_RUN','tag':'SUPERVISED_AUXILIARY_INTERVENTION','training_claim':False},
        'limitations':['No model/native/update executed; fresh human and independent exact launch review required.','Native joint/onpolicy backward unmeasured; independent limited-evidence admission certificate and explicit human/reviewer acceptance pending; readonly margin proof is not full training qualification.','At most2TOTALupdates may stop before4slots; acceptedGSPO separate from auxiliary-only.','Mean score is length-normalized, not canonical-class event logprobability; no claimed accuracy gain.','Heldout payloads/labels and final outputs never read by this preparation.']}



def validate_admission(prepared,scope,certificate):
    """Authenticate distinct readonly evidence without relabeling it as training."""
    require(certificate.get('schema')=='independent-class-margin-v2-compatibility-admission-v1','Unknown numerical admission certificate type')
    require(certificate.get('preparation_sha256')==inherited.scope_digest(prepared) and certificate.get('training_source_sha256')==prepared['source_sha256'] and
            certificate.get('training_budget')==prepared['budget'],'Admission preparation/source/budget changed')
    def authenticated(key):
        record=certificate[key];require(inherited.digest(record['path'])==record['sha256'],'Admission evidence file digest differs: '+key)
        return _json(record['path'])
    proof=authenticated('readonly_actual');probe=authenticated('readonly_scope');actual_review=authenticated('readonly_actual_review');cpu=authenticated('CPU_transaction_proof')
    require(proof.get('status')=='ACTUAL_MARGIN_VJP_READONLY_COMPLETE' and proof.get('optimizer_updates')==0 and proof.get('generation_calls')==0 and
            proof.get('scope_sha256')==inherited.scope_digest(probe) and proof.get('source_sha256')==probe.get('source_sha256'),'Readonly actual margin evidence absent/foreign; old inactive probes cannot substitute')
    require(probe.get('parent',{}).get('receipt_sha256')==prepared['parent']['receipt_sha256'],'Readonly probe has a different parent checkpoint')
    require(actual_review.get('actual_proof_sha256')==certificate['readonly_actual']['sha256'] and actual_review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and
            actual_review.get('reviewer_id') and actual_review.get('reviewer_kind') in ('isolated_codex','claude_code'),'Readonly actual evidence lacks authenticated independent review')
    common=('paired_classification_signal_v2.py','paired_class_margin_probe_v2.py','paired_z045_f32_backend_v1.py',
            'conditional_gspo_gradient_v2.py','native_probability_resolution_v1.py','rollout_precision_v2.py','mlx_continuous_gspo.py')
    for name in common:
        training={value for path,value in prepared['source_sha256'].items() if Path(path).name==name}
        readonly={value for path,value in probe['source_sha256'].items() if Path(path).name==name}
        require(len(training)==1 and training==readonly,'Readonly/training shared numerical source differs: '+name)
    installed={path:value for path,value in prepared['source_sha256'].items() if '/site-packages/' in path}
    require(installed and all(probe['source_sha256'].get(path)==value for path,value in installed.items()),'Installed numerical runtime compatibility differs')
    require(cpu.get('schema')=='class-margin-aux-v2-cpu-transactions' and cpu.get('status')=='TESTED_CPU' and cpu.get('native_execution') is False,'CPU transaction proof required; not native training evidence')
    for name in ('class_margin_aux_v2.py','class_margin_aux_v2_driver.py','class_margin_aux_v2_entry.py'):
        keys=[path for path in prepared['source_sha256'] if Path(path).name==name]
        require(len(keys)==1 and cpu.get('tested_source_sha256',{}).get(keys[0])==prepared['source_sha256'][keys[0]],'CPU proof source changed: '+name)
    require(certificate.get('dynamic_group_gates')=={'rollout_selected_vs_full_max_abs':.002,'own_old_current_max_abs':1e-5,'finite_nonzero_final_gradient':True,'single_persistent_Adam':True},'Dynamic pre-step qualification gates changed')
    require(scope.get('admission_certificate_sha256')==inherited.scope_digest(certificate),'Exact compatibility admission certificate is not scope bound')
    return True

def execute(args):
    required=('spec','scope','expected_scope_sha256','numeric','expected_numeric_sha256','human','expected_human_sha256','review','expected_review_sha256')
    require(all(getattr(args,key,None) for key in required),'NOT_AUTHORIZED: exact new scope/numeric/human/review artifacts required before model import')
    for key in ('scope','numeric','human','review'):require(inherited.digest(getattr(args,key))==getattr(args,'expected_'+key+'_sha256'),'Exact launch artifact changed: '+key)
    spec=_json(args.spec);prepared=prepare(spec);scope=_json(args.scope);numeric=_json(args.numeric)
    require(scope.get('preparation_sha256')==inherited.scope_digest(prepared) and scope.get('budget')==prepared['budget'] and scope.get('source_sha256')==prepared['source_sha256'],'Exact prepared training source/budget binding differs')
    validate_admission(prepared,scope,numeric)
    integration.authorize_launch(scope,numeric,_json(args.human),_json(args.review))
    require(len(str(prepared['bindings']['evaluation_sealed_manifest_sha256']))==64,'Prospective sealed evaluation manifest hash required before training; do not read payload')
    import fcntl
    from types import SimpleNamespace
    from mri_rl.paced_resources import PacedGate
    from mri_rl.two_task_training import TokenLedger,make_grader,validate_training_row
    from mri_rl.mlx_continuous_gspo import is_loading_unit
    from mri_rl.continuous_rl_pilot import cache_binding_for
    from mri_rl.experiment_contract import load_contract
    runtime=inputs(spec);out=Path(prepared['output']);gate=PacedGate(out.with_name(out.name+'_resources_PRIVATE.jsonl'));resident=[None]
    def before(name):
        deadline=out/'operation_deadline_PRIVATE.json'
        def check():
            if deadline.exists():require(time.time()<_json(deadline)['deadline'],'Absolute training operation-start deadline exhausted')
        check();gate.before_unit(name,loading=is_loading_unit(name,{'dir':runtime['cache_dir']}));check()
        inherited.verify_images(prepared['schedule_PRIVATE']+[r for pair in prepared['companions_PRIVATE'].values() for r in pair],prepared['expected_image_sha256'])
    with Path(runtime['global_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);require(prepare(spec)==prepared,'Preparation changed while obtaining global lock')
        evidence=out.with_name(out.name+'_launch_evidence_PRIVATE');require(not evidence.exists(),'Fresh per-invocation evidence output required');evidence.mkdir(parents=True)
        def factory(adapter,configuration):
            before('certified_cache_binding')
            derived=cache_binding_for(SimpleNamespace(cache_dir=Path(runtime['cache_dir']),cache_certificate=Path(runtime['cache_certificate']),cache_certificate_sha256=runtime['cache_certificate_sha256'],model=Path(runtime['model'])))
            require(derived==prepared['derived_cache'],'Certified original base cache changed')
            backend=integration.ClassMarginAuxBackend(adapter,configuration,model_path=Path(runtime['model']),contract=load_contract(Path(runtime['contract'])),seed=13,before_unit=before,cache={'dir':runtime['cache_dir']})
            resident[0]=backend;require(len(backend.trainable_signature())==504,'Exact504LoRA inventory required')
            backend.configure_auxiliary(prepared['companions_PRIVATE'],prepared['aux_token_plan_PRIVATE'],prepared['expected_image_sha256'],integration.ComputeBudget())
            return backend
        plan={'group_size':4,'max_tokens':256,'budget':{'maximum_attempted_groups':4,'text_visual_token_cap':9040,'input_tokens_min':213,'input_tokens_max':405}}
        def run():return driver.run(output=out,schedule=prepared['schedule_PRIVATE'],bindings=prepared['bindings'],backend_factory=factory,
            authorize=lambda context:before('authorize_'+context['phase']),parent_checkpoint=runtime['parent_checkpoint'],parent_receipt_sha256=runtime['parent_receipt_sha256'],
            resume=bool(spec.get('resume')),maximum_seconds=1200,config=prepared['configuration'],row_validator=validate_training_row,
            grader=make_grader(_json(runtime['calibration'])),ledger=TokenLedger(plan),input_tokens=lambda row:213 if row['ability']=='mri_sequence' else 405)
        result=inherited.run_with_release(run,resident,evidence)
        require(prepare(spec)==prepared,'Frozen training sources/artifacts changed during execution')
        driver.old.write_json(evidence/'complete_PRIVATE.json',{**result,'scope_sha256':inherited.scope_digest(scope)})
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('inspect','train'))
    for name in ('spec','output','scope','numeric','human','review'):parser.add_argument('--'+name,type=Path)
    for name in ('scope','numeric','human','review'):parser.add_argument('--expected-'+name+'-sha256')
    args=parser.parse_args();require(args.spec,'Specification required')
    if args.action=='train':return execute(args)
    require(args.output and not args.output.exists(),'Fresh CPU preparation output required')
    prepared=prepare(_json(args.spec));driver.old.write_json(args.output,prepared)
    print(json.dumps({'status':prepared['implementation_status'],'preparation_sha256':inherited.scope_digest(prepared),'model_loads':0,'optimizer_updates':0,'training_authorized':False}))

if __name__=='__main__':main()
