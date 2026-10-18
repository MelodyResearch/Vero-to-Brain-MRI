"""Saved-only class/format/generation-cost supplement; no inference or bootstrap.
The unchanged canonical95d9 result supplies reviewed raw-file/source bindings.
Counts below preserve its EXACT native parser and separately report format==1.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,statistics,sys
from collections import Counter
from pathlib import Path
W=Path(__file__).resolve().parents[1]
CANONICAL_SOURCE='95d9d2b0c4709498c103334b3d95c6a9634d8877a7e77ed60ddeb6045a5d4e4a'
PARSER_SOURCE='411ab099edcfb45305630c675f7ca4870d6501e69dd084c9e1218ad2cc7a0ed8'
LABELS={'mri_sequence':('FLAIR','contrast-enhanced T1'),'mri_visit_correspondence':('different','same')}
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def auth(d):require(sha(d['path'])==d['sha256'],'Externally supplied saved-file digest differs');return load(d['path'])
def module(p,name,expected):
 require(sha(p)==expected,'Frozen source differs');s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def recount(reference,outputs,answer_and_format):
 require(len(reference)==len({r['id'] for r in reference}) and len(outputs)==len({r['record_id'] for r in outputs}) and {r['id'] for r in reference}=={r['record_id'] for r in outputs},'Exact unique saved-output coverage required');by={r['record_id']:r for r in outputs};result={}
 for task,labels in LABELS.items():
  rows=[r for r in reference if r['ability']==task];require(rows,'Both frozen tasks required');classes={l:{'questions':0,'native_correct':0,'full_format_correct':0} for l in labels};conf={l:{p:0 for p in (*labels,'INVALID')} for l in labels};strictconf={l:{p:0 for p in (*labels,'INVALID')} for l in labels};formats=Counter();finish=Counter();invalid=0;lengths=[];times=[];caps=0;truncated=0
  for r in rows:
   o=by[r['id']];truth=r['reward_model']['ground_truth'];require(truth in labels and o['category']==task and isinstance(o['response'],str),'Reference/output task or target differs');answer,fmt=answer_and_format(o['response']);normalized=next((l for l in labels if l.lower()==answer.lower()),None);valid=fmt>0 and normalized is not None;hit=bool(valid and normalized==truth);full=bool(hit and fmt==1)
   classes[truth]['questions']+=1;classes[truth]['native_correct']+=hit;classes[truth]['full_format_correct']+=full;conf[truth][normalized if valid else 'INVALID']+=1;strictconf[truth][normalized if valid and fmt==1 else 'INVALID']+=1;invalid+=not valid;formats[str(fmt)]+=1
   ids=o.get('token_ids');cap=o.get('max_tokens');require(isinstance(ids,list) and all(type(x) is int and x>=0 for x in ids) and type(cap) is int and cap>0 and len(ids)<=cap,'Actual saved token/cap metadata malformed');require('generation_tokens' not in o or o['generation_tokens']==len(ids),'Saved generation count differs');lengths.append(len(ids));finish[o.get('finish_reason','NOT_RECORDED')]+=1;caps+=o.get('hit_max_tokens') is True;truncated+=o.get('finish_reason')=='length' and len(ids)==cap
   if 'elapsed_seconds' in o:
    t=o['elapsed_seconds'];require(type(t) in (int,float) and math.isfinite(t) and t>=0,'Nonfinite/negative saved generation-call time');times.append(t)
  for c in classes.values():
   for kind in ('native','full_format'):c[kind+'_recall']=c[kind+'_correct']/c['questions'] if c['questions'] else None
  n=len(rows);result[task]={'questions':n,'native_correct':sum(c['native_correct'] for c in classes.values()),'full_format_correct':sum(c['full_format_correct'] for c in classes.values()),'native_question_accuracy':sum(c['native_correct'] for c in classes.values())/n,'full_format_question_accuracy':sum(c['full_format_correct'] for c in classes.values())/n,'class_counts':classes,'native_confusion':conf,'full_format_confusion':strictconf,'invalid_answers':invalid,'format_score_counts':dict(formats),'format_half_records':formats['0.5'],'finish_reason_counts':dict(finish),'literal_hit_max_tokens_records':caps,'length_truncated_records':truncated,'tokens':{'records_with_actual_token_ids':n,'sum':sum(lengths),'min':min(lengths),'median':statistics.median(lengths),'max':max(lengths)},'call_time':{'records_with_actual_elapsed':len(times),'records_without_elapsed':n-len(times),'actual_elapsed_seconds_sum':math.fsum(times) if times else None,'actual_elapsed_seconds_mean':statistics.mean(times) if times else None},'aggregation':'Question/class counts only; canonical patient/cycle estimates, seed summaries and2000-bootstrap uncertainty remain in the unchanged main matrix.'}
 return result

def run(inputs):
 require(inputs.get('schema')=='original22-saved-recount-input-v1' and inputs.get('source_sha256')==sha(__file__),'Exact supplementary input/source required');spec=auth(inputs['actual_analysis_spec']);canonical=auth(inputs['canonical_result']);e=auth(inputs['canonical_exit']);review=auth(inputs['completion_spec_review']);require(sha(inputs['canonical_log']['path'])==inputs['canonical_log']['sha256']==e['closed_stdout_sha256'],'Canonical closed-log binding differs')
 require(e['exit_code']==0 and e['output_sha256']==inputs['canonical_result']['sha256'] and e['spec_sha256']==inputs['actual_analysis_spec']['sha256'] and e['analyzer_sha256']==CANONICAL_SOURCE and e['approval_file_sha256']==inputs['completion_spec_review']['sha256'],'Canonical actual execution witness differs');require(review['verdict'] in {'PASS','PASS_WITH_LIMITATIONS'} and review['approved_analysis_spec_file_sha256']==inputs['actual_analysis_spec']['sha256'] and review['approved_analyzer_source_sha256']==CANONICAL_SOURCE,'Independent exact completion-spec review differs')
 require(canonical['status']=='ALL22_ACTUAL_OUTPUTS_AUTHENTICATED_AND_SCORED' and canonical['analyzer_sha256']==CANONICAL_SOURCE and canonical['bootstrap_replicates']==2000 and canonical['distinct_model_inference_runs']==22,'Existing complete canonical matrix required');require(canonical['input_evidence_sha256']=={k:spec[k]['sha256'] for k in ('bundle','scope_set','native_review_set','release_review')},'Main-result/spec evidence differs')
 primary=Path(spec['primary_root']);binding=module(W/'scripts/analyze_original22_saved_final_v1.py','saved_original22_authentication',CANONICAL_SOURCE)
 for n,h in spec['source_sha256'].items():require(sha(primary/'src/mri_rl'/n)==h,'Existing canonical source changed')
 b=auth(spec['bundle']);ss=auth(spec['scope_set']);rs=auth(spec['native_review_set']);reviews={r['checkpoint_id']:r['review'] for r in rs['reviews']};completed=binding.verify_all_completed(ss['runs'],spec['completed_runs'],reviews)
 source=Path(b['projection_directory'])/'final_test.verl.jsonl';require(sha(source)==canonical['source_verl_manifest_sha256'],'Canonical frozen reference differs');reference=[json.loads(x) for x in source.read_text().splitlines() if x.strip()];require(len(reference)==100,'Frozen complete100 reference required');sys.path.insert(0,str(primary/'src'));from mri_rl.two_task_evaluation import project_records
 projected={r['record_id']:r for r in project_records(reference,load(b['contract_path']))};parser=module(primary/'src/mri_rl/vero_mri_reward.py','saved_final_exact_parser',PARSER_SOURCE)._answer_and_format;models={}
 for r in ss['runs']:
  key=r['checkpoint_id'];scope=auth(r['scope']);proofkeys=['B28:'+str(s) for s in (7,11,13)] if key=='B28' else [('same_C_SFT:'+key.split(':')[1]) if key.startswith('sft:') else key.removeprefix('rl:')]
  for pk in proofkeys:
   p=canonical['file_binding_proofs'][pk];require(p['output_sha256']==completed[key]['output']['sha256'] and p['adapter_checkpoint_sha256']==r['expected_config']['adapter_checkpoint_sha256'],'Canonical raw-output/checkpoint proof differs')
  outputs=[json.loads(x) for x in Path(r['output']).read_text().splitlines() if x.strip()]
  for row in outputs:
   target=projected[row['record_id']];require(all(row.get(k)==v for k,v in r['expected_config'].items()) and row['prompt']==target['prompt'] and row['image_sha256']==scope['image_sha256'][row['record_id']] and row['engine']=='continuous_rl_dev_eval_native_serial_v1','Canonical row/config/prompt/image proof differs')
  models[key]={'tasks':recount(reference,outputs,parser),'output_sha256':completed[key]['output']['sha256']}
 require(len(models)==22,'Exact22 unique saved models required')
 return {'schema':'original22-final-class-format-cost-supplement-v1','status':'ACTUAL_SAVED_OUTPUTS_RECOUNTED','models':models,'unique_model_runs':22,'unique_model_task_count_tables':44,'baseline_model_runs':1,'canonical_matrix_task_cells':48,'canonical_result_sha256':inputs['canonical_result']['sha256'],'actual_analysis_spec_sha256':inputs['actual_analysis_spec']['sha256'],'source_sha256':sha(__file__),'parser_source_sha256':PARSER_SOURCE,'native_models_loaded':0,'additional_forwards':0,'bootstrap_repeated':False,'canonical_actual_test_execution_false_semantics':'The reused adapter flag describes its saved-only analysis operation, not absence of native evaluation. Exact all22 exit0/closed-log/output witnesses and canonical execution receipt bind already executed native final100 outputs.','private_case_identifiers_or_text_included':False,'metric_contract':'Existing _answer_and_format: native correctness requires format>0 and exact allowed label ignoring case; full-format correctness additionally requires format==1. No parser or threshold change.','cost_contract':'Actual stored token_ids and elapsed_seconds only. Timed generate_record spans include its prompt/image/hash/cache work and inside-call resource callbacks or HOLD waits; initial model loading and outer JSON write/flush overhead are excluded. These are elapsed call times, not pure GPU time. Missing elapsed data remains unavailable. EOS stop at the cap is not classified as length truncation.','evidence_tier':'OBSERVED','research_tag':'PRIMARY_VERO_TRANSFER','generalization_limitations':'Question/class recount is descriptive. Canonical patient/cycle and between-seed limitations remain unchanged; no clinical or global cross-source independence claim.'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--expected-spec-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(sha(a.spec)==a.expected_spec_sha256,'Independent supplementary input digest differs');require(a.output.resolve().is_relative_to(W/'results'),'Aggregate destination required');result=run(load(a.spec))
 with a.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':result['status'],'aggregate_sha256':sha(a.output),'unique_models':22,'new_model_calls':0}))
if __name__=='__main__':main()
