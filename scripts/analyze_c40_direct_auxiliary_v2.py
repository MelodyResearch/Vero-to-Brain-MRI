"""CPU-only authenticated auxiliary accuracy + all-three-score comparison.
Original nine B28 records stay untouched. No sealed test, native model or p-values.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math
from collections import defaultdict
from pathlib import Path
from statistics import mean
WORKTREE=Path(__file__).resolve().parents[1]
METHODS=('greedy','mean','raw','first_label')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def authenticated(p,digest):require(sha(p)==digest,'Externally authenticated evidence hash differs');return load(p)
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def accuracy_summary(rows,pairs,labels,parse_answer):
 result={'tasks':{},'equal_task_unit_macro':{},'worst_task_accuracy':{}}
 for task,vocabulary in labels.items():
  selected=[r for r in rows.values() if r['category']==task and r['condition']=='real_image'];require(selected,'Real displayed-image examples required')
  units={p[key]:p['unit'] for p in pairs if p['task']==task for key in ('negative_id','positive_id')}
  require(set(units)=={r['record_id'] for r in selected},'Exact paired units required')
  task_result={}
  for method in METHODS:
   counts={truth:{p:0 for p in [*vocabulary,'TIE','INVALID']} for truth in vocabulary};per_unit=defaultdict(list)
   for row in selected:
    truth=row['visible_target_PRIVATE'];require(truth in vocabulary,'Displayed-image truth missing')
    if method=='greedy':
     answer,fmt=parse_answer(row['response']);lookup={label.lower():label for label in vocabulary};pred=lookup.get(answer.lower(),'INVALID') if fmt>0 else 'INVALID'
    elif method=='first_label':
     margin=row['prefix_oracle_first_logps'][1]-row['prefix_oracle_first_logps'][0];pred=vocabulary[1] if margin>0 else vocabulary[0] if margin<0 else 'TIE'
    else:pred=row['comparisons'][method+'_preferred']
    require(pred in counts[truth],'Unknown predicted category');counts[truth][pred]+=1;per_unit[units[row['record_id']]].append(int(truth==pred))
   n=len(selected);correct=sum(counts[label][label] for label in vocabulary)
   task_result[method]={'n':n,'correct':correct,'accuracy':correct/n,'class_recalls':{label:counts[label][label]/sum(counts[label].values()) for label in vocabulary},
    'confusion':counts,'independent_units':len(per_unit),'unit_macro_accuracy':mean(mean(v) for v in per_unit.values())}
  result['tasks'][task]=task_result
 for method in METHODS:
  result['equal_task_unit_macro'][method]=mean(t[method]['unit_macro_accuracy'] for t in result['tasks'].values())
  result['worst_task_accuracy'][method]=min(t[method]['accuracy'] for t in result['tasks'].values())
 return result

def validate_shape_oracle_fields(row,labels):
 if row.get('legacy_oracle_gate')!='FAIL':return
 evidence=row['legacy_oracle_evidence_PRIVATE'];control=row['shape_matched_control'];scores=row['candidate_scores']
 first=[scores[label]['first_label_candidate_logps'] for label in labels]
 require(first==evidence['first'],'Captured first-label candidate vectors changed')
 prefixes=[scores[label]['target_token_logps'][:len(scores[label]['partition']['prefix_ids'])] for label in labels]
 require(prefixes==evidence['prefix_scores'] and len(prefixes[0])==len(prefixes[1]),'Captured common prefixes changed')
 legacy_errors=[abs(a-b) for a,b in zip(*prefixes)]+[abs(a-b) for vector in first for a,b in zip(vector,evidence['oracle'])]
 padded_errors=[abs(a-b) for vector in first for a,b in zip(vector,control['first_label_candidate_logps'])]
 require(legacy_errors==evidence['errors'] and max(legacy_errors)==row['legacy_oracle_max_disagreement']>.002,'Legacy FAIL error arithmetic changed')
 require(padded_errors==control['candidate_first_label_abs_errors'] and max(padded_errors)==control['maximum_candidate_first_label_error']<=.002,'Padded oracle error arithmetic changed')
 require(row['prefix_oracle_first_logps']==control['first_label_candidate_logps'],'Auxiliary first-label oracle differs')

def validate_partitions(api,row,record,tokenizer):
 labels=api.LABELS[record['category']];partition,_=api.bound_candidate_functions(labels)
 P=row['legacy_oracle_evidence_PRIVATE']['P'] if row.get('legacy_oracle_gate')=='FAIL' else len(row['prompt_ids'])
 for label in labels:
  candidate=row['candidate_scores'][label];require(candidate['partition']==partition(tokenizer,label,prompt_length=P),'Actual target token IDs or causal alignment differs')
  summary=api.al.summarize_tokens(candidate['target_token_logps'],candidate['partition']);require(all(candidate[k]==v for k,v in summary.items()),'Actual token-derived scores differ')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True)
 for cp in ('b28','u7'):
  for field in ('scope','output','receipt'):p.add_argument('--'+cp+'-'+field,type=Path,required=True);p.add_argument('--'+cp+'-'+field+'-sha256',required=True)
 p.add_argument('--private-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 runner=module(WORKTREE/'scripts/run_c40_direct_auxiliary_v2.py','aux_runner_analysis')
 primary=a.primary_root.resolve();early=module(WORKTREE/'scripts/run_c40_early_visual_probe_v1.py','early_aux_analysis');api=early.primary_api(primary)
 analysis_path=primary/'scripts/analyze_c40_visual_probe_results.py';require(sha(analysis_path)=='89ef601afd3286ba95aa31dd490ea5f8c7036ee01e3ae32972d2ddc9b5782b59','Original pure score analysis changed')
 old=module(analysis_path,'original_c40_pure_analysis')
 from tokenizers import Tokenizer
 scopes={};evidence={};all_rows={};plan=None
 for cp,arg in (('B28','b28'),('U_raw_seed7','u7')):
  scope=authenticated(getattr(a,arg+'_scope'),getattr(a,arg+'_scope_sha256'));output=getattr(a,arg+'_output');receipt=authenticated(getattr(a,arg+'_receipt'),getattr(a,arg+'_receipt_sha256'))
  require(sha(output)==getattr(a,arg+'_output_sha256')==receipt['output_sha256'],'Completed output hash differs')
  require(scope['kind']=='C40_direct_idle_auxiliary_v2' and scope['checkpoint']==cp and scope['runner_sha256']==sha(runner.__file__) and scope['sources']==api.source_hashes() and scope['weight_updates']==0,'Reviewed auxiliary source/config differs')
  require(scope['output']==str(output.resolve()) and receipt['scope_sha256']==runner.h(scope) and receipt['status']=='ACTUAL_DIRECT_AUXILIARY_COMPLETED' and receipt['native_children_released'] is True and receipt['adapter_tensors_exact']==504 and receipt['optimizer_step']==0 and receipt['weight_updates']==0,'Actual completed read-only receipt differs')
  current=api.validate_plan(Path(scope['original_plan_path']));require(sha(scope['original_plan_path'])==scope['original_plan_sha256'] and scope['adapter']==current['checkpoints'][cp] and scope['derived_cache']==current['derived_cache'],'Frozen plan/adapter/cache differs')
  require(scope['indices']==runner.indices_for(cp) and runner.completed(output,scope,current['records_PRIVATE'])==len(scope['indices']),'Exact finite native coverage missing')
  raw=[json.loads(x) for x in output.read_text().splitlines()]
  if cp=='B28':
   require(sha(scope['legacy_prefix'])==scope['legacy_prefix_sha256'] and sha(scope['legacy_scope'])==scope['legacy_scope_sha256'],'Immutable original nine evidence changed')
   prior=load(scope['legacy_scope']);prefix=[json.loads(x) for x in Path(scope['legacy_prefix']).read_text().splitlines()];require(len(prefix)==9,'Nine original B28 rows required')
   for index,row in enumerate(prefix):
    require(row['run_identity']==prior['run_identity'] and row['checkpoint']==cp and row['adapter_sha256']==scope['adapter']['sha256'] and all(row.get(k)==v for k,v in current['records_PRIVATE'][index].items()),'Original nine provenance differs');api.validate_numeric_route(row)
   raw=prefix+raw
  require(len(raw)==10 and len({r['record_id'] for r in raw})==10,'Exact full10 coverage missing')
  tokenizer_raw=Tokenizer.from_file(str(Path(current['model_path'])/'tokenizer.json'))
  class TokenizerAdapter:
   def encode(self,text,add_special_tokens=False):return tokenizer_raw.encode(text,add_special_tokens=add_special_tokens).ids
  saved,saved_sha=api.saved_predictions(cp);require(saved_sha==scope['saved_real_outputs_sha256'],'Authenticated real-answer source differs')
  for row,record in zip(raw,current['records_PRIVATE']):
   runner.validate_auxiliary(api,{**row,'legacy_oracle_gate':row.get('legacy_oracle_gate','PASS')},record);validate_partitions(api,row,record,TokenizerAdapter());validate_shape_oracle_fields(row,api.LABELS[record['category']])
   require(row['new_greedy_generation'] is (record['condition']=='no_image'),'Generation reuse flag differs')
   if record['condition']=='real_image':require(row['response']==saved[record['source_record_id']]['response'],'Saved real-image answer changed')
  all_rows[cp]={r['record_id']:r for r in raw};scopes[cp]=scope;plan=current;evidence[cp]={f:getattr(a,arg+'_'+f+'_sha256') for f in ('scope','output','receipt')}
 require(scopes['B28']['original_plan_sha256']==scopes['U_raw_seed7']['original_plan_sha256'] and scopes['B28']['sources']==scopes['U_raw_seed7']['sources'] and scopes['B28']['derived_cache']==scopes['U_raw_seed7']['derived_cache'] and scopes['B28']['legacy_prefix_sha256']==scopes['U_raw_seed7']['legacy_prefix_sha256'],'Paired scientific contracts differ')
 pairs=old.private_pairs(plan);margins,private=old.compare_checkpoints(all_rows,pairs)
 from mri_rl.vero_mri_reward import _answer_and_format
 accuracy={cp:accuracy_summary(rows,pairs,api.LABELS,_answer_and_format) for cp,rows in all_rows.items()}
 result={'schema':'C40-direct-idle-auxiliary-accuracy-v2','status':'ACTUAL_AUTHENTICATED_AUXILIARY20_ANALYZED','input_evidence':evidence,'analyzer_sha256':sha(__file__),
  'accuracy':accuracy,'margins':margins,'score_definitions':{'primary':'mean label-token log probability','secondary':['raw label-token sum','independent first-label token']},
  'accuracy_truth':'Always displayed-image visible_target_PRIVATE, including paired substitutions. Null examples excluded from accuracy.',
  'blank_image_controls':'NOT_RUN','new_real_greedy_generations':0,'weight_updates':0,'sealed_test_used':False,'p_values_computed':False,
  'frozen_experiment_completed':False,'legacy_failure_counts':{cp:sum(r.get('legacy_oracle_gate')=='FAIL' for r in rows.values()) for cp,rows in all_rows.items()},
  'limitations':['Previously examined internal-development subset: two sequence people and one matched Dallas cycle; no independent-patient or clinical-generalization inference.',
   'Real greedy answers are authenticated saved matching-image answers, not newly generated. Conditional margins are not calibrated class probabilities.',
   'The versioned shape oracle preserves the legacy failure and original candidate scores. Its support is narrow and does not identify a kernel cause.',
   'Changed accuracy or paired separation on this subset does not prove robust brain-specific grounding or a training causal mechanism.'],
  'accuracy_change_U7_minus_B28':{task:{method:accuracy['U_raw_seed7']['tasks'][task][method]['accuracy']-accuracy['B28']['tasks'][task][method]['accuracy'] for method in METHODS} for task in api.LABELS}}
 require(a.private_output.resolve().is_relative_to(WORKTREE/'local_data') and a.output.resolve().is_relative_to(WORKTREE/'results'),'Private detail and safe aggregate destinations required')
 early.save(a.private_output,{'input_evidence':evidence,'margins_PRIVATE':private});result['private_detail_sha256']=sha(a.private_output);early.save(a.output,result)
 print(json.dumps({'status':result['status'],'output_sha256':sha(a.output),'private_detail_sha256':sha(a.private_output),'models_launched':0}))
if __name__=='__main__':main()
