"""CPU-only saved-generation diagnosis. No parser/metric changes, generation, images or model tensors are loaded."""
import argparse,ast,hashlib,importlib.util,json,os,re,statistics,sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','RAYON_NUM_THREADS'):os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
W=Path(__file__).resolve().parents[1]
REWARD_SHA='411ab099edcfb45305630c675f7ca4870d6501e69dd084c9e1218ad2cc7a0ed8'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def reward():
 path=W/'src/mri_rl/vero_mri_reward.py';require(sha(path)==REWARD_SHA,'Frozen strict reward source changed');s=importlib.util.spec_from_file_location('saved_generation_reward',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def token_bounds(text,ids,tokenizer):
 prefixes=[tokenizer.decode(ids[:i],skip_special_tokens=True) for i in range(len(ids)+1)];exact=prefixes[-1]==text;consistent=exact and all(text.startswith(p) for p in prefixes) and all(len(a)<=len(b) for a,b in zip(prefixes,prefixes[1:]))
 tags={'think_open':'<think>','think_close':'</think>','answer_open':'<answer>','answer_close':'</answer>'};out={'offsets_verified':consistent}
 for name,tag in tags.items():out[name+'_complete_at_tokens']=next((i for i,p in enumerate(prefixes) if tag in p),None) if exact else None
 out['think_content_token_bounds']=None
 out['unclosed_think_token_lower_bound']=None
 if consistent and text.count('<think>')==1 and '</think>' not in text and '<answer>' not in text and out['think_open_complete_at_tokens'] is not None:out['unclosed_think_token_lower_bound']=len(ids)-out['think_open_complete_at_tokens']
 if consistent and text.count('<think>')==text.count('</think>')==1:
  a=text.index('<think>')+len('<think>');b=text.index('</think>')
  if b>=a:
   wholly=sum(a<=len(left)<len(right)<=b for left,right in zip(prefixes,prefixes[1:]));overlap=sum(max(a,len(left))<min(b,len(right)) for left,right in zip(prefixes,prefixes[1:]));out['think_content_token_bounds']={'wholly_inside':wholly,'any_overlap':overlap}
 return exact,out

def inspect_case(row,labels,truth,tokenizer,eos,max_tokens):
 lib=reward();text=row['response'];ids=row['token_ids'];answer,fmt=lib._answer_and_format(text);canonical={v.casefold():v for v in labels};strict_label=canonical.get(answer.casefold()) if fmt==1 else None;truth=truth.casefold()
 boxes=lib._boxed_contents(text);boxlabel=canonical.get(boxes[0].casefold()) if len(boxes)==1 else None
 span=re.search(r'<answer>(.*?)</answer>',text,re.S);spanlabel=None
 if span:
  inside=span.group(1).strip();innerboxes=lib._boxed_contents(inside);spanlabel=canonical.get((innerboxes[0] if len(innerboxes)==1 else inside).casefold())
 tags={tag:text.count(tag) for tag in ('<think>','</think>','<answer>','</answer>')};flags=[]
 if not text.strip():flags.append('empty_response')
 if any(n!=1 for n in tags.values()):flags.append('missing_or_repeated_tags')
 if all(n==1 for n in tags.values()) and not lib._FORMAT.fullmatch(text):flags.append('nonmatching_tag_structure_or_extraneous_text')
 think=re.search(r'<think>(.*?)</think>',text,re.S)
 if think and not think.group(1).strip():flags.append('empty_think')
 if span and not span.group(1).strip():flags.append('empty_answer')
 if span and len(lib._boxed_contents(span.group(1)))!=1:flags.append('missing_or_multiple_answer_boxes')
 if fmt==1 and strict_label is None:flags.append('noncanonical_boxed_answer')
 exact,bounds=token_bounds(text,ids,tokenizer)
 terminal=bool(ids and ids[-1] in eos);eos_positions=[i for i,v in enumerate(ids) if v in eos];finish=row.get('finish_reason')
 return {'strict_valid':strict_label is not None,'strict_correct':strict_label is not None and strict_label.casefold()==truth,'strict_label':strict_label,'reward_format':fmt,'format_flags':flags,'tag_counts':tags,'empty_answer':bool(span and not span.group(1).strip()),'finish_reason':finish,'saved_tokens':len(ids),'generation_tokens':row.get('generation_tokens'),'generation_count_matches_saved':row.get('generation_tokens')==len(ids),'hit_max_tokens_field':row.get('hit_max_tokens'),'confirmed_length_termination':finish=='length' and len(ids)==max_tokens,'terminal_eos_present':terminal,'eos_count':len(eos_positions),'eos_before_terminal':any(i<len(ids)-1 for i in eos_positions),'eos_stop_consistent':(finish=='stop' and terminal) or (finish=='length' and not terminal),'token_text_exact_match':exact,'tag_token_bounds':bounds,'boxed_anywhere_salvage_label':boxlabel,'boxed_anywhere_salvage_correct':boxlabel is not None and boxlabel.casefold()==truth,'answer_span_salvage_label':spanlabel,'answer_span_salvage_correct':spanlabel is not None and spanlabel.casefold()==truth}

def source_eos(spec,tokenizer):
 # Execute only this tiny standard-library helper's AST; never import mlx_vlm or mlx.
 source=Path(spec['installed_utils']['path']).read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_eos_token_ids');namespace={'List':list};exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned_resolve_eos','exec'),namespace)
 cfg=load(spec['generation_config']['path']);tokcfg=load(spec['tokenizer_config']['path']);eos_token=tokenizer.token_to_id(tokcfg['eos_token']);return set(namespace['resolve_eos_token_ids'](cfg['eos_token_id'],SimpleNamespace(eos_token_id=eos_token))),cfg

def summarize(cases):
 def subset(rows):
  invalid=[x for x in rows if not x['strict_valid']];length=[x for x in rows if x['confirmed_length_termination']]
  return {'records':len(rows),'strict_correct':sum(x['strict_correct'] for x in rows),'strict_invalid':len(invalid),'finish_reason_counts':dict(Counter(x['finish_reason'] for x in rows)),'length_terminated':len(length),'length_terminated_invalid':sum(not x['strict_valid'] for x in length),'invalid_finish_reason_counts':dict(Counter(x['finish_reason'] for x in invalid)),'invalid_format_flags':dict(Counter(flag for x in invalid for flag in x['format_flags'])),'invalid_terminal_eos_present':sum(x['terminal_eos_present'] for x in invalid),'invalid_no_answer_open_tag':sum(x['tag_counts']['<answer>']==0 for x in invalid),'invalid_think_not_closed':sum(x['tag_counts']['<think>']>0 and x['tag_counts']['</think>']==0 for x in invalid),'empty_answer_records':sum(x['empty_answer'] for x in rows),'stop_records_with_terminal_eos':sum(x['finish_reason']=='stop' and x['terminal_eos_present'] for x in rows),'eos_consistency_failures':sum(not x['eos_stop_consistent'] for x in rows),'generation_count_mismatches':sum(not x['generation_count_matches_saved'] for x in rows),'exact_token_text_matches':sum(x['token_text_exact_match'] for x in rows),'descriptive_salvage_NOT_STRICT_ACCURACY':{'one_canonical_box_anywhere_correct':sum(x['boxed_anywhere_salvage_correct'] for x in rows),'canonical_complete_answer_span_correct':sum(x['answer_span_salvage_correct'] for x in rows),'invalid_with_correct_one_box_anywhere':sum(x['boxed_anywhere_salvage_correct'] for x in invalid),'invalid_with_correct_complete_answer_span':sum(x['answer_span_salvage_correct'] for x in invalid)},'generated_token_counts':{'min':min(x['saved_tokens'] for x in rows),'median':statistics.median(x['saved_tokens'] for x in rows),'max':max(x['saved_tokens'] for x in rows)},'unclosed_think_token_lower_bounds':{'observed':sum(x['tag_token_bounds']['unclosed_think_token_lower_bound'] is not None for x in rows),'min':min([x['tag_token_bounds']['unclosed_think_token_lower_bound'] for x in rows if x['tag_token_bounds']['unclosed_think_token_lower_bound'] is not None],default=None),'max':max([x['tag_token_bounds']['unclosed_think_token_lower_bound'] for x in rows if x['tag_token_bounds']['unclosed_think_token_lower_bound'] is not None],default=None)},'token_offset_verified_records':sum(x['tag_token_bounds']['offsets_verified'] for x in rows),'think_close_completed_at_tokens':{'observed':sum(x['tag_token_bounds']['think_close_complete_at_tokens'] is not None for x in rows),'median_of_observed':statistics.median([x['tag_token_bounds']['think_close_complete_at_tokens'] for x in rows if x['tag_token_bounds']['think_close_complete_at_tokens'] is not None]) if any(x['tag_token_bounds']['think_close_complete_at_tokens'] is not None for x in rows) else None}}
 return {task:subset([x for x in cases if x['task']==task]) for task in ('mri_sequence','mri_visit_correspondence')}

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--expected-spec-sha256',required=True);p.add_argument('--private-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(sha(a.spec)==a.expected_spec_sha256,'External saved-evidence spec changed');spec=load(a.spec)
 for key,item in spec['artifacts'].items():require(sha(item['path'])==item['sha256'],'Frozen diagnosis artifact changed: '+key)
 from tokenizers import Tokenizer
 tok=Tokenizer.from_file(spec['artifacts']['tokenizer']['path']);eos,cfg=source_eos({k:spec['artifacts'][k] for k in ('installed_utils','generation_config','tokenizer_config')},tok);allcases={};summaries={}
 for checkpoint,bundle in spec['checkpoints'].items():
  for key,item in bundle.items():require(sha(item['path'])==item['sha256'],'Closed generation artifact changed')
  scope=load(bundle['scope']['path']);exitw=load(bundle['exit_receipt']['path']);complete=load(bundle['eval_receipt']['path']);require(exitw['exit_code']==0 and exitw['raw_log_sha256']==bundle['closed_log']['sha256'] and exitw['scope_file_sha256']==bundle['scope']['sha256'],'Actual completed generation witness differs');require(complete['native_release_verified'] is True and complete['greedy_sha256']==bundle['greedy_output']['sha256'],'Closed greedy output receipt differs')
  rows=[json.loads(x) for x in Path(bundle['greedy_output']['path']).read_text().splitlines() if x.strip()];require(len(rows)==40 and len({x['record_id'] for x in rows})==40,'Exactly40 existing saved outputs required');cases=[]
  for index,row in enumerate(rows):
   require(row['run_identity']==scope['run_identity'] and row['weight_updates']==0,'Saved checkpoint identity differs');labels=('FLAIR','contrast-enhanced T1') if row['category']=='mri_sequence' else ('different','same');case=inspect_case(row,labels,row['visible_target_PRIVATE'],tok,eos,256);case.update(index_PRIVATE=index,record_id_PRIVATE=row['record_id'],task=row['category'],response_sha256=hashlib.sha256(row['response'].encode()).hexdigest());cases.append(case)
  allcases[checkpoint]=cases;summaries[checkpoint]=summarize(cases)
 keys=list(allcases);left={c['record_id_PRIVATE']:c for c in allcases[keys[0]]};right={c['record_id_PRIVATE']:c for c in allcases[keys[1]]};require(set(left)==set(right),'Parent/child saved cohorts differ');transitions={}
 for task in ('mri_sequence','mri_visit_correspondence'):
  pairs=[(v,right[k]) for k,v in left.items() if v['task']==task];transitions[task]={'invalid_in_both':sum(not a['strict_valid'] and not b['strict_valid'] for a,b in pairs),'parent_invalid_child_valid':sum(not a['strict_valid'] and b['strict_valid'] for a,b in pairs),'parent_valid_child_invalid':sum(a['strict_valid'] and not b['strict_valid'] for a,b in pairs),'incorrect_to_correct':sum(not a['strict_correct'] and b['strict_correct'] for a,b in pairs),'correct_to_incorrect':sum(a['strict_correct'] and not b['strict_correct'] for a,b in pairs)}
 result={'schema':'saved-generation-format-diagnosis-v1','evidence_tier':'OBSERVED_SAVED_OUTPUTS_AND_INSTALLED_SOURCE','checkpoints':summaries,'matched_record_transitions':transitions,'source_evidence_references':{'installed_dispatch':'mlx_vlm/generate/dispatch.py:1088-1096 append-before-stop;1112 length;1147 returned token_ids','configured_EOS_merge_and_union':'mlx_vlm/utils.py:89-104,2129-2149','native_call':'mri_rl/mlx_continuous_gspo.py:251-270 temperature0/max_tokens argument/enable_thinkingFalse'},'source_derived_configured_EOS_ids':sorted(eos),'native_generation_contract':{'temperature':0,'max_tokens':256,'enable_thinking_template_argument':False,'custom_eos_or_stopping_criteria_passed':False},'EOS_semantics_from_bound_source':{'generated_token_ids_include_sampled_EOS':True,'EOS_not_added_to_visible_detokenized_text':True,'finish_reason_stop_comes_from_stopping_criteria':True,'finish_reason_length_comes_from_exhausted_generate_step_budget':True,'hit_max_tokens_alone_not_truncation_proof':'An EOS at token256 is stop even if len>=max_tokens.'},'strict_parser_changed':False,'existing_candidate_accuracy_reference':'Both checkpoints15/24 for mean/raw/first; candidate readout is not greedy decoding accuracy.','salvage_limits':'Descriptive extraction from existing text only; it is not the frozen strict metric, new model output, or evidence of an accuracy improvement.','additional_processor_stop_ids':'Not instantiated live; all observed stop-token IDs are checked against source-derived configured/tokenizer IDs.','artifact_sha256':{k:v['sha256'] for k,v in spec['artifacts'].items()},'checkpoint_artifact_sha256':{c:{k:v['sha256'] for k,v in b.items()} for c,b in spec['checkpoints'].items()},'script_sha256':sha(__file__),'spec_sha256':sha(a.spec),'native_models_loaded':0,'sealed_final100_used':False,'limitations':['Think/token marker bounds are reported only when raw-token decoding exactly reproduces the stored visible text and prefix offsets are consistent.','Length termination proves the answer was not completed within this cap; it does not prove a longer cap would yield the correct answer.','A correct candidate preference or salvaged label cannot replace strict generation performance.','No native generation, tensor replay, new decoder setting or training was run.']}
 require(a.private_output.absolute().is_relative_to(W/'local_data') and a.output.absolute().is_relative_to(W/'results'),'Private/public destinations required')
 for dest,value in ((a.private_output,allcases),(a.output,result)):
  with dest.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'aggregate_sha256':sha(a.output),'native_models_loaded':0,'checkpoints':summaries}))
if __name__=='__main__':main()
