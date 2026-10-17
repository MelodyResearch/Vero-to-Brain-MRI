"""Independent bounded CPU arithmetic on completed EARLY captures only.

No helper/analyzer imports, model execution, process signalling or interpretation
threshold. Root supplies a SHA-authenticated evidence index after actual exits.
"""
from pathlib import Path
import argparse,array,hashlib,json,math,re,struct,sys
from collections import defaultdict
LABELS={'mri_sequence':('FLAIR','contrast-enhanced T1'),'mri_visit_correspondence':('different','same')}
CP=('B28','U_raw_seed7');KINDS=('mean','raw','first_label')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):return json.loads(Path(path).read_text())
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def bound(entry):
 require(sha(entry['path'])==entry['sha256'],'External evidence SHA differs');return Path(entry['path'])
def close(a,b,label,tol=1e-10):require(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=tol,abs_tol=tol),'Independent arithmetic differs: '+label)
def mean(xs):return math.fsum(xs)/len(xs)
def summary(xs):return {'n':len(xs),'mean':mean(xs),'minimum':min(xs),'maximum':max(xs)}
def record_margins(row):
 labels=LABELS[row['category']];scores=[]
 require(row['candidate_order']==list(labels),'Candidate order differs')
 for label in labels:
  c=row['candidate_scores'][label];p=c['partition'];n=len(p['prefix_ids']);k=len(p['label_ids']);v=c['target_token_logps']
  require(k>0 and len(v)==len(p['target_ids']) and all(math.isfinite(x) and x<=0 for x in v),'Invalid selected-token vector')
  subset=v[n:n+k];require(len(subset)==k,'Label slice out of range');raw=math.fsum(subset);avg=raw/k
  close(raw,c['label_logp_sum'],'candidate raw');close(avg,c['label_logp_mean'],'candidate mean')
  scores.append({'raw':raw,'mean':avg,'candidate_first':subset[0]})
 oracle=row['prefix_oracle_first_logps'];require(len(oracle)==2 and all(math.isfinite(x) and x<=0 for x in oracle),'Invalid first-token oracle')
 prefix_error=max(abs(scores[i]['candidate_first']-oracle[i]) for i in range(2));require(prefix_error<=.002,'Frozen candidate-prefix oracle differs')
 result={k:scores[1][k]-scores[0][k] for k in ('raw','mean')};result['first_label']=oracle[1]-oracle[0]
 suffix='t1_minus_flair' if row['category']=='mri_sequence' else 'same_minus_different'
 for k in ('raw','mean'):close(result[k],row['comparisons'][k+'_'+suffix],'stored margin')
 return result,prefix_error

def pair_index(plan):
 source={r['id']:r for r in map(json.loads,Path(plan['source_manifest']).read_text().splitlines())};groups=defaultdict(dict)
 for r in plan['records_PRIVATE']:
  if r['condition']!='real_image':continue
  s=source[r['source_record_id']];e=s['extra_info'];task=r['category'];label=s['reward_model']['ground_truth'];require(label in LABELS[task] and label==r['visible_target_PRIVATE'],'Frozen pair label differs')
  if task=='mri_sequence':
   match=re.fullmatch(re.escape(e['patient_key'])+r'_z(\d{3})_(flair|t1ce)_sequence',s['id']);require(match is not None,'Sequence pairing metadata malformed')
   key=(task,e['patient_key'],match[1]);unit=e['patient_key']
  else:key=(task,e['statistical_cluster_key'],e['query_source_sha256']);unit=e['statistical_cluster_key']
  require(label not in groups[key],'Duplicate pair label');groups[key][label]=(r['record_id'],unit)
 result=[]
 for key,by in groups.items():
  require(set(by)==set(LABELS[key[0]]),'Incomplete positive-negative pair');a,b=(by[x] for x in LABELS[key[0]])
  result.append({'task':key[0],'unit':a[1],'negative_id':a[0],'positive_id':b[0]})
 require(len(result)==4,'Frozen four-pair coverage differs');return result

def checkpoint_numbers(rows,pairs):
 margins={r['record_id']:record_margins(r)[0] for r in rows};result={}
 for task,labels in LABELS.items():
  null=[r for r in rows if r['category']==task and r['condition']=='no_image'];real=[r for r in rows if r['category']==task and r['condition']=='real_image'];require(len(null)==1 and len(real)==4,'Incomplete task/image conditions')
  result[task]={};tp=[p for p in pairs if p['task']==task]
  for k in KINDS:
   baseline=margins[null[0]['record_id']][k];sep=defaultdict(list);effects=defaultdict(list)
   for p in tp:
    low=margins[p['negative_id']][k];high=margins[p['positive_id']][k];sep[p['unit']].append(high-low);effects[p['unit']].extend((low-baseline,high-baseline))
   result[task][k]={'no_image_margin':baseline,'real_image_margins':summary([margins[r['record_id']][k] for r in real]),'real_minus_no_image_effects':summary([margins[r['record_id']][k]-baseline for r in real]),
    'unit_averaged_paired_separation':mean([mean(v) for v in sep.values()]),'unit_averaged_real_minus_no_image_effect':mean([mean(v) for v in effects.values()]),'pair_count':len(tp),'independent_units':len(sep)}
 return result,margins

def safetensors_header(path):
 with Path(path).open('rb') as f:
  first=f.read(8);require(len(first)==8,'Torn tensor header');n=struct.unpack('<Q',first)[0];require(0<n<=4*1024**2,'Unbounded tensor header');raw=f.read(n);require(len(raw)==n,'Truncated tensor metadata');h=json.loads(raw)
 h.pop('__metadata__',None);base=8+n;size=Path(path).stat().st_size;spans=[]
 for t in h.values():
  require(t['dtype']=='F32' and all(type(d) is int and d>0 for d in t['shape']),'Expected finite-shape F32 gradients');a,b=t['data_offsets'];require(type(a) is int and type(b) is int and 0<=a<b and b-a==4*math.prod(t['shape']) and base+b<=size,'Invalid tensor offsets');spans.append((a,b))
 spans.sort();require(spans and spans[0][0]==0 and all(a[1]==b[0] for a,b in zip(spans,spans[1:])) and base+spans[-1][1]==size,'Tensor data gaps/overlap/trailing bytes')
 return h,base

def gradient_metrics(real,zero,expected_tensors=504):
 rh,rb=safetensors_header(real);zh,zb=safetensors_header(zero);require(set(rh)==set(zh) and len(rh)==expected_tensors,'Gradient inventory differs')
 totals=[0.,0.,0.,0.];maximum=0.;count=0
 with Path(real).open('rb') as rf,Path(zero).open('rb') as zf:
  for name in sorted(rh):
   a,b=rh[name],zh[name];require(a['shape']==b['shape'],'Gradient shapes differ');rf.seek(rb+a['data_offsets'][0]);zf.seek(zb+b['data_offsets'][0]);remaining=math.prod(a['shape'])
   while remaining:
    n=min(remaining,16384);x=array.array('f');y=array.array('f');x.frombytes(rf.read(4*n));y.frombytes(zf.read(4*n))
    if sys.byteorder!='little':x.byteswap();y.byteswap()
    require(len(x)==len(y)==n and all(math.isfinite(v) for v in x) and all(math.isfinite(v) for v in y),'Nonfinite/truncated gradient')
    parts=(math.fsum(v*v for v in x),math.fsum(v*v for v in y),math.fsum(u*v for u,v in zip(x,y)),math.fsum((u-v)**2 for u,v in zip(x,y)))
    totals=[math.fsum((old,new)) for old,new in zip(totals,parts)];maximum=max(maximum,max(abs(u-v) for u,v in zip(x,y)));count+=n;remaining-=n
 rr,zz,dot,dd=totals
 return {'tensors':len(rh),'elements':count,'real_l2':math.sqrt(rr),'zero_l2':math.sqrt(zz),'difference_l2':math.sqrt(dd),'max_abs_component_difference':maximum,'cosine_similarity':dot/math.sqrt(rr*zz) if rr>0 and zz>0 else None}

def verify_adapter_file(path,fingerprints):
 with Path(path).open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];require(0<n<=4*1024**2,'Unbounded adapter header');header=json.loads(f.read(n));base=8+n
  header.pop('__metadata__',None);require(set(header)==set(fingerprints) and len(header)==504,'Actual adapter inventory differs')
  dtype={'F32':'float32','F16':'float16','BF16':'bfloat16'}
  for name,meta in header.items():
   fp=fingerprints[name];require(meta['shape']==fp['shape'] and str(fp['dtype']).endswith(dtype[meta['dtype']]),'Actual adapter shape/dtype differs')
   start,end=meta['data_offsets'];f.seek(base+start);remaining=end-start;h=hashlib.sha256()
   while remaining:
    data=f.read(min(remaining,1024**2));require(data,'Truncated adapter data');h.update(data);remaining-=len(data)
   require(h.hexdigest()==fp['sha256'],'Captured weights differ from actual checkpoint')

def conditioning(entry,scope):
 safe_path=bound(entry);safe=load(safe_path);folder=safe_path.parent;require(sha(scope['conditioning_probe']['helper_path'])==scope['conditioning_probe']['helper_sha256'],'Actual helper source changed');private_path=folder/'receipt_PRIVATE.json';require(sha(private_path)==safe['private_receipt_sha256'],'Conditioning raw capture changed');p=load(private_path)
 require(safe['status']=='MEASUREMENTS_COMPLETED_PENDING_INTERPRETATION' and safe['weight_updates']==0 and safe['verdict']=='NOT_VERIFIED','Mislabelled conditioning measurement')
 require(safe['source_binding']['native_backend']==p['contract']['native_backend_sha256']==sha(Path(scope['primary_root'])/'src/mri_rl/mlx_continuous_gspo.py'),'Native source capture differs')
 for name,value in safe['source_binding']['installed'].items():
  key='installed_mlx_vlm/'+name
  if key in scope['sources']:require(scope['sources'][key]==value,'Installed source capture differs')
 require(safe['helper_source_sha256']==scope['conditioning_probe']['helper_sha256'] and p['record_id']==scope['conditioning_probe']['record_id'],'Wrong helper/source record')
 require(p['weights_before']==p['weights_after'] and len(p['weights_before'])==504,'Adapter changed/inventory differs');require(canonical(p['weights_before'])==safe['adapter_before_sha256']==safe['adapter_after_sha256'],'Weight fingerprint summary differs')
 require(sha(scope['adapter']['path'])==scope['adapter']['sha256'],'Actual adapter file changed');verify_adapter_file(scope['adapter']['path'],p['weights_before'])
 real,zero=p['real_inputs'],p['zero_inputs'];require(set(real)==set(zero),'Input inventory differs')
 for name in real:
  if name=='pixel_values':require(real[name]['shape']==zero[name]['shape'] and real[name]['dtype']==zero[name]['dtype'] and real[name]['all_zero'] is False and zero[name]['all_zero'] is True,'Pixel zero control differs')
  else:require(real[name]==zero[name],'Nonpixel conditioning changed')
 trajectory=p['trajectory'];raw=trajectory['raw_selected_logps'];teacher=p['teacherforced_real'];require(len(raw)==len(teacher)==len(trajectory['token_ids'])>0,'Raw/TF alignment differs')
 delta=[b-a for a,b in zip(raw,teacher)];require(all(math.isfinite(v) for v in raw+teacher+p['teacherforced_zero']),'Nonfinite captured logps')
 metrics={'tokens':len(raw),'max_abs_delta':max(map(abs,delta)),'mean_abs_delta':mean(list(map(abs,delta))),'mean_signed_delta':mean(delta)}
 for k,v in metrics.items():close(v,safe['rollout_vs_teacherforced_real'][k],'raw vs TF '+k)
 per=safe['rollout_vs_teacherforced_real']['per_token'];require(len(per)==len(raw),'Missing per-token delta captures')
 for i,(a,b,d,x) in enumerate(zip(raw,teacher,delta,per)):
  require(x['index']==i,'Per-token index shifted')
  for k,v in (('stream_raw_logp',a),('teacherforced_logp',b),('teacherforced_minus_stream',d)):close(v,x[k],'per-token '+k)
 files={label:folder/('gradient_'+label+'_PRIVATE.safetensors') for label in ('real','zero')}
 for label,path in files.items():require(sha(path)==safe['gradient_files_sha256'][label]==p['gradient_files_sha256'][label],'Gradient file changed')
 grads=gradient_metrics(files['real'],files['zero'])
 for k,v in grads.items():
  if v is None:require(safe['gradient_condition_comparison'][k] is None,'Zero-norm cosine differs')
  else:close(v,safe['gradient_condition_comparison'][k],'gradient '+k,tol=1e-9)
 return {'raw_vs_TF':metrics,'gradient_metrics':grads,'adapter_unchanged':True,'input_fingerprints_unchanged_except_pixels':True,'interpretation':'NOT_VERIFIED','native_backward_independently_reexecuted':False}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--evidence-index',required=True);ap.add_argument('--expected-evidence-sha256',required=True);ap.add_argument('--output',required=True);ap.add_argument('--private-output',required=True);ap.add_argument('--reviewer-id',required=True);a=ap.parse_args();require(bool(a.reviewer_id.strip()),'Actual reviewer identity required');require(sha(a.evidence_index)==a.expected_evidence_sha256,'Root evidence index differs');index=load(a.evidence_index)
 plan=load(bound(index['plan']));lease_path=bound(index['lease']);lease=load(lease_path);contract=load(bound(index['contract']));require(lease['contract_sha256']==index['contract']['sha256'] and lease['no_model_children'] is True and lease['original_parent_exited'] is True,'Boundary lease differs')
 original_path=Path(plan['original_plan_path']);require(sha(original_path)==plan['original_plan_sha256'],'Original frozen plan differs');original=load(original_path);require(sha(original['source_manifest'])==original['source_manifest_sha256'],'Frozen pair source manifest changed')
 require(sha(contract['controller_source_path'])==contract['controller_source_sha256'] and sha(contract['handler_source_path'])==contract['handler_source_sha256'],'Actual EARLY sources changed')
 pairs=pair_index(original);numbers={};individual={};conditions={}
 for cp in CP:
  e=index['checkpoints'][cp];scope=load(bound(e['scope']));status=load(bound(e['status']));path=bound(e['output']);rows=[json.loads(x) for x in path.read_text().splitlines()]
  require(len(rows)==10 and len({r['record_id'] for r in rows})==10 and scope['checkpoint']==cp and status['checkpoint']==cp and status['completed']==10 and status['output_sha256']==e['output']['sha256'] and status['lease_sha256']==index['lease']['sha256'],'Actual completed output binding differs')
  require(status['status']=='ACTUAL_EARLY_PROBE_COMPLETED' and status['queue_resume_safe'] is True and status['native_children_released'] is True and status['critical_safety_condition'] is False,'Actual native safe completion missing')
  require(scope['boundary_contract_sha256']==index['contract']['sha256'] and scope['early_plan_sha256']==index['plan']['sha256'] and scope['weight_updates']==status['weight_updates']==0,'Wrong exact scope')
  for row,expected in zip(rows,plan['records_PRIVATE']):require(all(row.get(k)==v for k,v in expected.items()) and row['run_identity']==scope['run_identity'] and row['adapter_sha256']==scope['adapter']['sha256'] and row['checkpoint']==cp and row['weight_updates']==0,'Frozen record/adapter/order differs')
  numbers[cp],individual[cp]=checkpoint_numbers(rows,pairs);conditions[cp]=conditioning(e['conditioning'],scope)
 delta={task:{kind:{'delta_no_image_margin':numbers[CP[1]][task][kind]['no_image_margin']-numbers[CP[0]][task][kind]['no_image_margin'],'delta_unit_averaged_paired_separation':numbers[CP[1]][task][kind]['unit_averaged_paired_separation']-numbers[CP[0]][task][kind]['unit_averaged_paired_separation'],'delta_unit_averaged_real_minus_no_image_effect':numbers[CP[1]][task][kind]['unit_averaged_real_minus_no_image_effect']-numbers[CP[0]][task][kind]['unit_averaged_real_minus_no_image_effect']} for kind in KINDS} for task in LABELS}
 analyzer=load(bound(index['analyzer_result']))
 for cp in CP:
  for task in LABELS:
   for kind in KINDS:
    for key in ('no_image_margin','unit_averaged_paired_separation','unit_averaged_real_minus_no_image_effect'):close(numbers[cp][task][kind][key],analyzer['checkpoints'][cp][task][kind][key],'analyzer '+key)
    for key in delta[task][kind]:close(delta[task][kind][key],analyzer['U7_minus_B28'][task][kind][key],'analyzer '+key)
 report={'schema':'EARLY-independent-captured-numerical-verification-v1','verdict':'PASS_WITH_LIMITATIONS','reviewer_kind':'isolated_codex','reviewer_id':a.reviewer_id,'verifier_author':'/root/full_arm_execution_contract','evidence_index_sha256':a.expected_evidence_sha256,'source_sha256':sha(__file__),'checkpoints':numbers,'U7_minus_B28':delta,'conditioning':conditions,'models_launched':0,'native_backward_independently_reexecuted':False,'scientific_interpretation':'NOT_VERIFIED','limitations':['Captured-vector arithmetic verified; native generation/backpropagation not independently reexecuted.','Zero pixels are OOD; sensitivity is not MRI grounding.','Internal examined dev only; no patient-generalization or significance claim.']}
 for path,value in ((a.output,report),(a.private_output,{'individual_record_margins_PRIVATE':individual,'matched_pairs_PRIVATE':pairs})):
  dest=Path(path);require(not dest.exists(),'Preserve existing proof');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'verdict':report['verdict'],'output_sha256':sha(a.output)}))
if __name__=='__main__':main()
