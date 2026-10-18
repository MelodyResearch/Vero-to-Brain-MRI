"""Bounded CPU verification of saved native paired LABEL-CE gradient evidence.
No native imports/forward/backward/refit. Public proof contains pair scalars only.
"""
import argparse,hashlib,json,math,struct,sys
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1]
NAMES=('REAL_PAIR','SWAPPED_PIXEL_TENSORS_FIXED_TRUTH','ZERO_NORMALIZED_PIXEL_TENSOR')
PINS={'scope_READY_REQUIRES_REVIEW_PRIVATE.json':'efe80a6297b5684276434c51597b8b6560b45999d585ff3c6b2b8da71eb1df4b','INDEPENDENT_REVIEW_PRIVATE.json':'b0d7802771c2044bbd56a312a0ac702e63c56ea163c07730d03330595e83e035','actual_exit_PRIVATE.json':'7881afdb124a34e27d791da3a81b186f2a93ff45e99d02ef2cfc68cd9d06d87e','native_stdout_PRIVATE.log':'77820df288b3abe0cc5a3a580a4f5f879505561b9edc306e958adfe88fc95b0c','actual/complete_SAFE.json':'2b2436178a6881d8707da013f9256af873f73a6d1cb38f1c06cb667c11e05cee'}
def req(value,message):
 if not value:raise ValueError(message)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):return json.loads(Path(path).read_text())
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def near(a,b,atol=1e-10):req(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=atol),'Independent scalar mismatch')
def near_ce(a,b):
 # Candidate means and native logaddexp are F32; compare independent F64 CE
 # within two ULPs of the stored F32 loss, without changing any model gate.
 near(a,b,atol=2*abs(float(np.spacing(np.float32(b))))+1e-12)
def header(path):
 with Path(path).open('rb') as f:
  prefix=f.read(8);req(len(prefix)==8,'Missing tensor header');n=struct.unpack('<Q',prefix)[0];req(0<n<4_000_000,'Unbounded tensor header');h=json.loads(f.read(n))
 h.pop('__metadata__',None);size=Path(path).stat().st_size-8-n;intervals=[]
 for name,v in h.items():
  lo,hi=v['data_offsets'];req(type(lo) is int and type(hi) is int and 0<=lo<=hi<=size and all(type(x) is int and x>=0 for x in v['shape']),'Malformed tensor offsets/shape');intervals.append((lo,hi))
 req(sorted(intervals)[0][0]==0 and sorted(intervals)[-1][1]==size and all(a[1]==b[0] for a,b in zip(sorted(intervals),sorted(intervals)[1:])),'Tensor bytes missing/overlap')
 return h,8+n
def tensor_bytes(f,base,v):
 lo,hi=v['data_offsets'];f.seek(base+lo);x=f.read(hi-lo);req(len(x)==hi-lo,'Truncated tensor');return x
def verify_fingerprints(path,fp):
 h,base=header(path);req(set(h)==set(fp),'Fingerprint tensor inventory differs')
 with Path(path).open('rb') as f:
  for name,v in h.items():
   p=fp[name];req(v['shape']==p['shape'] and {'F32':'mlx.core.float32','U32':'mlx.core.uint32','U64':'mlx.core.uint64','BF16':'mlx.core.bfloat16'}[v['dtype']]==p['dtype'],'Fingerprint shape/dtype differs');req(hashlib.sha256(tensor_bytes(f,base,v)).hexdigest()==p['sha256'],'Native fingerprint differs from saved checkpoint bytes')
 return len(h)
def vector_comparison(left,right,expected):
 lh,lb=header(left);rh,rb=header(right);req(set(lh)==set(rh)==set(expected) and len(lh)==504,'504 gradient tensors required');aa=[];bb=[];ab=[];dd=[];maximum=0.;elements=0
 with Path(left).open('rb') as lf,Path(right).open('rb') as rf:
  for name in sorted(lh):
   l,r=lh[name],rh[name];req(l['dtype']==r['dtype']=='F32' and l['shape']==r['shape']==expected[name]['shape'],'Gradient dtype/shape changed');a=np.frombuffer(tensor_bytes(lf,lb,l),dtype='<f4').astype(np.float64);b=np.frombuffer(tensor_bytes(rf,rb,r),dtype='<f4').astype(np.float64);req(len(a)==math.prod(l['shape']) and np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite or malformed gradient');d=a-b;aa.append(float(np.dot(a,a)));bb.append(float(np.dot(b,b)));ab.append(float(np.dot(a,b)));dd.append(float(np.dot(d,d)));maximum=max(maximum,float(np.max(np.abs(d))));elements+=len(a)
 a2,b2,dot,d2=map(math.fsum,(aa,bb,ab,dd));return {'tensors':504,'elements':elements,'left_l2':math.sqrt(a2),'right_l2':math.sqrt(b2),'difference_l2':math.sqrt(d2),'cosine':dot/math.sqrt(a2*b2) if a2>0 and b2>0 else None,'max_abs_difference':maximum}
def zeros_sha(shape,dtype):
 sizes={'mlx.core.float32':4,'mlx.core.bfloat16':2};req(dtype in sizes,'Unknown pixel dtype');n=math.prod(shape)*sizes[dtype];d=hashlib.sha256();chunk=bytes(65536)
 while n:d.update(chunk[:min(n,len(chunk))]);n-=min(n,len(chunk))
 return d.hexdigest()
def verify(a):
 req(a.reviewer_id.strip(),'Actual reviewer required');d=W/'local_data/paired_native_gradient_probe_v1';out=d/'actual'
 for name,pin in PINS.items():req(sha(d/name)==pin,'Externally pinned actual evidence differs')
 scope=load(d/'scope_READY_REQUIRES_REVIEW_PRIVATE.json');review=load(d/'INDEPENDENT_REVIEW_PRIVATE.json');ex=load(d/'actual_exit_PRIVATE.json');complete=load(out/'complete_SAFE.json')
 req(ex['exit_code']==0 and ex['scope_file_sha256']==PINS['scope_READY_REQUIRES_REVIEW_PRIVATE.json'] and ex['review_sha256']==PINS['INDEPENDENT_REVIEW_PRIVATE.json'] and ex['raw_log_sha256']==PINS['native_stdout_PRIVATE.log'],'Actual native exit binding differs');req(review['verdict']=='PASS_WITH_LIMITATIONS' and all(review.get(k)==v for k,v in scope.items()),'Exact independent scope approval missing')
 runner=W/'scripts/run_paired_native_gradient_probe_v1.py';helper=W/'src/mri_rl/paired_native_gradient_probe_v1.py';req(sha(runner)==ex['source_sha256']==scope['entry_sha256']==complete['scope_source_sha256'] and sha(helper)==scope['helper_sha256'],'Actual diagnostic source changed');req(complete['scope_sha256']==canonical(scope),'Actual complete scope differs')
 req(complete['status']=='ACTUAL_THREE_PAIR_GRADIENT_CONDITIONS_COMPLETE_AND_RELEASED' and complete['native_release_verified'] is True and complete['active_MLX_bytes_after_release']==1032<128*1024**2,'Actual release missing');req(complete['optimizer_step_before']==complete['optimizer_step_after']==56 and complete['optimizer_RNG_preserved'] is True and complete['adapter_tensors_checked']==504,'State count not exact');req(complete['budget']==scope['budget']=={'candidate_forwards':12,'generation_tokens':0,'images':2,'logical_pair_conditions':3,'optimizer_updates':0,'primitive_one_image_CE_gradients':6,'simultaneous_image_graphs':1},'Actual finite call budget differs')
 req(complete['adapter_sha256']==scope['adapter']['sha256']==sha(scope['adapter']['path'])=='23b6ce522f770bf39acbadd9b66ccf00529de7822a5bb4173e7c4b96cd8618a7','Selected child weights differ');req(sha(out/'integrity_PRIVATE.json')==complete['integrity_sha256'],'Integrity evidence differs');integrity=load(out/'integrity_PRIVATE.json');wb=integrity['weights_before'];req(wb==integrity['weights_after'] and integrity['state_before']==integrity['state_after'],'Native state changed');req(canonical(wb)==complete['adapter_before_digest']==complete['adapter_after_digest'],'Adapter digest differs');verify_fingerprints(scope['adapter']['path'],wb)
 child=scope['child_checkpoint'];req(sha(child['proof_path'])==child['proof_sha256']=='39d62aa3d9949b3ec39a52fb9b64fe82eb1233fdafdd63158fb7e0d558a87967','Child inherited-state proof changed');boundary=Path(child['receipt_path']).parent;req(sha(child['receipt_path'])==child['receipt_sha256'],'Actual child receipt differs');om=load(boundary/'optimizer/manifest.json');req(sha(boundary/'optimizer/manifest.json')==scope['optimizer_manifest_sha256'] and sha(boundary/'optimizer/state.safetensors')==scope['optimizer_state_sha256']==om['tensors_sha256'],'Selected optimizer bytes differ');req(om['optimizer_updates']==56 and om['completed_groups']==121 and om['layout']==integrity['state_before']['layout'],'Saved/native state layout differs');state_count=verify_fingerprints(boundary/'optimizer/state.safetensors',integrity['state_before']['arrays'])
 h,base=header(boundary/'optimizer/state.safetensors');stepnode=dict(dict(om['layout']['items'])['optimizer']['items'])['step'];stepkey=stepnode['key']
 with (boundary/'optimizer/state.safetensors').open('rb') as f:req(int.from_bytes(tensor_bytes(f,base,h[stepkey]),'little')==56,'Actual byte-level Adam counter differs')
 conditions={};private={};inputfp=integrity['input_fingerprints'];req(len(inputfp)==2 and all(set(x)=={'input_ids','attention_mask','image_grid_thw','pixel_values'} for x in inputfp),'Actual native input inventory differs');req(all(inputfp[0][k]==inputfp[1][k] for k in ('input_ids','attention_mask','image_grid_thw')),'Paired prompt/mask/grid differs')
 for name in NAMES:
  p=out/(name+'_PRIVATE.json');x=load(p);private[name]=x;safe={k:v for k,v in x.items() if not k.endswith('_PRIVATE')};req(safe==complete['conditions'][name],'Per-condition/full receipt differs');req(x['primitive_gradients']==2 and x['candidate_forwards']==4 and x['optimizer_RNG_unchanged'] is True and x['adapter_digest']==canonical(wb),'Condition counts/state differ');gp=out/(name+'_gradient_PRIVATE.safetensors');req(sha(gp)==x['gradient_sha256'],'Actual gradient file differs')
  scores=x['candidate_mean_label_logps'];req(len(scores)==2 and all(len(v)==2 and all(math.isfinite(z) and z<=1e-6 for z in v) for v in scores),'Invalid saved scores');losses=[];margins=[]
  for truth,v in enumerate(scores):losses.append(float(np.logaddexp(float(v[1-truth])-float(v[truth]),0.)));margins.append(v[1]-v[0]);near_ce(losses[-1],x['per_image_CE_losses'][truth]);near(margins[-1],x['t1_minus_flair_margins'][truth])
  ce=math.fsum(losses)/2;sep=margins[1]-margins[0];near_ce(ce,x['mean_pair_CE_loss']);near(sep,x['paired_separation']);pixel=x['pixel_tensor_fingerprints_PRIVATE'];req(set(pixel)=={'0','1'},'Wrong two pixel tensors')
  for i in range(2):
   fp=pixel[str(i)];ref=inputfp[i if name!=NAMES[1] else 1-i]['pixel_values'];req(fp['shape']==ref['shape'] and fp['dtype']==ref['dtype'],'Routed pixels shape/dtype differs');req(fp['sha256']==(zeros_sha(ref['shape'],ref['dtype']) if name==NAMES[2] else ref['sha256']),'Actual real/swap/zero pixel bytes differ')
  conditions[name]={'mean_pair_CE_loss_recomputed_F64':ce,'mean_pair_CE_loss_native_F32':x['mean_pair_CE_loss'],'paired_separation':sep,'gradient_sha256':sha(gp),'condition_receipt_sha256':sha(p)}
 req(private[NAMES[1]]['candidate_mean_label_logps']==private[NAMES[0]]['candidate_mean_label_logps'][::-1],'Swapped score matrix is not exact row reversal');req(private[NAMES[2]]['candidate_mean_label_logps'][0]==private[NAMES[2]]['candidate_mean_label_logps'][1],'Zero condition differs across identical prompts');req(conditions[NAMES[1]]['paired_separation']==-conditions[NAMES[0]]['paired_separation'] and conditions[NAMES[2]]['paired_separation']==0,'Separation route identity differs')
 comparisons={}
 for name in NAMES[1:]:
  key=name+'_vs_REAL_PAIR';x=vector_comparison(out/(NAMES[0]+'_gradient_PRIVATE.safetensors'),out/(name+'_gradient_PRIVATE.safetensors'),wb);saved=complete['comparisons'][key]
  for field,val in saved.items():
   if type(val) is int:req(val==x[field],'Gradient count differs')
   elif val is None:req(x[field] is None,'Zero norm cosine differs')
   else:near(x[field],val)
  comparisons[key]=x
 req(not any(k.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'} for k in sys.modules),'Unexpected native import')
 proof={'schema':'PAIRED_NATIVE_LABEL_CE_GRADIENT_SAVED_NUMERICAL_INDEPENDENT_V1','verdict':'PASS_WITH_LIMITATIONS','reviewer_kind':'isolated_codex','reviewer_id':a.reviewer_id,'verifier_author':'/root/full_arm_execution_contract','verifier_sha256':sha(__file__),'evidence_sha256':PINS,'native_runner_sha256':sha(runner),'native_helper_sha256':sha(helper),'native_integrity_sha256':sha(out/'integrity_PRIVATE.json'),'selected_child_adapter_sha256':scope['adapter']['sha256'],'selected_child_optimizer_state_sha256':scope['optimizer_state_sha256'],'selected_child_proof_sha256':child['proof_sha256'],'pair_aggregate_conditions':conditions,'gradient_scalar_comparisons':comparisons,'swapped_score_matrix_exact_reversal':True,'zero_pixel_pair_score_rows_exactly_equal':True,'native_input_prompt_mask_grid_same':True,'actual_pixel_routes_authenticated':True,'adapter_tensors_independently_byte_matched':504,'optimizer_rng_tensors_independently_byte_matched':state_count,'optimizer_step_before_and_after':56,'optimizer_rng_layout_and_fingerprints_unchanged':True,'native_release_verified':True,'active_MLX_bytes_after_release':1032,'call_budget':scope['budget'],'native_models_forwards_backwards_by_reviewer':0,'per_image_arrays_publicly_exported':False,'CE_F32_to_F64_tolerance':'two ULPs of stored F32 scalar plus1e-12; only CPU verification roundoff, no model threshold change','limitations':['Saved tensor arithmetic, state bytes and source/actual receipts verified; native backward was not independently re-executed or finite-differenced.','Image-conditioned LABEL-CE gradient sensitivity on exactly one prespecified training pair is not MRI accuracy/generalization or anatomical grounding.','This mean-label CE diagnostic differs from GSPO loss; it does not establish GSPO/native-loss parity or identify the cause of failed accuracy improvement.','Zero normalized pixels are OOD with unchanged prompt/grid; swapped pixels retain intentionally contradictory fixed truth.','Pair-level gradient aggregate only: per-image score/loss arrays and patient-linked metadata remain private.','No new training, algorithm change or automated expansion is approved by this numerical proof.']}
 req(a.output.is_absolute() and a.output.parent==W/'results' and not a.output.exists(),'Fresh sanitized destination required');a.output.write_text(json.dumps(proof,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps({'verdict':proof['verdict'],'proof_sha256':sha(a.output),'conditions':conditions,'gradient_comparisons':comparisons,'state_tensors':state_count},sort_keys=True))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reviewer-id',required=True);p.add_argument('--output',type=Path,required=True);verify(p.parse_args())
if __name__=='__main__':main()
