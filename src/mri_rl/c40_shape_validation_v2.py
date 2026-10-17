"""Experimental validation helper for the single evidenced C40 null-image case.

No change to the original scorer, candidate likelihoods, .002 threshold, training
or original result files. This is a tagged auxiliary result, never a legacy PASS.
Actual native support currently covers B28/index9 P115/prefix11/equal129 only;
the pinned R2 control rejects any other shape. Future uses need separate review.
"""
from __future__ import annotations
import hashlib,importlib.util,json,math,traceback
from pathlib import Path

VERSION='C40_no_image_equal_shape_oracle_v2'
THRESHOLD=.002
R2_SHA='2d5687e2964ab9dac7a2feb6e87a45ba143e8b20fe15bf7fca360f6ef68df0ed'
EVIDENCE_KEYS=('first','oracle','errors','prefix_scores','candidates','parts','P','visual','grid','label_first','prompt_list')

def require(ok,message):
 if not ok:raise ValueError(message)

def _load_shape_control():
 path=Path(__file__).resolve().parents[2]/'scripts/run_c40_prefix_oracle_shape_v2.py'
 require(hashlib.sha256(path.read_bytes()).hexdigest()==R2_SHA,'Independently reviewed R2 source changed')
 spec=importlib.util.spec_from_file_location('bound_c40_r2_shape',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.padded_control

def score_record_v2(api,backend,record,index):
 """Return separately tagged evidence; caller owns output persistence/resources.

Exactly original score_record calls first. Only its specific oracle ValueError
permits ONE R2 control forward. Other errors propagate. This function neither
appends to frozen results nor changes any metric, candidate value or model weight.
 """
 require(record.get('condition')=='no_image' and record.get('category')=='mri_visit_correspondence' and record.get('image') is None,'NOT_VERIFIED: only null-image correspondence is eligible')
 try:
  result=api.score_record(backend,record,index)
 except ValueError as error:
  if str(error)!='Independent prefix oracle disagrees':raise
  frame=None;tb=error.__traceback__
  try:
   while tb is not None:
    if tb.tb_frame.f_code is api.score_record.__code__:frame=tb.tb_frame
    tb=tb.tb_next
   require(frame is not None,'NOT_VERIFIED: exact legacy scorer failure frame missing')
   values=frame.f_locals
   parts=values.get('parts',{});lengths=[len(x['target_ids']) for x in parts.values()]
   require(len(lengths)==2 and len(set(lengths))==1 and values.get('pixels') is None,'NOT_VERIFIED: equal candidate shapes and no pixels required')
   evidence=json.loads(json.dumps({k:values[k] for k in EVIDENCE_KEYS if k in values},allow_nan=False))
   require(all(k in evidence for k in ('candidates','first','oracle','errors','prefix_scores')),'NOT_VERIFIED: original numerical evidence missing')
   prefix=evidence['prefix_scores']
   require(len(prefix)==2 and len(prefix[0])==len(prefix[1]) and len(prefix[0])>0 and all(math.isfinite(v) for row in prefix for v in row) and max(abs(a-b) for a,b in zip(*prefix))<=THRESHOLD,'NOT_VERIFIED: original common prefix disagreement cannot be corrected')
   control=_load_shape_control()(backend,values)
   maximum=control['maximum_candidate_first_label_error']
   require(math.isfinite(maximum) and control['unchanged_tolerance']==THRESHOLD and control['control_within_original_tolerance']==(maximum<=THRESHOLD),'NOT_VERIFIED: invalid shape control evidence')
   return {'validation_version':VERSION,'scope':'Experimental validation correction only; no MRI algorithm or training change',
    'verdict':'PASS_WITH_LIMITATIONS' if maximum<=THRESHOLD else 'NOT_VERIFIED',
    'legacy_oracle_gate':'FAIL','legacy_evidence_PRIVATE':evidence,'candidate_scores':evidence['candidates'],
    'shape_matched_oracle_gate':'PASS' if maximum<=THRESHOLD else 'FAIL','shape_matched_control':control,'threshold':THRESHOLD,
    'frozen_experiment_completed':False,'weight_updates':0,
    'limitations':['Native support is one B28 null-image correspondence case; no cross-checkpoint or real-image validation claimed.',
     'Candidate scores are preserved. Legacy failure is not converted into legacy PASS.',
     'Shape dependence does not identify a kernel cause or establish MRI understanding.']}
  finally:
   traceback.clear_frames(error.__traceback__)
   # Do not retain the failed score frame or its Metal arrays in returned evidence.
   frame=None;tb=None
   if 'values' in locals():values=None
 else:
  return {'validation_version':VERSION,'legacy_oracle_gate':'PASS','correction_applied':False,'legacy_result':result,
   'frozen_experiment_completed':False,'weight_updates':0}
