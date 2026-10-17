"""Separate EARLY-scope authentication; reuse frozen pure C40 comparisons.

No fake all21 flag, native model, new metric, p value or sealed-test access.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path

WORKTREE=Path(__file__).resolve().parents[1]
ORIGINAL_PLAN_SHA='732260275ca977ccce1b59ea5856701f77a54b0150899acfc17aee2a6484b0ab'
PURE_ANALYZER_SHA='89ef601afd3286ba95aa31dd490ea5f8c7036ee01e3ae32972d2ddc9b5782b59'

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def h(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require_hash(path,expected):require(Path(path).is_file() and sha(path)==expected,'Externally supplied artifact digest differs')
def save(path,value):
 content=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();path=Path(path)
 if path.exists():require(path.read_bytes()==content,'Existing immutable analysis differs')
 else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)

def validate_early_bindings(scope,lease,status,checkpoint,plan_sha,output_sha,lease_sha):
 require(scope.get('kind')=='C40_early_natural_boundary_likelihood_v1' and scope.get('checkpoint')==checkpoint and
  scope.get('weight_updates')==0 and scope.get('original_plan_sha256')==ORIGINAL_PLAN_SHA and scope.get('early_plan_sha256')==plan_sha and
  scope.get('boundary_contract_ready') is True and 'original21training_ready' not in scope,'Exact EARLY scope required; do not relabel as original all21 probe')
 require(lease.get('schema')=='natural-arm-boundary-handoff-lease-v1' and lease.get('contract_sha256')==scope['boundary_contract_sha256'] and
  lease.get('controller_source_sha256')==scope['boundary_controller_source_sha256'] and lease.get('boundary_stage_id')==scope['boundary_stage_id'] and
  lease.get('original_parent_exited') is True and lease.get('no_model_children') is True,'Actual boundary lease differs from reviewed EARLY contract')
 require(status.get('status')=='ACTUAL_EARLY_PROBE_COMPLETED' and status.get('checkpoint')==checkpoint and status.get('completed')==10 and
  status.get('output_sha256')==output_sha and status.get('lease_sha256')==lease_sha and status.get('weight_updates')==0 and
  status.get('queue_resume_safe') is True and status.get('critical_safety_condition') is False and status.get('native_children_released') is True,
  'Actual completed released checkpoint/status/output/lease binding missing')

def original_pure_analyzer(primary):
 path=primary/'scripts/analyze_c40_visual_probe_results.py';require_hash(path,PURE_ANALYZER_SHA)
 spec=importlib.util.spec_from_file_location('original_c40_pure_analysis',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--early-plan',type=Path,required=True)
 p.add_argument('--expected-early-plan-sha256',required=True);p.add_argument('--lease',type=Path,required=True);p.add_argument('--expected-lease-sha256',required=True)
 p.add_argument('--boundary-contract',type=Path,required=True);p.add_argument('--expected-boundary-contract-sha256',required=True)
 for cp in ('b28','u7'):
  for name in ('scope','output','status'):
   p.add_argument('--'+cp+'-'+name,type=Path,required=True);p.add_argument('--'+cp+'-'+name+'-sha256',required=True)
 p.add_argument('--private-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();primary=a.primary_root.resolve()
 require_hash(a.early_plan,a.expected_early_plan_sha256);require_hash(a.lease,a.expected_lease_sha256);require_hash(a.boundary_contract,a.expected_boundary_contract_sha256)
 early=load(a.early_plan);lease=load(a.lease);contract=load(a.boundary_contract)
 require(early['schema']=='C40-early-natural-boundary-v1' and early['original_plan_sha256']==ORIGINAL_PLAN_SHA and
  lease['contract_sha256']==a.expected_boundary_contract_sha256 and contract['primary_root']==str(primary),'Early plan/lease/primary contract differs')
 require_hash(contract['controller_source_path'],contract['controller_source_sha256']);require_hash(contract['handler_source_path'],early['early_handler_sha256'])
 require(contract['handler_source_sha256']==early['early_handler_sha256'] and lease['controller_source_sha256']==contract['controller_source_sha256'],'Historical EARLY controller/handler source differs')
 pure=original_pure_analyzer(primary);api=pure.probe;original=api.validate_plan(Path(early['original_plan_path']))
 for field in ('records_PRIVATE','checkpoints','derived_cache','likelihood','greedy'):require(early[field]==original[field],'EARLY changed original numerical/data contract')
 require(early['primary_sources']==api.source_hashes(),'Frozen original sources changed')
 from tokenizers import Tokenizer
 tokenizer=Tokenizer.from_file(str(Path(original['model_path'])/'tokenizer.json'))
 class RawTokenizer:
  def encode(self,text,add_special_tokens=False):return tokenizer.encode(text,add_special_tokens=add_special_tokens).ids
 outputs={};evidence={}
 for cp,short in (('B28','b28'),('U_raw_seed7','u7')):
  path=getattr(a,short+'_output');scopepath=getattr(a,short+'_scope');statuspath=getattr(a,short+'_status')
  oh=getattr(a,short+'_output_sha256');sh=getattr(a,short+'_scope_sha256');sth=getattr(a,short+'_status_sha256')
  for fn,digest in ((path,oh),(scopepath,sh),(statuspath,sth)):require_hash(fn,digest)
  scope=load(scopepath);status=load(statuspath);validate_early_bindings(scope,lease,status,cp,a.expected_early_plan_sha256,oh,a.expected_lease_sha256)
  require(scope['adapter']==original['checkpoints'][cp] and scope['derived_cache']==original['derived_cache'] and scope['sources']==early['primary_sources'] and
   scope['early_handler_sha256']==early['early_handler_sha256'] and scope['boundary_contract_sha256']==a.expected_boundary_contract_sha256 and
   scope['output']==str(path.resolve()),'Scope weights/cache/source/output differs')
  identity=h({'early_plan':a.expected_early_plan_sha256,'checkpoint':cp,'adapter':scope['adapter'],'sources':scope['sources'],'boundary_contract':a.expected_boundary_contract_sha256})
  require(scope['run_identity']==identity and scope['fixed_prefix']==original['likelihood']['prefix'] and scope['fixed_suffix']==original['likelihood']['suffix'] and scope['prefix_tolerance']==.002,'EARLY identity or candidate prefix differs')
  data=path.read_bytes();require(data.endswith(b'\n') and len(data.splitlines())==10,'Complete untorn10-record output required')
  rows=[json.loads(x) for x in data.splitlines()];require(all(r.get('early_release') is True for r in rows),'Missing actual EARLY execution annotation')
  count=scope['completed_records'];require(type(count) is int and 0<=count<=10 and scope['new_record_ids']==[r['record_id'] for r in original['records_PRIVATE'][count:]],'Frozen resumed-prefix cursor differs')
  if count:require(hashlib.sha256(b''.join(data.splitlines(keepends=True)[:count])).hexdigest()==scope['existing_output_sha256'],'Durable EARLY prefix changed')
  else:require(scope['existing_output_sha256'] is None,'Unexpected prior output')
  outputs[cp]=pure.validate_records(rows,original['records_PRIVATE'],scope,cp,RawTokenizer())
  require_hash(path,oh);evidence[cp]={'output_sha256':oh,'scope_file_sha256':sh,'status_file_sha256':sth}
 result,private=pure.compare_checkpoints(outputs,pure.private_pairs(original))
 require(a.private_output.resolve().is_relative_to(WORKTREE/'local_data') or a.private_output.resolve().is_relative_to(primary/'local_data'),'Private detail must remain local')
 require(a.output.resolve().is_relative_to(WORKTREE/'results') or a.output.resolve().is_relative_to(primary/'results'),'Aggregate output must remain in results')
 private.update({'early_input_hashes':evidence,'actual_lease_sha256':a.expected_lease_sha256});save(a.private_output,private)
 result.update({'schema':'C40-EARLY-actual-visual-probe-analysis-v1','status':'ACTUAL_EARLY20_RECORDS_AUTHENTICATED_AND_ANALYZED','author':'Codex secondary CPU analyzer; actual execution provenance retained in scope/status/lease',
  'eligibility':'Independently reviewed actual natural-arm boundary; this is explicitly not an all21-complete launch.',
  'original_frozen_plan_sha256':ORIGINAL_PLAN_SHA,'early_plan_sha256':a.expected_early_plan_sha256,'actual_boundary_lease_sha256':a.expected_lease_sha256,
  'boundary_contract_sha256':a.expected_boundary_contract_sha256,'input_hashes':evidence,'private_detail_sha256':sha(a.private_output),'analyzer_sha256':sha(__file__),
  'pure_original_analyzer_sha256':PURE_ANALYZER_SHA,'primary_score':'Mean label-token log-probability margin; raw sums and first-label margins retained without cherry-picking.',
  'margin_direction':{'mri_sequence':'T1 minus FLAIR','mri_visit_correspondence':'same minus different'},
  'conditioning_auxiliary':'Separate finite engineering measurements, not analyzed/interpreted by this visual-margin analyzer.',
  'native_models_launched_by_analysis':0,'weight_updates':0,'sealed_test_used':False,'p_values_computed':False,
  'limitations':['Previously examined internal-dev only: two sequence people at a fixed level and one Dallas matched cycle; descriptive means, not independent-patient generalization.',
   'No-image candidate margins have no image truth/accuracy and are not calibrated class probabilities.',
   'A common answer-prior shift cancels from positive-minus-negative image separation; changed separation still does not establish brain-specific grounding or causal training mechanism.',
   'Whole-head/intensity/acquisition shortcuts and source/task confounding remain; no significance or clinical claim.']})
 require(not any(k.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'} for k in sys.modules),'Unexpected native model module import')
 save(a.output,result);print(json.dumps({'status':result['status'],'records':20,'output_sha256':sha(a.output),'weight_updates':0}))

if __name__=='__main__':main()
