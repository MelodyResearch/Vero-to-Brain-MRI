"""Thin original22 final100 release/scope/serial-CLI bridge. Never self-approves.

All three commands require an externally supplied bundle digest. `release` needs
an independent final_test_release approval; `scopes` runs only after that release;
`run` needs22 independently authored native CLI reviews. Import does no data I/O.
No alternate prompt/checkpoint, adaptive arm, subset, training, or retry is added.
"""
from __future__ import annotations
import argparse,json,os,subprocess
from pathlib import Path
from types import SimpleNamespace
from mri_rl.two_task_pool import canonical_sha,require,sha,write_exact

ROOT=Path(__file__).resolve().parents[2]
METHOD_FREEZE_SHA='35e69aed73cfd120d659cb28fbd63ab49f959a61b0e7dcafe228d7c5af3a4658'
load=lambda p:json.loads(Path(p).read_text())


def saved(path,value):
 write_exact(Path(path),(json.dumps(value,sort_keys=True,indent=2)+'\n').encode())


def bundle_at(path,expected):
 require(sha(path)==expected,'External bundle digest differs');b=load(path)
 require(b['status']=='PREPARED_NOT_RELEASED_NOT_EXECUTED','Not the prepared original22 bundle')
 require(b['method_freeze']['sha256']==METHOD_FREEZE_SHA and sha(b['method_freeze']['path'])==METHOD_FREEZE_SHA,'Exact current finite method freeze required')
 for d in b['metadata_and_source_descriptors']:
  require(sha(d['path'])==d['sha256'],'Reviewed metadata/source changed')
 from mri_rl.final_test_release import required_checkpoints
 needed=required_checkpoints(b['plan']);ids=[i['checkpoint_id'] for i in b['inventory']]
 require(len(ids)==len(set(ids))==22 and set(ids)==set(needed),'Exact original22 checkpoint inventory required')
 require(b['generation']=={'max_tokens':256,'temperature':0} and b['weight_updates']==0,'Frozen greedy256/temp0/zero-update scope required')
 require(b['executor_source_sha256']==sha(__file__),'Executor source changed')
 return b


def release_bundle(b,approval_path,approval_digest):
 return {'plan':b['plan'],'inventory':b['inventory'],'bindings':b['bindings'],
         'approval':load(approval_path),'expected_approval_sha256':approval_digest}


def check_released(b,release):
 from mri_rl.final_test_release import validate_release
 validate_release(release['plan'],release['inventory'],release['bindings'],release['approval'],expected_approval_sha256=release['expected_approval_sha256'])
 d=Path(b['projection_directory']);p=load(d/'projection_receipt_SAFE.json')
 require(p['status']=='RELEASED_FINAL_TEST_PROJECTION_FROZEN' and p['cases']==100 and p['approval_sha256']==release['expected_approval_sha256'] and p['bindings']==b['bindings'],'Exact independent final100 release required')
 require(sha(d/'final_test_native_system_matched.jsonl')==p['eval_manifest_sha256'] and sha(d/'final_test.verl.jsonl')==p['source_verl_manifest_sha256'],'Released manifests changed')
 return p


def cli_argv(b,run,review,review_sha):
 return [b['python'],'-u','-m','mri_rl.continuous_rl_dev_eval','--model',b['model'],'--contract',b['contract_path'],
  '--adapter',run['adapter'],'--manifest',run['manifest'],'--output',run['output'],'--label',run['label'],'--seed',str(run['seed']),
  '--cache-dir',b['cache_dir'],'--cache-certificate',b['cache_certificate'],'--cache-certificate-sha256',b['cache_certificate_sha256'],
  '--review-verdict',str(review),'--expected-review-sha256',review_sha]


def prepare_scopes(b,release):
 """AFTER released rendering, calculate real manifest/image/model inventories. No model import."""
 from mri_rl import continuous_rl_dev_eval as api
 from mri_rl.continuous_rl_pilot import cache_binding_for
 from mri_rl.experiment_contract import load_contract
 from mri_rl.final_test_release import required_checkpoints
 p=check_released(b,release);manifest=Path(b['projection_directory'])/'final_test_native_system_matched.jsonl'
 records=api.select_records(manifest,None);require(len(records)==100,'Full100 case manifest required')
 contract=load_contract(Path(b['contract_path']));require(contract.generation['max_tokens']==256,'Frozen native cap differs')
 a=SimpleNamespace(model=Path(b['model']),cache_dir=Path(b['cache_dir']),cache_certificate=Path(b['cache_certificate']),cache_certificate_sha256=b['cache_certificate_sha256'])
 cache=cache_binding_for(a);inventory=api.model_inventory(a.model,contract,Path(b['contract_path']));images={r['record_id']:sha(r['image']) for r in records}
 items={i['checkpoint_id']:i for i in b['inventory']};runs=[]
 for key in required_checkpoints(b['plan']):
  item=items[key];d=Path(b['execution_directory'])/key.replace(':','_');output=d/'outputs_PRIVATE.jsonl'
  require(not output.exists(),'Initial scope preparation cannot relabel existing outputs')
  expected=api.expected_config(a.model,manifest,Path(item['adapter']['path']),256)
  require(expected['adapter_checkpoint_sha256']==item['adapter']['sha256'] and expected['adapter_config_sha256']==item['adapter_config']['sha256'],'Final adapter differs')
  seed=7 if key=='B28' else int(key.rsplit(':',1)[1]);label='ORIGINAL22_FINAL100_'+key.replace(':','_')
  scope={'execution_tier':'B','kind':'continuous_rl_dev_eval_v1','label':label,'contract_id':contract.identity(),
   'manifest_sha256':expected['manifest_sha256'],'subset_record_ids':[r['record_id'] for r in records],
   'max_tokens':256,'temperature':0,'adapter_checkpoint_sha256':expected['adapter_checkpoint_sha256'],
   'adapter_config_sha256':expected['adapter_config_sha256'],'output':str(output),'weight_updates':0,'seed':seed,'resume':False,
   'derived_cache':cache,'starting_output_sha256':None,'image_sha256':images,'model_inventory':inventory,'source_sha256':api.source_hashes()}
  sp=d/'scope_REQUIRES_INDEPENDENT_REVIEW_PRIVATE.json';saved(sp,scope)
  runs.append({'checkpoint_id':key,'adapter':item['adapter']['path'],'manifest':str(manifest),'output':str(output),
   'label':label,'seed':seed,'scope':{'path':str(sp),'sha256':sha(sp)},'expected_config':expected,
   'log':str(d/'stdout_PRIVATE.log'),'exit':str(d/'actual_exit_SAFE.json')})
 result={'status':'EXACT22_NATIVE_SCOPES_NOT_APPROVED_NOT_RUN','executor_source_sha256':sha(__file__),
  'projection_receipt_sha256':sha(Path(b['projection_directory'])/'projection_receipt_SAFE.json'),
  'release_approval_sha256':release['expected_approval_sha256'],'inventory_sha256':canonical_sha(b['inventory']),'runs':runs}
 saved(Path(b['execution_directory'])/'scope_set_REQUIRES_INDEPENDENT_REVIEW_PRIVATE.json',result);return result


def validate_review_set(path,expected,scope_set,scope_sha):
 require(sha(path)==expected,'External native-review-set digest differs');reviews=load(path)
 require(reviews.get('scope_set_sha256')==scope_sha,'Exact reviewed scope set differs')
 rows=reviews.get('reviews',[]);required=[r['checkpoint_id'] for r in scope_set['runs']]
 require(len(required)==len(set(required))==22 and len(rows)==22 and len({r['checkpoint_id'] for r in rows})==22 and {r['checkpoint_id'] for r in rows}==set(required),'Exact22 external native reviews required')
 from mri_rl.experiment_contract import _check_review
 mapping={r['checkpoint_id']:r['review'] for r in rows}
 for run in scope_set['runs']:
  s=run['scope'];require(sha(s['path'])==s['sha256'],'Prepared native scope changed');d=mapping[run['checkpoint_id']]
  _check_review(review_verdict_path=Path(d['path']),expected_review_sha256=d['sha256'],scope=load(s['path']))
 return mapping


def completed_stage(run,review_sha):
 output=Path(run['output']);log=Path(run['log']);w=Path(run['exit'])
 if not w.exists():
  require(not output.exists() and not log.exists(),'Partial stage requires separately reviewed recovery; no automatic retry');return False
 x=load(w);require(x.get('exit_code')==0,'Previously failed stage must not retry automatically')
 require(x.get('checkpoint_id')==run['checkpoint_id'] and x.get('review_sha256')==review_sha,'Completed checkpoint/review differs')
 require(log.is_file() and sha(log)==x['raw_log_sha256'],'Completed closed log changed')
 require(output.is_file() and sha(output)==x['output_sha256'],'Completed output changed')
 rows=[json.loads(l) for l in output.read_text().splitlines() if l.strip()]
 require(len(rows)==len({r['record_id'] for r in rows})==x['records']==100,'Completed full100 coverage required');return True


def run_serial(b,release,scope_set_path,reviews_path,review_sha):
 """Existing native CLI only, with one global flock across all22 serial children."""
 import fcntl
 check_released(b,release);ss=load(scope_set_path)
 require(ss['executor_source_sha256']==sha(__file__) and ss['release_approval_sha256']==release['expected_approval_sha256'] and ss['inventory_sha256']==canonical_sha(b['inventory']) and ss['projection_receipt_sha256']==sha(Path(b['projection_directory'])/'projection_receipt_SAFE.json'),'Scope set/release/checkpoints differ')
 reviews=validate_review_set(reviews_path,review_sha,ss,sha(scope_set_path))
 from mri_rl.final_test_release import required_checkpoints
 require([r['checkpoint_id'] for r in ss['runs']]==list(required_checkpoints(b['plan'])),'Fixed original checkpoint order differs')
 lock=Path(b['global_lock']);require(lock==ROOT/'local_data/c_two_task_pool_v1/tier_c_v1/remaining20_bundle_v1/remaining20.lock','Existing global native lock required')
 env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
 with lock.open('a') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for index,run in enumerate(ss['runs']):
   review=reviews[run['checkpoint_id']]
   if not completed_stage(run,review['sha256']):
    with Path(run['log']).open('xb') as log:
     result=subprocess.run(cli_argv(b,run,review['path'],review['sha256']),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);log.flush();os.fsync(log.fileno())
    output=Path(run['output']);rows=[json.loads(l) for l in output.read_text().splitlines() if l.strip()] if output.exists() else []
    witness={'checkpoint_id':run['checkpoint_id'],'exit_code':result.returncode,'review_sha256':review['sha256'],
     'raw_log_sha256':sha(run['log']),'output_sha256':sha(output) if output.exists() else None,'records':len(rows),'executor':'Local subprocess; caller identity retained by coordinator','scope_sha256':run['scope']['sha256']}
    saved(run['exit'],witness);require(result.returncode==0,'Native stage failed; outputs preserved and queue halted without retry');completed_stage(run,review['sha256'])
   saved(Path(b['execution_directory'])/f'cursor_{index+1:02d}_SAFE.json',{'completed':index+1,'total':22,'last_checkpoint_id':run['checkpoint_id'],'actual_exit_sha256':sha(run['exit']),'review_set_sha256':review_sha,'test_accuracy_claim':False})
 return {'status':'ALL22_NATIVE_FINAL100_OUTPUTS_DURABLE_NOT_SCORED','checkpoints':22,'cases_per_checkpoint':100,'weight_updates':0}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('release','scopes','run'));p.add_argument('--bundle',type=Path,required=True);p.add_argument('--expected-bundle-sha256',required=True);p.add_argument('--release-review',type=Path,required=True);p.add_argument('--expected-release-review-sha256',required=True,help='Independent canonical approval digest used by final_test_release');p.add_argument('--native-review-set',type=Path);p.add_argument('--expected-native-review-set-sha256');a=p.parse_args()
 b=bundle_at(a.bundle,a.expected_bundle_sha256);release=release_bundle(b,a.release_review,a.expected_release_review_sha256)
 if a.command=='release':
  from mri_rl.final_test_loader import make_released_loader
  from mri_rl.final_test_release import release_projection
  loader=make_released_loader(release_bundle=release,**b['loader_arguments'])
  _,_,receipt=release_projection(b['plan'],b['inventory'],b['bindings'],release['approval'],expected_approval_sha256=a.expected_release_review_sha256,load_released_rows=loader,contract_path=b['contract_path'],destination=b['projection_directory']);print(json.dumps({'status':receipt['status'],'cases':receipt['cases']}))
 elif a.command=='scopes':
  result=prepare_scopes(b,release);print(json.dumps({'status':result['status'],'checkpoints':len(result['runs'])}))
 else:
  require(a.native_review_set and a.expected_native_review_set_sha256,'Externally authenticated native reviews required')
  print(json.dumps(run_serial(b,release,Path(b['execution_directory'])/'scope_set_REQUIRES_INDEPENDENT_REVIEW_PRIVATE.json',a.native_review_set,a.expected_native_review_set_sha256)))
if __name__=='__main__':main()
