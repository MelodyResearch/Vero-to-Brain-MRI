"""One newly authorized V2 slot: retire ONLY a verified idle queue controller.
Uses the reviewed no-child signal helper, never historical P_raw13 hooks.
No model launch, training edit, forced lock, cursor edit or SIGKILL.
"""
import argparse,hashlib,importlib.util,json,os,shlex
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def require(x,msg):
 if not x:raise ValueError(msg)
def check_boundary(c,helper):
 state=load(c['cursor']);helper.quiet_cursor(c['cursor'],c['stage_id'])
 require(state['stages'][c['stage_id']]['state']=='DONE','Completed target required')
 require(not any(v['state']=='RUNNING' for v in state['stages'].values()),'Another stage reserved')
 for key in ('exit','proof','final_receipt'):
  require(sha(c[key+'_path'])==c[key+'_sha256'],'Completed evidence changed: '+key)
 witness=load(c['exit_path']);proof=load(c['proof_path']);final=load(c['final_receipt_path'])
 require(witness['exit_code']==0 and witness['stage_id']==c['stage_id'] and witness['raw_log_sha256']==sha(c['raw_log']),'Actual closed exit binding differs')
 require(proof['full_arm_completed'] and proof['verdict']=='PASS_WITH_LIMITATIONS' and proof['stage_id']==c['stage_id'],'Independent terminal proof required')
 require(final['completed_groups']==proof['completed_groups']==state['stages'][c['stage_id']]['summary']['completed_groups'] and final['optimizer_updates']==proof['optimizer_updates'],'Terminal counters differ')
 require(sha(c['manifest'])==c['manifest_sha256'] and sha(c['executor'])==c['executor_sha256'],'Frozen original release changed')
 return {'stage_id':c['stage_id'],'completed_groups':proof['completed_groups'],'optimizer_updates':proof['optimizer_updates'],'cursor_sha256':sha(c['cursor']),'no_other_RUNNING_stage':True}
def main():
 p=argparse.ArgumentParser();p.add_argument('contract',type=Path);p.add_argument('review',type=Path);p.add_argument('review_sha256');a=p.parse_args();c=load(a.contract);r=load(a.review)
 require(sha(a.review)==a.review_sha256 and r['verdict'] in ('PASS','PASS_WITH_LIMITATIONS') and r['reviewer_id'] and r['reviewer_kind']=='isolated_codex' and r['source_sha256']==sha(__file__) and r['contract_sha256']==sha(a.contract),'Exact independent handoff review required')
 require(c['user_authorization']=='ONE_COMPLETED_GSPO_BOUNDARY_V2_SLOT_2026-10-17' and sha(c['helper'])=='8799bb8a714af5a38768316f85fa209ea9f14c43675d9886a2420c72616f702b','Wrong helper/authorization')
 s=importlib.util.spec_from_file_location('reviewed_retire_helper',c['helper']);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
 observed=shlex.split(c['parent']['command']);require(observed[1:]==c['resume_argv'][1:] and c['resume_argv'][0]==load(c['manifest'])['lexical_python'],'Wrong queue argv')
 require('--execute' in observed and observed[observed.index('--expected-manifest-sha256')+1]==c['manifest_sha256'],'Not original approved executor')
 require(not Path(c['lease']).exists(),'One-off handoff already attempted');check_boundary(c,h)
 h.save(c['lease'],{'status':'ARMING_USER_AUTHORIZED_IDLE_BOUNDARY_ONLY','source_sha256':sha(__file__),'contract_sha256':sha(a.contract),'parent':c['parent'],'resume_argv':c['resume_argv'],'queue_lock_path':c['queue_lock'],'native_models_started':0})
 summary,status=h.retire_at_boundary(c['parent'],h.RealOps(),lambda:check_boundary(c,h),lambda:h.quiet_cursor(c['cursor'],c['stage_id']),timeout=30)
 if summary is None:
  h.save(c['lease'],{'status':status,'queue_left_running':True,'native_models_started':0,'contract_sha256':sha(a.contract)});print(json.dumps({'status':status}));return
 require(h.identity(c['parent']['pid']) is None or not h.alive(c['parent']),'Old controller still alive')
 # The old controller has released its own descriptors. Do not acquire/remove its lock.
 h.save(c['lease'],{'status':'IDLE_CONTROLLER_RETIRED_V2_SLOT_OPEN','source_sha256':sha(__file__),'contract_sha256':sha(a.contract),'review_sha256':a.review_sha256,'parent':c['parent'],'boundary':summary,'paused_cursor_sha256':sha(c['cursor']),'resume_argv':c['resume_argv'],'queue_lock_path':c['queue_lock'],'native_models_started':0,'original_queue_resume_required':True,'resume_after_V2_actual_release_and_resource_gates':True})
 print(json.dumps({'status':'IDLE_CONTROLLER_RETIRED_V2_SLOT_OPEN','lease_sha256':sha(c['lease'])}))
if __name__=='__main__':main()

