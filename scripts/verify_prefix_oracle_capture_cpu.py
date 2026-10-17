"""Independent scalar/vector arithmetic for the failed B28 index9 capture only."""
from pathlib import Path
import argparse,hashlib,json,math

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('capture','receipt','scope','output','reviewer-id'):p.add_argument('--'+k,required=True)
 for k in ('capture','receipt','scope'):p.add_argument('--expected-'+k+'-sha256',required=True)
 a=p.parse_args();require(a.reviewer_id.strip(),'Actual reviewer identity required')
 for key in ('capture','receipt','scope'):require(sha(getattr(a,key))==getattr(a,'expected_'+key+'_sha256'),'Trusted actual artifact differs: '+key)
 capture=load(a.capture);receipt=load(a.receipt);scope=load(a.scope);v=capture['numerics_PRIVATE']
 require(receipt['capture_sha256']==a.expected_capture_sha256 and receipt['scope_sha256']==canon(scope),'Actual forensic binding differs')
 require(scope['record_index']==9 and scope['checkpoint']=='B28' and scope['candidate_forwards']==3 and scope['backward_calls']==scope['weight_updates']==0,'Wrong finite forensic scope')
 require(capture['oracle_gate']==receipt['oracle_gate']=='FAIL' and capture['unchanged_tolerance']==scope['prefix_tolerance']==.002 and not receipt['frozen_experiment_completed'],'Oracle failure was waived')
 labels=scope['record_PRIVATE']['candidate_labels'];require(labels==['different','same'],'Candidate order differs')
 require(v['P']==len(v['prompt_list']) and v['visual']==0 and v['grid'] is None,'Prompt/visual evidence differs')
 require(len(v['first'])==len(v['oracle'])==len(v['prefix_scores'])==2 and all(len(x)==2 for x in v['first']),'Candidate/oracle vector shape differs')
 lengths={};logps=[]
 for i,label in enumerate(labels):
  part=v['parts'][label];c=v['candidates'][label];n=len(part['prefix_ids']);k=len(part['label_ids']);m=len(part['suffix_ids'])
  require(part==c['partition'] and part['target_ids']==part['prefix_ids']+part['label_ids']+part['suffix_ids'],'Target partition differs')
  require(k>0 and len(c['target_token_logps'])==n+k+m and part['label_logit_positions']==list(range(v['P']+n-1,v['P']+n+k-1)),'Causal target indexing differs')
  require(v['prefix_scores'][i]==c['target_token_logps'][:n] and v['label_first'][i]==part['label_ids'][0] and v['first'][i]==c['first_label_candidate_logps'],'Captured ordering differs')
  label_lp=c['target_token_logps'][n:n+k];require(math.isclose(math.fsum(label_lp),c['label_logp_sum'],abs_tol=1e-12) and math.isclose(math.fsum(label_lp)/k,c['label_logp_mean'],abs_tol=1e-12),'Captured label summaries differ')
  require(label_lp[0]==v['first'][i][i],'Own selected first logp differs')
  lengths[label]={'prompt_tokens':v['P'],'prefix_tokens':n,'label_tokens':k,'suffix_tokens':m,'target_tokens':n+k+m,'candidate_model_input_tokens':v['P']+n+k+m-1,'independent_oracle_input_tokens':v['P']+n,'first_label_causal_logit_position_zero_based':v['P']+n-1}
  logps.extend(c['target_token_logps']);logps.extend(v['first'][i]);logps.extend(v['prefix_scores'][i])
 require(v['parts'][labels[0]]['prefix_ids']==v['parts'][labels[1]]['prefix_ids'],'Candidate prefixes differ')
 logps.extend(v['oracle']);require(all(math.isfinite(x) and x<=0 for x in logps),'Nonfinite/positive captured logps')
 prefix=[abs(x-y) for x,y in zip(*v['prefix_scores'])];require(len(prefix)==len(v['prefix_scores'][0])==len(v['prefix_scores'][1]),'Cross-prefix shape differs')
 signed=[[x-y for x,y in zip(row,v['oracle'])] for row in v['first']];absolute=[[abs(x) for x in row] for row in signed];recomputed=prefix+[x for row in absolute for x in row]
 require(recomputed==v['errors'],'Captured aggregate errors differ from independently rebuilt order')
 require(max(recomputed)>.002,'Expected original gate failure not reproduced in arithmetic')
 different,same=v['first'][0];od,osame=v['oracle'];report={'schema':'B28-index9-forensic-independent-arithmetic-v1','verdict':'PASS_WITH_LIMITATIONS','reviewer_kind':'isolated_codex','reviewer_id':a.reviewer_id,'verifier_author':'/root/full_arm_execution_contract','source_sha256':sha(__file__),'evidence_tier':'REPRODUCED','frozen_oracle_gate':'FAIL','frozen_experiment_completed':False,'input_sha256':{'capture':a.expected_capture_sha256,'receipt':a.expected_receipt_sha256,'scope':a.expected_scope_sha256},'all_captured_logps_finite_nonpositive':True,'candidate_order':labels,'lengths':lengths,'cross_candidate_prefix_max_abs_error':max(prefix),'cross_candidate_first_vector_max_abs_error':max(abs(x-y) for x,y in zip(*v['first'])),'candidate_first_minus_short_oracle':signed,'candidate_first_vs_oracle_abs_errors':absolute,'aggregate_error_count':len(recomputed),'max_abs_error':max(recomputed),'original_threshold':.002,'candidate_same_minus_different_margin':same-different,'short_oracle_same_minus_different_margin':osame-od,'candidate_margin_minus_oracle_margin':(same-different)-(osame-od),'read_only_receipt':{k:receipt[k] for k in ('adapter_tensors_exact','optimizer_step','weight_updates','native_children_released','critical_safety_condition','queue_resume_safe')},'limitations':['This verifies arithmetic of captured vectors; no independent native forward/backward run.','Three captured calls differ in input sequence length; no causal kernel or native GSPO optimizer defect is established.','The original fixed-tolerance oracle gate remains failed; nine durable outputs do not constitute the ten-record comparison.']}
 dest=Path(a.output);require(not dest.exists(),'Preserve prior proof');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'verdict':report['verdict'],'max_abs_error':report['max_abs_error'],'output_sha256':sha(dest)}))
if __name__=='__main__':main()
