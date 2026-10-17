"""Independent completed R2 capture arithmetic; no native runtime or causal verdict."""
from pathlib import Path
import argparse,hashlib,json,math

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for key in ('capture','receipt','scope','prior-capture'):
  ap.add_argument('--'+key,required=True);ap.add_argument('--expected-'+key+'-sha256',required=True)
 ap.add_argument('--output',required=True);ap.add_argument('--reviewer-id',required=True);a=ap.parse_args();require(a.reviewer_id.strip(),'Actual reviewer identity required')
 for key in ('capture','receipt','scope','prior_capture'):require(sha(getattr(a,key))==getattr(a,'expected_'+key+'_sha256'),'Actual evidence hash differs')
 capture,receipt,scope,prior=(load(getattr(a,k)) for k in ('capture','receipt','scope','prior_capture'));v=capture['numerics_PRIVATE'];control=capture['shape_matched_control']
 require(receipt['capture_sha256']==a.expected_capture_sha256 and receipt['scope_sha256']==canonical(scope) and scope['prior_R1_capture_sha256']==a.expected_prior_capture_sha256,'Actual scope/capture/prior binding differs')
 require(sha(Path(__file__).with_name('run_c40_prefix_oracle_shape_v2.py'))==scope['runner_sha256']=='2d5687e2964ab9dac7a2feb6e87a45ba143e8b20fe15bf7fca360f6ef68df0ed','Actual runner source differs')
 require(scope['record_index']==9 and scope['checkpoint']=='B28' and scope['original_unchanged_forwards']==3 and scope['shape_control_forwards']==1 and scope['candidate_forwards']==4 and scope['weight_updates']==scope['backward_calls']==0,'Finite R2 scope differs')
 require(sha(scope['prefix_output'])==scope['prefix_output_sha256']==receipt['preserved_prefix_sha256'],'Nine preserved results changed')
 require(capture['oracle_gate']==receipt['oracle_gate']=='FAIL' and capture['unchanged_tolerance']==scope['prefix_tolerance']==.002 and receipt['frozen_experiment_completed'] is False,'Frozen gate was waived')
 require(v==prior['numerics_PRIVATE'],'Unchanged three-call captures differ from actual R1')
 labels=scope['record_PRIVATE']['candidate_labels'];require(labels==['different','same'],'Candidate order differs');require(v['P']==len(v['prompt_list'])==115 and v['visual']==0 and v['grid'] is None,'Prompt/no-image evidence differs')
 first,oracle,prefix=v['first'],v['oracle'],v['prefix_scores'];require(len(first)==len(oracle)==len(prefix)==2 and all(len(x)==2 for x in first),'Candidate order/shape differs')
 cross=[abs(x-y) for x,y in zip(*prefix)];short=[abs(x-y) for row in first for x,y in zip(row,oracle)];require(cross+short==v['errors'],'Original error decomposition differs')
 part=v['parts']['different'];require(part['prefix_ids']==v['parts']['same']['prefix_ids'],'Common prefix differs');n=len(part['prefix_ids']);target=len(part['target_ids']);length=v['P']+target-1;position=v['P']+n-1;future=length-position-1
 require((n,target,length,position,future)==(11,15,129,125,3),'Independent causal layout differs')
 for key,value in (('input_length',length),('score_position',position),('common_prefix_length',position+1),('future_padding_count',future),('future_padding_token',0),('unchanged_tolerance',.002)):require(control[key]==value,'Captured layout differs: '+key)
 pad=control['first_label_candidate_logps'];padded=[abs(x-y) for row in first for x,y in zip(row,pad)];require(padded==control['candidate_first_label_abs_errors'] and max(padded)==control['maximum_candidate_first_label_error'] and control['control_within_original_tolerance']==(max(padded)<=.002),'Padded error arithmetic differs')
 values=[x for row in first+prefix for x in row]+oracle+pad+v['errors'];require(all(math.isfinite(x) for x in values),'Nonfinite capture');require(all(x<=0 for row in first+prefix for x in row) and all(x<=0 for x in oracle+pad),'Positive log probability')
 require(control==receipt['shape_matched_control'] and receipt['short_oracle_maximum_error']==max(short),'Receipt numerical summary differs')
 require(receipt['adapter_tensors_exact']==504 and receipt['optimizer_step']==receipt['weight_updates']==0 and receipt['native_children_released'] is True and receipt['critical_safety_condition'] is False,'Read-only/release receipt differs')
 resources_path=Path(a.capture).parent/'resources_PRIVATE.jsonl';resources=[json.loads(x) for x in resources_path.read_text().splitlines()];require(resources and all(x['action']=='CONTINUE' and x['supervisor_level']=='NORMAL' for x in resources),'Resource journal contains nonnormal action')
 result={'schema':'B28-index9-R2-independent-shape-arithmetic-v1','verdict':'PASS_WITH_LIMITATIONS','reviewer_kind':'isolated_codex','reviewer_id':a.reviewer_id,'verifier_author':'/root/full_arm_execution_contract','source_sha256':sha(__file__),'evidence_tier':'REPRODUCED','input_sha256':{key:getattr(a,'expected_'+key+'_sha256') for key in ('capture','receipt','scope','prior_capture')},'actual_runner_sha256':scope['runner_sha256'],'all_captured_values_finite':True,'original_three_call_numerics_identical_to_R1':True,'candidate_order':labels,'cross_candidate_prefix_max_error':max(cross),'short_oracle_absolute_errors':short,'short_oracle_max_error':max(short),'padded_oracle_absolute_errors':padded,'padded_oracle_max_error':max(padded),'prompt_tokens':115,'fixed_prefix_tokens':11,'candidate_input_tokens':129,'short_oracle_input_tokens':126,'padded_input_tokens':length,'read_position_zero_based':position,'future_padding_count':future,'future_padding_token':0,'padded_within_original_tolerance':max(padded)<=.002,'frozen_oracle_gate':'FAIL','frozen_experiment_completed':False,'read_only_receipt':{k:receipt[k] for k in ('adapter_tensors_exact','optimizer_step','weight_updates','native_children_released','critical_safety_condition')},'resource_samples_all_NORMAL_CONTINUE':len(resources),'resource_journal_sha256':sha(resources_path),'native_calls_by_reviewer':0,'native_backward_independently_verified':False,'limitations':['The observed mismatch is associated with changed call length/shape in this case; a particular kernel cause is not identified.','One filler sequence does not prove general causal future-token invariance.','The original .002 oracle contract still fails; no replacement score or completed ten-record comparison is claimed.','This no-image engineering diagnostic provides no MRI understanding/generalization evidence.']}
 out=Path(a.output);require(not out.exists(),'Preserve previous proof');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps({'verdict':result['verdict'],'output_sha256':sha(out),'short_error':max(short),'padded_error':max(padded)}))
if __name__=='__main__':main()
