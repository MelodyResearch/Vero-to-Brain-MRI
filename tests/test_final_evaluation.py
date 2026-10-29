import copy,json,subprocess,sys
from pathlib import Path
import pytest
from mri_rl.final_evaluation import validate_protocol,validate_rows,summarize,check_execution_permission
ROOT=Path(__file__).resolve().parents[1]
def protocol():return json.loads((ROOT/'configs/final_evaluation_v1.json').read_text())
def rows():
 return [{'patient':f'p{i}','truth':label,'image_sha256':f'{i*2+j:064x}','row_PRIVATE':{'id':f'r{i}_{j}'}} for i in range(16) for j,label in enumerate(['FLAIR','contrast-enhanced T1'])]
def test_fixed_protocol_and_all_pairs_required():
 p=protocol();validate_protocol(p);validate_rows(rows(),[])
 with pytest.raises(ValueError):validate_rows(rows()[:-1],[])
 bad=rows();bad[1]['truth']='FLAIR'
 with pytest.raises(ValueError):validate_rows(bad,[])
 with pytest.raises(ValueError):validate_rows(rows(),['p0'])
def test_no_training_or_parent_substitution():
 p=protocol();p['budget']['optimizer_updates']=1
 with pytest.raises(ValueError):validate_protocol(p)
 p=protocol();p['adapter_sha256']='0'*64
 with pytest.raises(ValueError):validate_protocol(p)
def test_no_output_or_approval_is_permission():
 with pytest.raises(ValueError):check_execution_permission(protocol(),{}, {},'a'*64)
def test_invalids_stay_in_denominator_and_pair_summary():
 r=rows()
 for i,x in enumerate(r):x['prediction']=i%2
 r[0]['prediction']=None;m=summarize(r,[])
 assert m['correct']==31 and m['invalid']==1 and m['patients']==16 and m['fully_correct_pairs']==15
 assert m['accuracy']==31/32 and m['class_recall']==[15/16,1.]
 assert len(m['patient_cluster_bootstrap95'])==2
 assert m['quality_flagged_patients']==0

def test_validation_imports_no_native_and_stops_without_inputs():
 cmd=[sys.executable,'scripts/evaluate.py','--validate-only']
 p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
 assert p.returncode==2 and 'NOT_RUN' in p.stdout and 'INPUT' in p.stdout
 assert 'mlx.core' not in sys.modules

def test_reviewed_output_cannot_be_replayed_elsewhere(tmp_path):
 from mri_rl.final_evaluation import check_output_binding
 check_output_binding({'output':str(tmp_path/'fixed')},tmp_path/'fixed')
 with pytest.raises(ValueError):check_output_binding({'output':str(tmp_path/'fixed')},tmp_path/'different')

def test_partial_constructor_stays_owned():
 from mri_rl.final_evaluation import create_owned
 class Failing:
  def __init__(self):self.partial=True;raise RuntimeError('partial load')
 resident=[None]
 with pytest.raises(RuntimeError):create_owned(Failing,resident)
 assert resident[0].partial

def test_fixed_supervisor_deadline_does_not_sample_after_expiration():
 from mri_rl.final_evaluation import deadline_sample
 called=[]
 with pytest.raises(TimeoutError):deadline_sample(10,lambda pid:called.append(pid),123,now=lambda:11)
 assert not called
 assert deadline_sample(10,lambda pid:pid,123,now=lambda:9)==123

def test_payload_hash_paths_must_be_the_actual_runtime_locations(tmp_path):
 from mri_rl.final_evaluation import validate_payload_locations
 runtime={'parent_checkpoint':str(tmp_path/'parent'),'cache_dir':str(tmp_path/'cache'),'model':str(tmp_path/'model'),'cache_certificate':str(tmp_path/'cert.json'),'contract':str(tmp_path/'contract.json')}
 scope={'payload_paths':{'Parent_checkpoint.adapter.safetensors':str(tmp_path/'parent/adapter.safetensors'),'base_cache.config.json':str(tmp_path/'cache/config.json')}}
 validate_payload_locations(scope,runtime,scope['payload_paths'])
 scope['payload_paths']['base_cache.config.json']=str(tmp_path/'unused/config.json')
 with pytest.raises(ValueError):validate_payload_locations(scope,runtime,scope['payload_paths'])
