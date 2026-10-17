import importlib.util,json,math,struct,subprocess,sys
from pathlib import Path
import pytest
P=Path(__file__).parents[1]/'scripts/verify_early_probe_numerics_cpu.py'
sp=importlib.util.spec_from_file_location('independent_early_math',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def tensor(path,values):
 raw=struct.pack('<'+'f'*len(values),*values);header=json.dumps({'x':{'dtype':'F32','shape':[len(values)],'data_offsets':[0,len(raw)]}}).encode();path.write_bytes(struct.pack('<Q',len(header))+header+raw)
def test_independent_f32_gradient_math(tmp_path):
 a,b=tmp_path/'a',tmp_path/'b';tensor(a,[3,4]);tensor(b,[0,4]);s=m.gradient_metrics(a,b,1)
 assert s['real_l2']==5 and s['zero_l2']==4 and s['difference_l2']==3 and s['max_abs_component_difference']==3 and s['cosine_similarity']==.8
 assert s['elements']==2
@pytest.mark.parametrize('values',[[float('nan')],[float('inf')]])
def test_nonfinite_gradient_rejected(tmp_path,values):
 a,b=tmp_path/'a',tmp_path/'b';tensor(a,values);tensor(b,[1])
 with pytest.raises(ValueError):m.gradient_metrics(a,b,1)
def test_zero_norm_cosine_is_not_fabricated(tmp_path):
 a,b=tmp_path/'a',tmp_path/'b';tensor(a,[0,0]);tensor(b,[1,1]);assert m.gradient_metrics(a,b,1)['cosine_similarity'] is None
def test_tensor_trailing_bytes_fail(tmp_path):
 p=tmp_path/'a';tensor(p,[1]);p.write_bytes(p.read_bytes()+b'extra')
 with pytest.raises(ValueError):m.safetensors_header(p)
def test_raw_mean_first_are_independently_reconstructed():
 labels=m.LABELS['mri_sequence'];c={}
 for label,logps in zip(labels,([-1,-2,-4,-1],[-1,-1,-1,-1])):
  c[label]={'partition':{'prefix_ids':[1],'label_ids':[2,3],'target_ids':[1,2,3,4]},'target_token_logps':logps,'label_logp_sum':sum(logps[1:3]),'label_logp_mean':sum(logps[1:3])/2}
 row={'category':'mri_sequence','candidate_order':list(labels),'candidate_scores':c,'prefix_oracle_first_logps':[-2,-1],'comparisons':{'mean_t1_minus_flair':2,'raw_t1_minus_flair':4}}
 v,e=m.record_margins(row);assert v=={'mean':2,'raw':4,'first_label':1} and e==0
 row['comparisons']['mean_t1_minus_flair']=99
 with pytest.raises(ValueError):m.record_margins(row)

def test_cli_requires_actual_reviewer_identity():
 result=subprocess.run([sys.executable,str(P),'--evidence-index','unused','--expected-evidence-sha256','unused','--output','unused','--private-output','unused'],text=True,capture_output=True)
 assert result.returncode==2 and '--reviewer-id' in result.stderr
