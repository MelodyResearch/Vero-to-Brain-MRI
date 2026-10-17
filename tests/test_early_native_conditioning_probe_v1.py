
import math
from types import SimpleNamespace as S
import pytest
from mri_rl.early_native_conditioning_probe_v1 import (
 capture_stream,logprob_comparison,validate_condition_fingerprints,
 gradient_summary,preserve_active,check_weights,CONTRACT,validate_logprob_shape)

def packet(token,number,*,finish=None,ids=None,prompt=7,value=-2.):
 probs=[[-3.]*10];probs[0][token]=value
 return S(token=token,generation_tokens=number,prompt_tokens=prompt,logprobs=probs,finish_reason=finish,token_ids=ids)
def pick(values,token):
 validate_logprob_shape((len(values),len(values[0])),token)
 return float(values[0][token])
def collect(packets,eos=(9,)):
 return capture_stream(iter(packets),selected_logp=pick,before_unit=lambda _:None,eos_ids=set(eos),prompt_tokens=7,max_tokens=32)

def test_stream_captures_finalEOS_and_deduplicates_length_final_packet():
 eos=collect([packet(2,1),packet(9,2,finish='stop',ids=[2,9])])
 assert eos['token_ids']==[2,9] and eos['raw_selected_logps']==[-2.,-2.] and eos['eos_observed']
 length=collect([packet(2,1),packet(2,1,finish='length',ids=[2])])
 assert length['token_ids']==[2] and not length['eos_observed']
 assert length['EOS_coverage']=='NOT_OBSERVED_LENGTH_LIMIT'
 assert CONTRACT['teacherforced_forwards']==2 and CONTRACT['backward_calls']==2 and CONTRACT['optimizer_updates']==0

@pytest.mark.parametrize('packets',[
 [packet(2,1)], # no final packet
 [packet(2,2,finish='length',ids=[2])], # shifted position
 [packet(2,1),packet(2,1,finish='length',ids=[3])], # final IDs differ
 [packet(2,1,finish='stop',ids=[2])], # stop not EOS
 [packet(2,1,prompt=8,finish='length',ids=[2])],
 [packet(2,1,value=float('nan'),finish='length',ids=[2])],
 [packet(2,1),packet(2,1,value=-1.,finish='length',ids=[2])],
 [packet(2,33,finish='length',ids=[2])],
])
def test_missing_shifted_nonfinite_or_inconsistent_stream_fails(packets):
 with pytest.raises(ValueError):collect(packets)

def test_shape_and_raw_teacher_deltas_measured_without_invented_threshold():
 with pytest.raises(ValueError):validate_logprob_shape((2,10),1)
 with pytest.raises(ValueError):validate_logprob_shape((1,10),10)
 measured=logprob_comparison([-2.,-3.],[-2.,-4.])
 assert measured['max_abs_delta']==1 and measured['mean_abs_delta']==.5
 assert measured['numerical_closeness_verdict']=='NOT_VERIFIED'
 assert measured['per_token'][1]['teacherforced_minus_stream']==-1
 for left,right in (([-2.],[-2.,-3.]),([float('inf')],[-2.])):
  with pytest.raises(ValueError):logprob_comparison(left,right)

def test_zero_pixels_keep_prompt_mask_grid_dtype_and_shape():
 real={'input_ids':{'shape':[1,7],'dtype':'int32','sha256':'a'},'attention_mask':{'shape':[1,7],'dtype':'int32','sha256':'b'},
       'image_grid_thw':{'shape':[1,3],'dtype':'int32','sha256':'c'},'pixel_values':{'shape':[4,8],'dtype':'float32','sha256':'r','all_zero':False}}
 zero={**real,'pixel_values':{**real['pixel_values'],'sha256':'z','all_zero':True}}
 validate_condition_fingerprints(real,zero)
 for key,wrong in (('input_ids',{'shape':[1,7],'dtype':'int32','sha256':'changed'}),('pixel_values',{**zero['pixel_values'],'shape':[1,8]}),('pixel_values',{**zero['pixel_values'],'all_zero':False})):
  with pytest.raises(ValueError):validate_condition_fingerprints(real,{**zero,key:wrong})

def test_gradient_separation_and_zero_gradient_are_observations_not_pass():
 separated=gradient_summary([('lora',(2,),[1.,0.],[0.,1.])])
 assert separated['real_l2']==separated['zero_l2']==1
 assert separated['difference_l2']==math.sqrt(2) and separated['cosine_similarity']==0
 equal=gradient_summary([('lora',(2,),[0.,0.],[0.,0.])])
 assert equal['difference_l2']==0 and equal['cosine_similarity'] is None
 assert equal['conditioning_interpretation']=='NOT_VERIFIED'
 with pytest.raises(ValueError):gradient_summary([('lora',(3,),[1.,0.],[1.,0.])])

def test_active_restored_on_failure_and_weight_mutation_rejected():
 prior=object();backend=S(active=prior)
 with pytest.raises(RuntimeError):
  with preserve_active(backend):
   backend.active={'temporary':True};raise RuntimeError('fixture')
 assert backend.active is prior
 assert check_weights({'lora':'a'},{'lora':'a'}) is True
 with pytest.raises(ValueError):check_weights({'lora':'a'},{'lora':'changed'})
