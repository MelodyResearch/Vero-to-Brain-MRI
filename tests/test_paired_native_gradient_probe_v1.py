import importlib.util
from pathlib import Path
import numpy as np
import pytest

def module():
 p=Path(__file__).resolve().parents[1]/'src/mri_rl/paired_native_gradient_probe_v1.py';s=importlib.util.spec_from_file_location('pair_grad',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_serial_route_has_six_primitive_calls_and_truth_fixed():
 m=module();pixels=[np.ones((2,3)),np.full((2,3),2.)];inputs=[{'input_ids':np.array([[1,2]]),'pixel_values':v,'image_grid_thw':[[1,2,2]]} for v in pixels];seen=[]
 def vg(model,ids,pixel,kwargs,parts,truth):
  assert 'pixel_values' not in kwargs and kwargs['image_grid_thw']==[[1,2,2]];seen.append((truth,float(pixel.sum())));return (np.array(float(pixel.sum())+truth),np.array([pixel.mean(),pixel.mean()+truth])),{'lora':np.array([pixel.sum()])}
 def tree_map(fn,*trees):return {k:fn(*(t[k] for t in trees)) for k in trees[0]}
 for condition in m.CONDITIONS:
  gradient,losses,scores=m.serial_pair_gradient(vg,None,inputs,m.route_pixels(condition,pixels,np.zeros_like),[None,None],tree_map,lambda *args:None,lambda *args:None,lambda:None);assert len(losses)==len(scores)==2
 assert seen==[(0,6.),(1,12.),(0,12.),(1,6.),(0,0.),(1,0.)];assert m.BUDGET['primitive_one_image_CE_gradients']==6 and m.BUDGET['candidate_forwards']==12 and m.BUDGET['optimizer_updates']==0

def test_wrong_shape_and_stale_image_feature_route_rejected():
 m=module()
 with pytest.raises(ValueError):m.route_pixels(m.CONDITIONS[1],[np.zeros(2),np.zeros(3)],np.zeros_like)
 inputs=[{'input_ids':np.array([[1]]),'cached_image_features':np.ones(2)}]*2
 with pytest.raises(ValueError):m.serial_pair_gradient(lambda *args:pytest.fail('Stale cached route reached native scorer'),None,inputs,[np.zeros(2)]*2,[None]*2,None,None,lambda *args:None,None)

def test_training_pair_source_order_and_level_fail_closed():
 m=module()
 def row(label,image):return {'ability':'mri_sequence','images':[image],'reward_model':{'ground_truth':label},'extra_info':{'internal_split':'train','patient_key':'person','level':45,'question':'same'}}
 pair=[row('FLAIR','a.png'),row('contrast-enhanced T1','b.png')];m.validate_pair(pair)
 with pytest.raises(ValueError):m.validate_pair(pair[::-1])
 pair[1]['extra_info']['level']=91
 with pytest.raises(ValueError):m.validate_pair(pair)

def test_gradient_metrics_detect_difference_and_handle_zero_norm():
 m=module();chunks=[(str(i),[2],np.array([1.,2.]),np.array([1.,2.])) for i in range(504)];r=m.vector_comparison(chunks);assert r['difference_l2']==0 and r['cosine']==pytest.approx(1.)
 z=[(str(i),[2],np.ones(2),np.zeros(2)) for i in range(504)];r=m.vector_comparison(z);assert r['cosine'] is None and r['difference_l2']==r['left_l2']
