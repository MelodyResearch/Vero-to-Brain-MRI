import copy
import numpy as np
import pytest
from PIL import Image
from mri_rl.v2_cpu_representation import validate_pack,pixel_features,fit_predict,metrics,paired_units,pair_features,apply_fitted

def row(person,label,level=91,split='train',task='mri_sequence',candidate=None,cycle=None):
 e={'patient_key':person,'internal_split':split,'patient_lineage':'BIDS_SUBJECT_VERIFIED','reward_type':task,'level':level}
 if task!='mri_sequence':e.update(candidate_patient_key=candidate,statistical_cluster_key=cycle,query_source_sha256='q'+person,candidate_source_sha256='c'+candidate)
 return {'id':f'{person}_{level}_{label}','ability':task,'data_source':'OpenNeuro_ds007045' if task=='mri_sequence' else 'OpenNeuro_ds004856','reward_model':{'ground_truth':label},'extra_info':e,'images':['unused']}
def pack(split,people):
 s=[row(p,l,z,split) for p in people[:2] for z in (45,91,136) for l in ('FLAIR','contrast-enhanced T1')]
 s += [row(p,'same' if p==c else 'different',0,split,'mri_visit_correspondence',c,'cycle'+split) for p in people[2:] for c in people[2:]]
 return s

def test_pairing_patient_cycle_disjoint_and_no_test_required():
 a,b=pack('train',['a','b','c','d']),pack('internal_dev',['e','f','g','h'])
 validate_pack(a,b)
 with pytest.raises(ValueError):validate_pack(a[:-1],b)
 wrong=copy.deepcopy(a);wrong[0]['extra_info']['internal_split']='test'
 with pytest.raises(ValueError):validate_pack(wrong,b)
 wrong=copy.deepcopy(a);wrong[1]['extra_info']['level']=92
 with pytest.raises(ValueError):validate_pack(wrong,b)
 with pytest.raises(ValueError):validate_pack(a,pack('internal_dev',['a','f','g','h']))

def test_histogram_permutation_invariance_and_spatial_sensitivity():
 x=np.arange(256,dtype=np.uint8).reshape(16,16);a,h=pixel_features(Image.fromarray(x));b,j=pixel_features(Image.fromarray(x[::-1]))
 assert np.array_equal(h,j) and not np.array_equal(a,b)
 assert np.array_equal(a,x.ravel()/255) and np.array_equal(h,np.ones(16)/16)

def test_train_only_scaling_and_actual_fitted_regularized_head():
 train=np.array([[-2.],[-1.],[1.],[2.]])
 p,proof=fit_predict(train,np.array([0,0,1,1]),np.array([[-5.],[5.]]))
 assert np.array_equal(p,apply_fitted(proof,np.array([[-5.],[5.]])))
 assert list(p)==[0,1] and proof['trained_rows']==4 and proof['fitted_parameters']>0
 assert np.allclose(proof['training_feature_mean'],[0.])
 other,proof2=fit_predict(train,np.array([0,0,1,1]),np.array([[-500.],[500.]]))
 assert proof2['training_feature_mean']==proof['training_feature_mean']
 with pytest.raises(ValueError):fit_predict(train,np.array([0,0,0,0]),train)

def test_metrics_macro_groups_not_rows_and_paired_units():
 rows=pack('internal_dev',['e','f','g','h']);truth=np.array([int(r['reward_model']['ground_truth'] in ('contrast-enhanced T1','same')) for r in rows]);guess=truth.copy();guess[:6]=1-guess[:6]
 scored=metrics(rows,guess);assert scored['tasks']['mri_sequence']['unit_macro_accuracy']==.5
 assert scored['tasks']['mri_visit_correspondence']['unit_macro_accuracy']==1
 assert scored['equal_task_unit_macro']==.75 and scored['worst_task_accuracy']==.5
 assert scored['tasks']['mri_sequence']['class_recall']=={'FLAIR':.5,'contrast-enhanced T1':.5}
 paired=paired_units(rows,truth,guess);assert paired['mri_sequence']['mean_difference']==-.5
 duplicate=rows+[rows[0]]
 with pytest.raises(ValueError):metrics(duplicate,np.append(guess,guess[0]))

def test_correspondence_features_are_symmetric_under_panel_reversal():
 a=Image.fromarray(np.arange(256,dtype=np.uint8).reshape(16,16));b=Image.fromarray(np.zeros((16,16),dtype=np.uint8));left=pair_features(a,b);right=pair_features(b,a)
 assert all(np.array_equal(x,y) for x,y in zip(left,right))
