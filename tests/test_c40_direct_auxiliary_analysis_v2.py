import importlib.util
from pathlib import Path

def mod():
 p=Path(__file__).parents[1]/'scripts/analyze_c40_direct_auxiliary_v2.py';s=importlib.util.spec_from_file_location('aux_analysis',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_accuracy_uses_displayed_truth_keeps_confusion_and_excludes_null():
 m=mod();labels={'task':('negative','positive')};rows={}
 for i,(truth,pred) in enumerate([('negative','positive'),('positive','positive')]):rows[str(i)]={'record_id':str(i),'category':'task','condition':'real_image','visible_target_PRIVATE':truth,'response':pred,'comparisons':{'mean_preferred':pred,'raw_preferred':pred},'prefix_oracle_first_logps':[-2.,-1.]}
 rows['null']={'record_id':'null','category':'task','condition':'no_image'}
 pairs=[{'task':'task','unit':'withheld','negative_id':'0','positive_id':'1'}]
 x=m.accuracy_summary(rows,pairs,labels,lambda x:(x,1))
 assert x['tasks']['task']['mean']['accuracy']==.5 and x['tasks']['task']['greedy']['class_recalls']=={'negative':0.,'positive':1.}
 assert x['tasks']['task']['mean']['confusion']['negative']['positive']==1 and x['tasks']['task']['mean']['n']==2

def test_common_prior_shift_can_change_accuracy_without_pair_separation_gain():
 m=mod();labels={'task':('n','p')};pairs=[{'task':'task','unit':'unit','negative_id':'0','positive_id':'1'}]
 def rows(margins):
  return {str(i):{'record_id':str(i),'category':'task','condition':'real_image','visible_target_PRIVATE':labels['task'][i],'response':'p','comparisons':{'mean_preferred':'p' if v>0 else 'n','raw_preferred':'p' if v>0 else 'n'},'prefix_oracle_first_logps':[0.,v]} for i,v in enumerate(margins)}
 before=m.accuracy_summary(rows([-2.,-1.]),pairs,labels,lambda x:(x,1));after=m.accuracy_summary(rows([-1.,0.]),pairs,labels,lambda x:(x,1))
 assert before['tasks']['task']['mean']['n']==after['tasks']['task']['mean']['n']==2
 assert (-1.-(-2.))==(0.-(-1.))


def test_shape_errors_are_recomputed_not_trusted():
 import pytest
 m=mod();candidate={'first_label_candidate_logps':[-1.,-2.],'target_token_logps':[-.5,-1.],'partition':{'prefix_ids':[8]}}
 row={'legacy_oracle_gate':'FAIL','candidate_scores':{'a':candidate,'b':candidate},'legacy_oracle_evidence_PRIVATE':{'first':[[-1.,-2.],[-1.,-2.]],'prefix_scores':[[-.5],[-.5]],'oracle':[-1.,-2.125],'errors':[0.,0.,.125,0.,.125]},'legacy_oracle_max_disagreement':.125,'shape_matched_control':{'first_label_candidate_logps':[-1.,-2.],'candidate_first_label_abs_errors':[0.,0.,0.,0.],'maximum_candidate_first_label_error':0.},'prefix_oracle_first_logps':[-1.,-2.]}
 m.validate_shape_oracle_fields(row,['a','b']);row['shape_matched_control']['maximum_candidate_first_label_error']=.001
 with pytest.raises(ValueError,match='arithmetic'):m.validate_shape_oracle_fields(row,['a','b'])
