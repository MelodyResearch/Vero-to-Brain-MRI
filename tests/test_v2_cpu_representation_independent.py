"""Independent metric-domain regression; synthetic rows only."""
import numpy as np
import pytest
from mri_rl.v2_cpu_representation import metrics

def test_fractional_class_predictions_are_rejected_before_metric_computation():
    rows=[]
    for person in ('s1','s2'):
        for label in ('FLAIR','contrast-enhanced T1'):
            rows.append({'id':person+label,'ability':'mri_sequence','data_source':'OpenNeuro_ds007045','reward_model':{'ground_truth':label},'extra_info':{'patient_key':person,'internal_split':'internal_dev','patient_lineage':'BIDS_SUBJECT_VERIFIED','reward_type':'mri_sequence','level':91}})
    for person in ('a','b'):
        for candidate in ('a','b'):
            label='same' if person==candidate else 'different'
            rows.append({'id':person+candidate,'ability':'mri_visit_correspondence','data_source':'OpenNeuro_ds004856','reward_model':{'ground_truth':label},'extra_info':{'patient_key':person,'candidate_patient_key':candidate,'internal_split':'internal_dev','patient_lineage':'BIDS_SUBJECT_VERIFIED','reward_type':'mri_visit_correspondence','statistical_cluster_key':'cycle','query_source_sha256':'query'+person,'candidate_source_sha256':'candidate'+candidate,'level':0}})
    with pytest.raises(ValueError,match='Prediction coverage/domain'):
        metrics(rows,np.full(len(rows),.5))
