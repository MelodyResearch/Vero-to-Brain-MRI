import json
import pytest
from mri_rl import final_accuracy_sprint_v1 as s


def test_replication_requires_accuracy_and_class_protection():
    assert s.replication_allowed(dict(correct=19,invalid=0,class_correct=[10,9],both_correct_pairs=7))
    assert not s.replication_allowed(dict(correct=19,invalid=0,class_correct=[9,10],both_correct_pairs=7))
    assert not s.replication_allowed(dict(correct=18,invalid=0,class_correct=[10,8],both_correct_pairs=7))


def test_budget_never_refunds_or_repeats_claim(tmp_path):
    p=tmp_path/'ledger.json'
    s.claim_run(p,0,'scope')
    with pytest.raises(ValueError):s.claim_run(p,0,'scope')
    with pytest.raises(ValueError):s.claim_run(p,1,'different')
    s.claim_run(p,1,'scope');s.claim_run(p,2,'scope')
    with pytest.raises(ValueError):s.claim_run(p,19,'scope')


def test_input_binding_fails_on_changed_file(tmp_path):
    p=tmp_path/'x';p.write_bytes(b'original')
    pins={str(p):s.sha(p)}
    s.verify_files(pins)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError):s.verify_files(pins)


def test_metrics_keep_invalids_in_denominator_and_cluster_people():
    rows=[{'patient':'a','truth':'FLAIR','prediction':0},
          {'patient':'a','truth':'contrast-enhanced T1','prediction':1},
          {'patient':'b','truth':'FLAIR','prediction':-1},
          {'patient':'b','truth':'contrast-enhanced T1','prediction':0}]
    m=s.metrics(rows)
    assert m['accuracy']==.5 and m['invalid']==1 and m['patients']==2
    assert m['class_correct']==[1,1] and m['confusion']==[[1,0,1],[1,1,0]]
    assert m['patient_accuracy']=={'a':1.,'b':0.}


def dev_fixture():
    return [{'id':f'{p}_{z}_{y}', 'row_PRIVATE':{'ability':'mri_sequence','reward_model':{'ground_truth':y},
             'extra_info':{'patient_key':str(p),'level':z,'internal_split':'internal_dev'}}}
            for p in range(4) for z in (45,91,136) for y in s.LABELS]


def test_dev_rejects_duplicate_pair_member_before_native_training():
    rows=dev_fixture();assert len(s.validate_dev(rows))==4
    rows[0]['row_PRIVATE']['reward_model']['ground_truth']=s.LABELS[1]
    with pytest.raises(ValueError):s.validate_dev(rows)


def test_child_replication_requires_real_completed_native_witness(tmp_path,monkeypatch):
    from mri_rl import final_balanced_sft_entry_v1 as entry
    monkeypatch.setattr(entry,'PRIVATE',tmp_path)
    with pytest.raises(ValueError):entry.verify_replication(1,'scope')
    (tmp_path/'seed_0_PRIVATE').mkdir();(tmp_path/'seed_0_witness_PRIVATE').mkdir()
    (tmp_path/'seed_0_PRIVATE/complete_PRIVATE.json').write_text(json.dumps({'scope_file_sha256':'scope',
         'metrics_PRIVATE':dict(correct=19,invalid=0,class_correct=[10,9],both_correct_pairs=7)}))
    p=tmp_path/'seed_0_witness_PRIVATE/actual_exit_PRIVATE.json'
    w=dict(scope_file_sha256='scope',native_exit_code=0,child_reaped=True,guardian_stop=True)
    p.write_text(json.dumps(w))
    with pytest.raises(ValueError):entry.verify_replication(1,'scope')
    w['guardian_stop']=False;p.write_text(json.dumps(w));entry.verify_replication(1,'scope')
