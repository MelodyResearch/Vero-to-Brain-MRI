"""Check user-facing V2 cohort selection and scoring without native model imports."""
import json
import sys
import types
import hashlib
from pathlib import Path
import pytest
from mri_rl.paired_sequence_accuracy_v2 import LABELS
from mri_rl import paired_sequence_accuracy_v2_entry as entry


def make_sequence_rows(directory, people, split):
    rows=[]
    for p in people:
        for level in (45,91):
            for mod,label in (("flair",LABELS[0]),("t1ce",LABELS[1])):
                name=f"{p}_z{level:03d}_{mod}_sequence"
                image=directory / f"{p}_z{level:03d}_{mod}.png"
                image.write_bytes((split+name).encode())
                q="Sequence?"
                rows.append({"id":name,"ability":"mri_sequence","data_source":"OpenNeuro_ds007045",
                             "images":[str(image)],"reward_model":{"ground_truth":label},
                             "extra_info":{"patient_key":p,"patient_lineage":"BIDS_SUBJECT_VERIFIED",
                                           "internal_split":split,"question":q},
                             "prompt":[{"role":"user","content":[
                                 {"type":"image","image":str(image)},{"type":"text","text":q}]}]})
    return rows


def saved(tmp_path, train_people=("a","b","c"), dev_people=("d","e")):
    t=tmp_path/"train.verl.jsonl"
    d=tmp_path/"internal_dev.verl.jsonl"
    t.write_text(''.join(json.dumps(row)+'\n' for row in make_sequence_rows(tmp_path,train_people,"train")))
    d.write_text(''.join(json.dumps(row)+'\n' for row in make_sequence_rows(tmp_path,dev_people,"internal_dev")))
    return t,d


def test_prepare_produces_approved_images_and_disjoint_people(tmp_path):
    train,dev=saved(tmp_path)
    result=entry.prepare(train,dev,seed=7,steps=3,levels=(45,91))
    assert result["pairs"]==6
    assert result["train_people"]==3
    assert result["dev_people"]==2
    assert len(result["schedule"])==3
    assert len({s["patient_key"] for s in result["schedule"]})==3
    for step in result["schedule"]:
        entry.validate_step(step)
        assert step["extra_info"]["reward_type"]=="paired-image-label-margin-v2"
    assert {s["reward_model"]["ground_truth"] for s in result["dev"]}==set(LABELS)


def test_patient_overlap_and_development_leak_blocked(tmp_path):
    train,dev=saved(tmp_path,dev_people=("a",))
    with pytest.raises(ValueError,match="overlaps"):
        entry.prepare(train,dev,seed=7,steps=1,levels=(45,91))
    # The developer cannot sneak a testing patient into source train input.
    rows=entry.read_jsonl(train)
    rows[0]["extra_info"]["internal_split"]="test"
    train.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match="training"):
        entry.prepare(train,dev,seed=7,steps=1,levels=(45,91))


def test_run_scope_checks_without_native_models(monkeypatch,tmp_path):
    class FakeCR:
        @staticmethod
        def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
        @staticmethod
        def _binding(schedule,bindings,config,validator):
            for s in schedule: validator(s)
            return {"schedule_record_ids":[s["id"] for s in schedule],"experiment":bindings,
                    "configuration":config,"schedule_sha256":"p"}
        @staticmethod
        def inspect_boundaries(output,binding):return None
    monkeypatch.setitem(sys.modules,"mri_rl.continuous_rl",FakeCR)
    monkeypatch.setattr(entry,"source_hashes",lambda:{"mock":"test-only"})
    train,dev=saved(tmp_path)
    prepared=entry.prepare(train,dev,seed=7,steps=2,levels=(45,91))
    model=tmp_path/"model";model.mkdir();(model/"config.json").write_text('{}')
    adapter=tmp_path/"adapter.safetensors";adapter.write_bytes(b'synthetic')
    contract=tmp_path/"contract.json";contract.write_text('{}')
    b=entry.run_binding(prepared,train=train,dev=dev,adapter=adapter,seed=7,model=model,contract=contract)
    assert b["experiment"]["all_source_image_sha256"]==entry.all_pair_image_digest(prepared["schedule"])
    scope=entry.training_scope(b,output=tmp_path/"new",resume=False,max_new_steps=1)
    assert scope["maximum_new_optimizer_steps"]==1 and len(scope["next_record_ids"])==1
    with pytest.raises(ValueError):entry.training_scope(b,output=tmp_path/"x",resume=False,max_new_steps=12)


def test_scoring_strict_accuracy_is_distinct_from_forced_prefix(monkeypatch,tmp_path):
    train,dev=saved(tmp_path)
    rows=entry.prepare(train,dev,seed=7,steps=2,levels=(45,91))["dev"]
    fake_cr=types.SimpleNamespace(canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':')),
                                  require=lambda ok,msg: None if ok else (_ for _ in ()).throw(ValueError(msg)))
    fake_reward=types.SimpleNamespace(compute_score_from_data_source=lambda src,text,gold,info:
                                      {"accuracy":float(text==gold),"format":1})
    monkeypatch.setitem(sys.modules,"mri_rl.continuous_rl",fake_cr)
    monkeypatch.setitem(sys.modules,"mri_rl.vero_mri_reward",fake_reward)
    class FakeBackend:
        def score_sequence(self,row,expected):
            assert expected==hashlib.sha256(Path(row['images'][0]).read_bytes()).hexdigest()
            return {"record_id":row['id'],"gold":row['reward_model']['ground_truth'],
                    "forced_prefix_correct":True}
        def generate_record(self, record, imagehash, limit):
            return {"response":"FLAIR", "finish_reason":"stop","generation_tokens":2}
    contract=types.SimpleNamespace(system_instruction="MRI")
    output=tmp_path/"dev_outputs.jsonl"
    r=entry.score_internal_dev(rows=rows,backend=FakeBackend(),output=output,contract=contract)
    assert r["candidate_accuracy"]==1
    assert r["strict_greedy_accuracy"]==.5
    assert r["patients"]==2
    r2=entry.score_internal_dev(rows=rows,backend=FakeBackend(),output=output,contract=contract,resume=True)
    assert r2==r
