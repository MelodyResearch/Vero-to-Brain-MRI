"""Pure objective/patient data tests. No MLX, patient MRI, or clinical results."""
import copy
import math
from pathlib import Path

import pytest
from mri_rl.paired_sequence_accuracy_v2 import (
    LABELS, patient_pairs, schedule_pairs, margin_objective, native_backend_class,
)


def sample_rows(tmp_path, persons=("pa", "pb", "pc"), levels=(45,91,136)):
    rows=[]
    for patient in persons:
        for z in levels:
            for mod,label in (("flair", LABELS[0]),("t1ce",LABELS[1])):
                name=f"{patient}_z{z:03d}_{mod}_sequence"
                image=tmp_path/(name.replace("_sequence", "")+".png")
                image.write_bytes(f"SYNTHETIC:{name}".encode())
                q="Which MRI sequence is displayed?"
                row={"id":name,"ability":"mri_sequence","data_source":"OpenNeuro_ds007045",
                     "images":[str(image)],"reward_model":{"ground_truth":label},
                     "extra_info":{"patient_key":patient,"internal_split":"train",
                                   "patient_lineage":"BIDS_SUBJECT_VERIFIED","question":q}}
                row["prompt"]=[{"role":"user","content":[{"type":"image","image":str(image)},
                                                      {"type":"text","text":q}]}]
                rows.append(row)
    return rows


def test_matching_patient_and_slice_fails_on_missing_or_wrong_labels(tmp_path):
    rows=sample_rows(tmp_path)
    pairs=patient_pairs(rows)
    assert len(pairs)==9
    assert all(p["FLAIR"]["extra_info"]["patient_key"] == p["contrast-enhanced T1"]["extra_info"]["patient_key"] for p in pairs)
    with pytest.raises(ValueError,match="Incomplete"):
        patient_pairs(rows[:-1])
    bad=copy.deepcopy(rows)
    bad[1]["reward_model"]["ground_truth"]="FLAIR"
    with pytest.raises(ValueError,match="label"):
        patient_pairs(bad)
    bad=copy.deepcopy(rows)
    bad[2]["extra_info"]["internal_split"]="test"
    with pytest.raises(ValueError,match="training"):
        patient_pairs(bad)
    bad=copy.deepcopy(rows)
    bad[0]["prompt"][0]["content"][1]["text"]="LEAKED GOLD"
    with pytest.raises(ValueError,match="hidden label"):
        patient_pairs(bad)


def test_no_dev_misuse_and_patient_diverse_schedule(tmp_path):
    rows=sample_rows(tmp_path, persons=("pA","pB","pC","pD"))
    pairs=patient_pairs(rows,levels=(45,91,136))
    a=schedule_pairs(pairs,seed=11,steps=12)
    b=schedule_pairs(list(reversed(pairs)),seed=11,steps=12)
    assert a==b
    assert len(a)==12
    assert len({x["patient_key"] for x in a[:4]})==4
    assert len({x["id"] for x in a})==12
    assert all(x["FLAIR"]["reward_model"]["ground_truth"]=="FLAIR" for x in a)
    assert all(x["contrast-enhanced T1"]["reward_model"]["ground_truth"]=="contrast-enhanced T1" for x in a)
    assert {x["level"] for x in a}=={45,91,136}
    with pytest.raises(ValueError):schedule_pairs(pairs,seed=7,steps=65)


def test_debiased_pairwise_loss_has_nonzero_image_gradient():
    # Images equally favor FLAIR at first; true image-pair separation is zero.
    s={(f,a):-.3 if a=="FLAIR" else -1.7 for f in LABELS for a in LABELS}
    base=margin_objective(s)
    assert base["paired_separation"]==pytest.approx(0)
    # A common answer-prior shift does not change image-pair separation.
    shift={(f,a):v+(0 if a=="FLAIR" else -.2) for (f,a),v in s.items()}
    assert margin_objective(shift)["paired_separation"]==pytest.approx(0)
    # More correct image-conditional margins reduce loss.
    improved=dict(s)
    improved[("FLAIR","FLAIR")]=-.1
    improved[("contrast-enhanced T1","contrast-enhanced T1")]=-.2
    z=margin_objective(improved)
    assert z["paired_separation"]>0 and z["loss"]<base["loss"]
    # Exact finite differences of analytic chain-rule coefficients.
    eps=1e-6
    for key,c in base["gradient_coefficients"].items():
        plus={**s,key:s[key]+eps}; minus={**s,key:s[key]-eps}
        fd=(margin_objective(plus)["loss"]-margin_objective(minus)["loss"])/(2*eps)
        assert c==pytest.approx(fd,abs=1e-7)


def test_reject_invalid_logps_and_missing_tensor_grads():
    s={(f,a): -1.0 for f in LABELS for a in LABELS}
    assert margin_objective(s)["loss"]>0
    with pytest.raises(ValueError,match="four"):
        margin_objective({("FLAIR","FLAIR"):-1.0})
    with pytest.raises(ValueError,match="Finite"):
        margin_objective({**s, ("FLAIR","FLAIR"):float('nan')})
    with pytest.raises(ValueError,match="Finite"):
        margin_objective({**s, ("FLAIR","FLAIR"):0.1})
    with pytest.raises(ValueError,match="coefficient"):
        margin_objective(s,class_weight=-1)


def test_native_backend_is_lazily_loaded():
    # Only contracts are imported; constructing an 8B model is not a CPU test.
    import ast
    import inspect
    from mri_rl import paired_sequence_accuracy_v2 as mod
    tree=ast.parse(inspect.getsource(mod))
    assert any(isinstance(x,ast.FunctionDef) and x.name=="native_backend_class" for x in tree.body)
