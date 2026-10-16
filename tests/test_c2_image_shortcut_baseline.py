import importlib.util
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

path=Path(__file__).resolve().parents[1]/'scripts/c2_image_shortcut_baseline.py'
spec=importlib.util.spec_from_file_location('c2shortcut',path)
m=importlib.util.module_from_spec(spec)
if path.exists(): spec.loader.exec_module(m)


def test_neighbor_evidence_excludes_patient_and_preserves_id_ties():
    assert hasattr(m,'nearest_neighbors')
    rows=[{'patient':'a','record_id':'a','label':'yes'},
          {'patient':'a','record_id':'a2','label':'yes'},
          {'patient':'b','record_id':'b','label':'no'},
          {'patient':'c','record_id':'c','label':'yes'}]
    features=np.array([[0.],[0.],[2.],[2.]])
    neighbors=m.nearest_neighbors(rows,features)
    assert neighbors[:2]==[2,2]
    assert all(rows[i]['patient']!=rows[j]['patient'] for i,j in enumerate(neighbors))
    assert m.c1.lopo_predictions(rows,features)==[rows[j]['label'] for j in neighbors]


def test_png_feature_contract_and_changed_bytes_rejected(tmp_path):
    assert hasattr(m,'extract_features')
    path=tmp_path/'pixels.png'
    pixels=np.arange(256,dtype=np.uint8).reshape(16,16)
    Image.fromarray(pixels).save(path)
    expected=m.c1.sha(path)
    thumbnail,histogram=m.extract_features(path,expected)
    assert np.array_equal(thumbnail,pixels.ravel()/255)
    assert np.array_equal(histogram,np.ones(16)/16)
    Image.fromarray(np.zeros((16,16),dtype=np.uint8)).save(path)
    with pytest.raises(ValueError,match='Image hash'):
        m.extract_features(path,expected)


def test_frozen_cohort_rejects_duplicate_patient_or_record():
    assert hasattr(m,'validate_rows')
    rows=[]
    for category,n in m.EXPECTED_COUNTS.items():
        for i in range(n):
            rows.append({'category':category,'record_id':category+str(i),
                         'patient':str(i),'label':'yes' if i%2 else 'no'})
    m.validate_rows(rows)
    rows[1]['patient']=rows[0]['patient']
    with pytest.raises(ValueError,match='distinct patients'):
        m.validate_rows(rows)
