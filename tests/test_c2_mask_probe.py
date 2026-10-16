import importlib.util
from pathlib import Path
import hashlib
import pytest
s=importlib.util.spec_from_file_location('c2',Path(__file__).resolve().parents[1]/'scripts/probe_c2_masks.py');m=importlib.util.module_from_spec(s)
if Path(s.origin).exists():s.loader.exec_module(m)


def test_payload_verification_rejects_wrong_size_and_hash():
    assert hasattr(m,'verify_payload')
    b=b'real bytes';item={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    m.verify_payload(b,item)
    with pytest.raises(ValueError):m.verify_payload(b+b'x',item)
    with pytest.raises(ValueError):m.verify_payload(b,dict(item,sha256='0'*64))


def test_candidate_selection_cannot_overlap_reserved_or_existing():
    assert hasattr(m,'verify_candidates')
    selected=[{'subject_PRIVATE':'new','payloads':[{'bytes':10},{'bytes':10},{'bytes':5}]}]
    m.verify_candidates(selected,{'existing'},{'reserved'},1,5)
    with pytest.raises(ValueError):m.verify_candidates(selected,{'new'},set(),1,5)
    with pytest.raises(ValueError):m.verify_candidates(selected,set(),{'new'},1,5)
