"""Independent regression from the actual complete-result export shape; CPU only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from mri_rl import native_group_signal_vjp_v1 as ngs

def test_actual_aggregate_shape_uses_only_allowed_scalar_count_keys():
    kinds = {"private_f": "MRI_FLAIR", "private_t": "MRI_T1", "private_d": "DALLAS"}
    effects = {p: {f: {m: {c: .1 for c in kinds} for m in ngs.METRICS} for f in ngs.FAMILIES} for p in ngs.PLANS}
    taus = {f: {m: {c: 1e-6 for c in kinds} for m in ngs.METRICS} for f in ngs.SCREEN_FAMILIES}
    values = {c: {m: .2 for m in ngs.METRICS} for c in kinds}
    summary = ngs.kind_summary(kinds, effects, taus, values)
    public = ngs.public_aggregate({"kind_summary": summary}, forbidden_values=tuple(kinds))
    assert public["kind_summary"]["MRI_FLAIR"]["count"] == 1
    assert public["kind_summary"]["MRI_T1"]["count"] == 1
    assert public["kind_summary"]["DALLAS"]["count"] == 1

def test_public_context_key_remains_forbidden_even_when_scalar():
    with pytest.raises(ValueError, match="case/group-level"):
        ngs.public_aggregate({"contexts": 4})
