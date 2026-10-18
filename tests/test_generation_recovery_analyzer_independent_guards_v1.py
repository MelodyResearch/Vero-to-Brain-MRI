"""Independent saved-analyzer guards; no native calls."""
import importlib.util
from pathlib import Path
W=Path(__file__).parents[1]
def module():
 p=W/'scripts/analyze_generation_recovery_v1.py';s=importlib.util.spec_from_file_location('gen_analysis_independent',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_eos_stop_exactly_at256_is_completed_not_truncated():
 m=module();base=[{'token_ids':[1,2],'response':'finished','hit_max_tokens':True,'finish_reason':'stop'}];control=[{'token_ids':[1,2],'response':'finished','finish_reason':'stop'}];r=m.cap_prefix_agreement(base,control,[0],[0]);assert r['baseline_capped_records']==0 and r['completed_baseline_records']==1
