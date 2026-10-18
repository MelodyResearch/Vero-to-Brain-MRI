import importlib.util,json,hashlib
from pathlib import Path
import pytest
W=Path(__file__).parents[1]
def module():
 p=W/"scripts/analyze_original22_saved_final_v1.py";assert p.exists(),"Prospective saved-analysis wrapper missing";s=importlib.util.spec_from_file_location("original22_saved_test",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def fixture(tmp_path):
 runs=[];completed=[];reviews={}
 for i in range(22):
  d=tmp_path/str(i);d.mkdir();out=d/"outputs";out.write_text("not labels or predictions");log=d/"log";log.write_text("closed");exit=d/"exit";key=f"c{i}";digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();e={"checkpoint_id":key,"exit_code":0,"review_sha256":"r"+key,"scope_sha256":"s"+key,"raw_log_sha256":digest(log),"output_sha256":digest(out),"records":100};exit.write_text(json.dumps(e));runs.append({"checkpoint_id":key,"scope":{"sha256":"s"+key},"exit":str(exit),"output":str(out),"log":str(log)});completed.append({"checkpoint_id":key,"exit":{"path":str(exit),"sha256":digest(exit)},"output":{"path":str(out),"sha256":digest(out)},"log":{"path":str(log),"sha256":digest(log)}});reviews[key]={"sha256":"r"+key}
 return runs,completed,reviews
def test_all22_exit_gate_precedes_any_output_read(tmp_path):
 m=module();runs,c,r=fixture(tmp_path);last=Path(c[-1]["exit"]["path"]);e=json.loads(last.read_text());e["exit_code"]=1;last.write_text(json.dumps(e));c[-1]["exit"]["sha256"]=m.sha(last)
 for row in c:Path(row["output"]["path"]).unlink()
 with pytest.raises(ValueError,match="exit0"):m.verify_all_completed(runs,c,r)
def test_exact22_complete_external_hashes_and_scope_binding(tmp_path):
 m=module();runs,c,r=fixture(tmp_path);assert len(m.verify_all_completed(runs,c,r))==22
 with pytest.raises(ValueError,match="22"):m.verify_all_completed(runs,c[:-1],r)
 c[0]["output"]["sha256"]=None
 with pytest.raises(ValueError):m.verify_all_completed(runs,c,r)
def test_completed_scope_mismatch_rejected_before_outputs(tmp_path):
 m=module();runs,c,r=fixture(tmp_path);e=Path(c[0]["exit"]["path"]);x=json.loads(e.read_text());x["scope_sha256"]="foreign";e.write_text(json.dumps(x));c[0]["exit"]["sha256"]=m.sha(e)
 with pytest.raises(ValueError,match="scope"):m.verify_all_completed(runs,c,r)
def test_matrix_reuses_one_B28_without_extra_model_runs():
 m=module();keys=[("B28",s) for s in (7,11,13)]+[(a,s) for a in ("S_sequence","S_correspondence","P_raw","U_raw","P_calibrated","U_calibrated","same_C_SFT") for s in (7,11,13)];ids=[m.checkpoint_key(a,s) for a,s in keys];assert len(ids)==24 and len(set(ids))==22 and ids.count("B28")==3


def test_not_run_template_blocks_before_any_reference_or_output_io():
 m=module()
 with pytest.raises(ValueError,match="NOT_RUN"):
  m.analyze({"schema":"original22-final-saved-analysis-input-v1","status":"NOT_RUN_FUTURE_COMPLETION_HASHES_REQUIRE_INDEPENDENT_COORDINATOR","primary_root":"/must-not-read"})
