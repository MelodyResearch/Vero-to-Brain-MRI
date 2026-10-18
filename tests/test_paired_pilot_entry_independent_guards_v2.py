"""Independent CPU-only counterexamples for new entry resume/finalization guards."""
import importlib.util,json,sys,types
from pathlib import Path
import pytest

def entry():
 p=Path(__file__).parents[1]/'scripts/run_paired_discrimination_pilot_v2.py'
 s=importlib.util.spec_from_file_location('paired_entry_independent',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

@pytest.mark.parametrize('changed', [{'image_sha256':'changed'}, {'candidate_scores':{'FLAIR':{'mean_label_logp':float('nan')}}}])
def test_existing_prefix_rejects_unverified_or_nonfinite_payload(tmp_path,changed):
 m=entry();path=tmp_path/'rows.jsonl'
 expected={'record_id':'x','image_sha256':'frozen','prompt':[{'role':'user','content':'fixed'}]}
 path.write_text(json.dumps({**expected,'run_identity':'same','weight_updates':0,'response':'ok',**changed})+'\n')
 # An output carrying only these fields is neither an authenticated native record
 # nor a complete finite score. It must not be silently adopted on resume.
 with pytest.raises((ValueError,KeyError,AssertionError)):
  m.durable_prefix(path,'same',[expected],lambda r:None)

def test_full_lengths_do_not_authorize_invalid_completion_receipt(tmp_path,monkeypatch):
 m=entry();out=tmp_path/'out';out.mkdir();R=tmp_path/'primary';(R/'results').mkdir(parents=True)
 (R/'results/EXACT_QUANT_CACHE_CERTIFICATE.json').write_text(json.dumps({'non_weight_files_identical':[]}));model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
 records=[{'record_id':f'g{i}'} for i in range(40)];candidates=[{'record_id':f'c{i}'} for i in range(25)]
 for path,rs in ((out/'greedy_PRIVATE.jsonl',records),(out/'candidates_PRIVATE.jsonl',candidates)):
  path.write_text(''.join(json.dumps({**r,'run_identity':'same','weight_updates':0,'response':'ok'})+'\n' for r in rs))
 (out/'eval_complete_SAFE.json').write_text(json.dumps({'native_release_verified':False,'scope_sha256':'different','greedy_sha256':'wrong','candidate_sha256':'wrong'}))
 # Mock pure boundaries; no MLX/model imports, no real original queue lock.
 gate=types.SimpleNamespace(before_unit=lambda *a,**k:None)
 modules={
 'mri_rl.paced_resources':types.SimpleNamespace(PacedGate=lambda *a,**k:gate),
 'mri_rl.mlx_continuous_gspo':types.SimpleNamespace(NativeGSPOBackend=lambda *a,**k:pytest.fail('No native backend permitted'),is_loading_unit=lambda *a:False),
 'mri_rl.experiment_contract':types.SimpleNamespace(load_contract=lambda *a:None),
 'mri_rl.likelihood_cache':types.SimpleNamespace(verify_cache_identity=lambda *a: {'cache_dir':'cache','certificate_sha256':'cert'})}
 for name,value in modules.items():monkeypatch.setitem(sys.modules,name,value)
 manifest=tmp_path/'manifest';(tmp_path/'remaining20.lock').touch()
 a=types.SimpleNamespace(review=None,expected_review_sha256=None,primary_root=R,output_directory=out,mode='eval',checkpoint='B28')
 original={'derived_cache':{'cache_dir':'cache','certificate_sha256':'cert'},'prompt_contract_path':'contract','model_path':str(model)}
 scope={'ready':True,'run_identity':'same','adapter':{'path':'adapter','sha256':'adapterSHA'},'checkpoint':'B28','evaluation_contract_sha256':'contractSHA','starting_outputs_sha256':{name:m.sha(out/name) for name in ('greedy_PRIVATE.jsonl','candidates_PRIVATE.jsonl','eval_complete_SAFE.json')}}
 evaluation={'greedy_records_PRIVATE':records,'candidate_records_PRIVATE':candidates}
 with pytest.raises((ValueError,KeyError,AssertionError)):
  m.execute(a,None,types.SimpleNamespace(verify_review=lambda *a:None),types.SimpleNamespace(MANIFEST=manifest,validate_numeric_route=lambda row:None),original,None,evaluation,scope)


def test_completion_validation_accepts_exact_evidence_and_rejects_each_tampered_binding(tmp_path):
 m=entry();g=tmp_path/'g';c=tmp_path/'c';g.write_text('g');c.write_text('c')
 scope={'checkpoint':'post2','run_identity':'run','adapter':{'sha256':'adapterSHA'},'evaluation_contract_sha256':'contractSHA'}
 receipt={'status':'ACTUAL_FIXED40_GREEDY_AND25_SEQUENCE_CANDIDATES_COMPLETE','checkpoint':'post2','run_identity':'run','adapter_sha256':'adapterSHA','native_release_verified':True,'weight_updates':0,'greedy_records':40,'candidate_records':25,'greedy_sha256':m.sha(g),'candidate_sha256':m.sha(c),'evaluation_contract_sha256':'contractSHA'}
 m.validate_completion(receipt,scope,{},g,c)
 for key,value in [('greedy_sha256','changed'),('candidate_sha256','changed'),('adapter_sha256','changed'),('run_identity','changed'),('native_release_verified',False),('greedy_records',39),('candidate_records',24),('evaluation_contract_sha256','changed')]:
  with pytest.raises(ValueError):m.validate_completion({**receipt,key:value},scope,{},g,c)
