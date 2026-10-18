"""Independent counter-offset inactive-path test; synthetic CPU backend only."""
import importlib.util,json
from pathlib import Path

def test_actual_middle_skip_preserves_absolute_step_and_split_resume(tmp_path):
 p=Path(__file__).with_name('test_adaptive_accuracy_gate_v1.py');s=importlib.util.spec_from_file_location('adaptive_fixture_independent',p);f=importlib.util.module_from_spec(s);s.loader.exec_module(f)
 m,B,parent,sha,rows=f.fixtures(tmp_path)
 class Mixed(B):
  def rollout(self,row,index,work):
   value=super().rollout(row,index,work)
   if index==115:value['responses'][0]['text']='wrong too'
   return value
 full=f.invoke(m,Mixed,parent,sha,rows,tmp_path/'full',3)
 f.invoke(m,Mixed,parent,sha,rows,tmp_path/'split',1)
 resumed=f.invoke(m,Mixed,parent,sha,rows,tmp_path/'split',2,True)
 assert full['completed_groups']==resumed['completed_groups']==3
 assert full['optimizer_updates']==resumed['optimizer_updates']==2
 assert full['absolute_optimizer_updates']==resumed['absolute_optimizer_updates']==54
 assert full['absolute_completed_groups']==resumed['absolute_completed_groups']==117
 for run in ('full','split'):
  skipped=json.loads((tmp_path/run/'boundary_0002/receipt.json').read_text())
  assert skipped['updated'] is False and skipped['optimizer_updates']==1 and skipped['absolute_optimizer_updates']==53
  assert skipped['optimizer_before_fingerprint']==skipped['optimizer_after_fingerprint']
  final=json.loads((tmp_path/run/'boundary_0003/optimizer/manifest.json').read_text())
  assert final['optimizer_updates']==54 and final['completed_groups']==117
 assert (tmp_path/'full/boundary_0003/adapter.safetensors').read_bytes()==(tmp_path/'split/boundary_0003/adapter.safetensors').read_bytes()
