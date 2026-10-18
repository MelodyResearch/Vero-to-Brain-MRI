import importlib.util
from pathlib import Path
W=Path(__file__).parents[1]
def mod():
 p=W/'scripts/diagnose_saved_generation_format_v1.py';assert p.exists(),'Saved-generation diagnosis missing';s=importlib.util.spec_from_file_location('format_diagnosis_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class Chars:
 def decode(self,ids,skip_special_tokens=True):return ''.join(chr(x) for x in ids if x not in {151645,151643})
def row(text,finish='stop',eos=True):
 ids=[ord(x) for x in text]+([151645] if eos else []);return {'response':text,'token_ids':ids,'finish_reason':finish,'generation_tokens':len(ids),'hit_max_tokens':False}
def test_length_cut_thinking_is_not_eos_configuration_failure():
 m=mod();r=row('<think>long unfinished reasoning',finish='length',eos=False);x=m.inspect_case(r,('FLAIR','contrast-enhanced T1'),'FLAIR',Chars(),{151645,151643},len(r['token_ids']))
 assert x['confirmed_length_termination'] and not x['terminal_eos_present'] and not x['strict_correct'] and x['tag_token_bounds']['think_close_complete_at_tokens'] is None
 assert x['boxed_anywhere_salvage_label'] is None

def test_eos_at_exact_cap_is_stop_not_cap_truncation():
 m=mod();r=row('<think>x</think><answer>\\boxed{FLAIR}</answer>');r['hit_max_tokens']=True;x=m.inspect_case(r,('FLAIR','contrast-enhanced T1'),'FLAIR',Chars(),{151645,151643},len(r['token_ids']))
 assert x['strict_correct'] and x['terminal_eos_present'] and not x['confirmed_length_termination'] and x['token_text_exact_match'] and x['tag_token_bounds']['think_close_complete_at_tokens']==16

def test_salvage_does_not_modify_strict_format_metric():
 m=mod();r=row('<think>x</think><answer>FLAIR</answer>');x=m.inspect_case(r,('FLAIR','contrast-enhanced T1'),'FLAIR',Chars(),{151645,151643},256)
 assert not x['strict_correct'] and x['reward_format']==.5 and x['answer_span_salvage_correct'] and x['boxed_anywhere_salvage_label'] is None
