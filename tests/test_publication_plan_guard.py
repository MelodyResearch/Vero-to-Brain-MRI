"""Publication imports preserve source bytes without treating a summary log as source history."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
def module():
 s=importlib.util.spec_from_file_location('publication_guard',ROOT/'scripts/check_research_plan_guard.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_actual_preserved_plan_import_has_valid_scientific_provenance():
 m=module();report=m.history_report('HEAD')
 assert report['errors']==[]

def test_after_freeze_mutation_remains_blocked(monkeypatch):
 m=module();anchor=m.snapshot('a71bb39a5b75d4cb65b7a958fa79548ff0805254');bad=dict(anchor);bad[m.PATHS[0]]+=b'changed'
 snapshots={'before':{m.PATHS[2]:b'publication log'},'imported':anchor,'changed':bad,'fixture':bad,'a71bb39a5b75d4cb65b7a958fa79548ff0805254':anchor}
 monkeypatch.setattr(m,'snapshot',lambda ref:snapshots[ref]);monkeypatch.setattr(m,'git',lambda *a,**kw:SimpleNamespace(stdout=b'before\nimported\nchanged\n'))
 assert 'PLAN_CHANGED' in m.history_report('fixture')['errors']

def test_wrong_initial_import_is_not_a_new_approval(monkeypatch):
 m=module();anchor=m.snapshot('a71bb39a5b75d4cb65b7a958fa79548ff0805254');wrong=dict(anchor);wrong[m.PATHS[0]]+=b'changed'
 snapshots={'before':{m.PATHS[2]:b'publication log'},'imported':wrong,'fixture':wrong,'a71bb39a5b75d4cb65b7a958fa79548ff0805254':anchor}
 monkeypatch.setattr(m,'snapshot',lambda ref:snapshots[ref]);monkeypatch.setattr(m,'git',lambda *a,**kw:SimpleNamespace(stdout=b'before\nimported\n'))
 assert m.history_report('fixture')['errors']
