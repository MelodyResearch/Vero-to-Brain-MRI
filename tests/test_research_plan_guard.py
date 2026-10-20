import hashlib
import json
from mri_rl.research_plan_guard import PLAN, LOCK, LOG, check_trees


def tree(plan=b'# frozen plan\n',log=b'# log\ninitial\n'):
    return {PLAN:plan, LOG:log, LOCK:json.dumps({'schema':'research-plan-lock-v1','plan_sha256':hashlib.sha256(plan).hexdigest(),
        'initial_researcher_instruction_sha256':'a'*64}).encode()}


def test_first_creation_and_ordinary_append():
    t=tree();assert check_trees({},t)['status']=='INITIAL_FREEZE'
    changed=dict(t);changed[LOG]+=b'new result\n'
    assert check_trees(t,changed)['status']=='UNCHANGED_PLAN_APPEND_ONLY_LOG'


def test_change_plan_and_lock_and_fake_approval_never_authorizes():
    base=tree();candidate=tree(b'# new plan\n')
    candidate['approved.json']=b'{"approved":true}'
    r=check_trees(base,candidate)
    assert r['status']=='BLOCKED' and 'PLAN_CHANGED' in r['errors']
    assert r['human_approval']=='NOT_MACHINE_VERIFIED'
    assert '-# frozen plan' in r['plan_diff']


def test_log_rewrite_truncate_or_delete_blocked():
    base=tree()
    for bad in (b'# log\nchanged\n',b'',None):
        t=dict(base)
        if bad is None:t.pop(LOG)
        else:t[LOG]=bad
        assert 'LOG_NOT_APPEND_ONLY' in check_trees(base,t)['errors']


def test_plan_or_lock_delete_and_hash_mismatch_blocked():
    base=tree()
    for key in (PLAN,LOCK):
        t=dict(base);t.pop(key)
        assert check_trees(base,t)['status']=='BLOCKED'
    t=tree();t[PLAN]+=b'changed'
    assert 'LOCK_DIGEST_MISMATCH' in check_trees({},t)['errors']


def test_lock_rewrite_even_without_plan_change_is_not_ordinary():
    base=tree();t=dict(base);t[LOCK]+=b' '
    assert 'LOCK_CHANGED' in check_trees(base,t)['errors']


def test_baseline_cannot_rebootstrap_when_plan_deleted():
    base=tree();base.pop(PLAN)
    assert 'INVALID_BASELINE' in check_trees(base,tree())['errors']


def test_existing_incomplete_freeze_is_not_a_valid_git_anchor():
    for missing in (LOCK,LOG):
        base=tree();base.pop(missing)
        assert 'INVALID_BASELINE' in check_trees(base,tree())['errors']


def test_new_branch_checks_earlier_commits_not_just_last(tmp_path):
    import subprocess
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/check_research_plan_guard.py'
    def git(*args):
        return subprocess.run(['git','-c','core.hooksPath=/dev/null',*args],cwd=tmp_path,capture_output=True,check=True)
    git('init','-q');git('config','user.email','fixture@example.invalid');git('config','user.name','Fixture')
    def commit(files,message):
        for name,data in files.items():
            path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        git('add','.');git('-c','commit.gpgsign=false','commit','-qm',message)
    commit(tree(),'explicit initial freeze')
    commit(tree(b'# unauthorized new plan\n'),'earlier hidden plan edit plus matching lock')
    commit({'ordinary.py':b'pass\n'},'ordinary later code')
    result=subprocess.run(['python3',str(script),'--history','--head','HEAD'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2 and 'PLAN_CHANGED' in result.stdout


def test_initial_staged_freeze_on_a_repository_without_prior_plan(tmp_path):
    import subprocess
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/check_research_plan_guard.py'
    def git(*args):return subprocess.run(['git','-c','core.hooksPath=/dev/null',*args],cwd=tmp_path,capture_output=True,check=True)
    git('init','-q');git('config','user.email','fixture@example.invalid');git('config','user.name','Fixture')
    (tmp_path/'README').write_text('existing repository\n');git('add','.');git('-c','commit.gpgsign=false','commit','-qm','existing repository')
    for name,data in tree().items():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    git('add','.')
    result=subprocess.run(['python3',str(script),'--base','HEAD','--staged'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0 and 'INITIAL_FREEZE' in result.stdout
