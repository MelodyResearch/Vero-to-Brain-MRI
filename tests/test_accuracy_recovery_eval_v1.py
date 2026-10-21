"""Adam52-compatible Parent/A/B evaluator fixtures (CPU, fake numpy model, synthetic records). No MLX, no MRI data."""
import importlib.util
import json
from pathlib import Path
import pytest
from mri_rl import accuracy_recovery_v1 as ar
from mri_rl import accuracy_recovery_eval_v1 as ae
from mri_rl import paired_z045_continuation_entry_v1 as inherited
from mri_rl import paired_z045_evaluation_v1 as ev
from mri_rl import shared_rollout_gspo_eval_v1 as shared_eval
from mri_rl.answer_likelihood import PREFIX


def load(name):
    spec=importlib.util.spec_from_file_location(name[:-3],Path(__file__).with_name(name));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


d=load('test_accuracy_recovery_driver_v1.py');m=load('test_paired_z045_matched_f32_eval_v1.py')
f,c,c2,launch,world=d.f,d.c,d.c2,d.launch,d.world
reset=d.reset
ROOT=Path(ar.__file__).resolve().parents[2]


def run_and_anchor(tmp_path,accuracy=None):
    if accuracy:f.Fake.accuracy=accuracy
    launch(tmp_path,tmp_path/'run');parent,receipt,_=world(tmp_path);frozen=c.frozen_for(parent,receipt)
    return tmp_path/'run',ar.adam52_parent_anchor(parent,receipt,frozen=frozen),frozen


def test_arm_checkpoints_authenticate_this_experiment_and_the_original_adam52_parent(tmp_path):
    run,anchor,_=run_and_anchor(tmp_path);arms={a:ae.arm_checkpoint(run,a,anchor) for a in 'AB'}
    for a in 'AB':assert arms[a]['groups']==115 and arms[a]['updates']==53 and arms[a]['alias_of_parent'] is False and arms[a]['adapter_sha256']!=anchor['adapter_sha256']
    assert arms['A']['bundle_sha256']==arms['B']['bundle_sha256'] and ae.evaluated_labels(arms)==['parent','A','B']


def test_no_update_arm_aliases_the_parent_and_is_not_scheduled_for_evaluation(tmp_path):
    run,anchor,_=run_and_anchor(tmp_path,{'F':[1,1,1,1],'T':[0,0,0,0]});arms={a:ae.arm_checkpoint(run,a,anchor) for a in 'AB'}
    assert arms['A']['alias_of_parent'] is True and arms['A']['updates']==52 and arms['A']['adapter_sha256']==anchor['adapter_sha256']
    assert ae.evaluated_labels(arms)==['parent','B'] and shared_eval.eval_budget(2)['checkpoints']==2


@pytest.mark.parametrize('case',['incomplete_run','foreign_parent','receipt_marker','gradient_archive','not_committed','wrong_experiment'])
def test_arm_checkpoint_rejects_foreign_or_unbound_terminals(tmp_path,case):
    run,anchor,_=run_and_anchor(tmp_path)
    if case=='incomplete_run':(run/'complete_SAFE.json').unlink()
    elif case=='foreign_parent':anchor={**anchor,'receipt_sha256':c.sha('adam60')}
    elif case=='receipt_marker':
        r=run/'arm_A/boundary_0001/receipt.json';data=json.loads(r.read_text());data['updated']=not data['updated'];r.write_text(json.dumps(data))
    elif case=='gradient_archive':
        meta=run/'gradients_PRIVATE/arm_A_gradient_meta_PRIVATE.json';meta.write_text(meta.read_text()+' ')
    elif case=='not_committed':(run/'arm_A_COMMITTED.json').unlink()
    elif case=='wrong_experiment':
        done=json.loads((run/'complete_SAFE.json').read_text());done['experiment']='SHARED_ROLLOUT_BATCHED_ONE_STEP_GSPO_MECHANISM_CONTROL';(run/'complete_SAFE.json').write_text(json.dumps(done))
    with pytest.raises(ValueError):ae.arm_checkpoint(run,'A',anchor)


# ------------------------------------------------------------------ authorization
def eval_scope(tmp_path,checkpoints=3,**edit):
    run,anchor,frozen=run_and_anchor(tmp_path) if not (tmp_path/'run').exists() else (tmp_path/'run',None,None)
    closure={**c.source_closure(),**{str(ROOT/n):ar.file_sha256(ROOT/n) for n in ar.EVAL_OWN_SOURCES}}
    scope={'scope_kind':ar.EVAL_SCOPE_KIND,'experiment':ar.EXPERIMENT,'execution_subtype':ar.EXECUTION_SUBTYPE,'numerical_policy':ar.POLICY,'recipe_label':ar.RECIPE_LABEL,'implemented_by':'claude_code',
           'budget':shared_eval.eval_budget(checkpoints),'source_sha256':closure,'input_sha256':c.sha('eval input'),
           'parent':{'receipt_sha256':ar.FROZEN_PARENT['receipt_sha256'],'adapter_sha256':ar.FROZEN_PARENT['adapter_sha256']},'output':'/x','evaluated_checkpoints':['parent','A','B']}
    scope.update(edit);return scope


def sign(scope,action=ar.EVAL_ACTION,kind='isolated_codex',instruction='instr'):
    binding=ar.scope_digest(scope)
    return ({'authorized':True,'action':action,'scope_sha256':binding,'human_instruction_sha256':c.sha(instruction)},
            {'verdict':'PASS','scope_sha256':binding,'reviewer_id':'codex','reviewer_kind':kind,'source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256']})


def test_evaluation_authorization_is_fail_closed_separate_from_training_and_consumed_once(tmp_path):
    scope=eval_scope(tmp_path);human,review=sign(scope)
    token=ar.authorize_evaluation(scope,human,review,consumed=ar.consume_once(tmp_path/'eval_ledger'));assert isinstance(token,ar.Authorization)
    with pytest.raises(ValueError,match='already consumed'):ar.authorize_evaluation(scope,human,review,consumed=ar.consume_once(tmp_path/'eval_ledger'))
    # the SAME genuine human instruction hash may bind the training scope: separate action, scope digest and ledger
    ar.consume_once(tmp_path/'train_ledger')(c.sha('instr'))
    for bad_action in (ar.ACTION,'readonly_matched_f32_evaluation'):
        h,r=sign(scope,action=bad_action)
        with pytest.raises(ValueError):ar.authorize_evaluation(scope,h,r,consumed=ar.consume_once(tmp_path/'l2'))
    h,r=sign(scope,kind='claude_code')
    with pytest.raises(ValueError,match='different agent'):ar.authorize_evaluation(scope,h,r,consumed=ar.consume_once(tmp_path/'l3'))
    for edit in ({'budget':{**shared_eval.eval_budget(3),'optimizer_updates':1}},{'scope_kind':ar.SCOPE_KIND},{'approved':True},{'parent':{'receipt_sha256':c.sha('x'),'adapter_sha256':c.sha('y')}}):
        s=eval_scope(tmp_path,**edit);h,r=sign(s)
        with pytest.raises(ValueError):ar.authorize_evaluation(s,h,r,consumed=ar.consume_once(tmp_path/'l4'))
    train_scope=c.scope_fixture();th,tr=c.signed(train_scope)
    with pytest.raises(ValueError):ar.authorize_evaluation(train_scope,th,tr,consumed=ar.consume_once(tmp_path/'l5'))
    es=eval_scope(tmp_path);eh,er=sign(es)
    with pytest.raises(ValueError):ar.authorize_launch(es,eh,er,consumed=ar.consume_once(tmp_path/'l6'))


# ------------------------------------------------------------------ serial runner
class FakeBackend:
    open_count=0;max_open=0;log=[]
    def __init__(self,label):FakeBackend.open_count+=1;FakeBackend.max_open=max(FakeBackend.max_open,FakeBackend.open_count);self.label=label
    def close(self):FakeBackend.open_count-=1


def runner_fixture(tmp_path,arms_alias=(False,False)):
    FakeBackend.open_count=0;FakeBackend.max_open=0;FakeBackend.log=[]
    scope=eval_scope(tmp_path);human,review=sign(scope);token=ar.authorize_evaluation(scope,human,review,consumed=ar.consume_once(tmp_path/'led'))
    prepared={'parent':{'path':'p'},'arms':{a:{'path':a,'alias_of_parent':al} for a,al in zip('AB',arms_alias)}}
    prepared['evaluated_checkpoints']=['parent']+[a for a in 'AB' if not prepared['arms'][a]['alias_of_parent']]
    def releaser(run,resident,directory):
        try:return run()
        finally:resident[0].close()
    def runner(backend,p,label,checkpoint,rows,counter,before):FakeBackend.log.append(label);return {'label':label}
    return prepared,token,dict(backend_factory=lambda label,cp:FakeBackend(label),runner=runner,before=lambda n:None,counter=None,releaser=releaser,load_state=lambda b,l,cp:None),scope


def test_runner_evaluates_parent_then_updated_arms_serially_one_model_at_a_time(tmp_path):
    prepared,token,kw,_=runner_fixture(tmp_path);results=ae.run_checkpoints(prepared,token,out=tmp_path/'eval',**kw)
    assert FakeBackend.log==['parent','A','B'] and FakeBackend.max_open==1 and FakeBackend.open_count==0 and [r['label'] for r in results]==['parent','A','B']


def test_runner_never_re_evaluates_a_no_update_arm_and_rejects_a_mismatched_inventory(tmp_path):
    prepared,token,kw,_=runner_fixture(tmp_path,(True,False));ae.run_checkpoints(prepared,token,out=tmp_path/'eval',**kw);assert FakeBackend.log==['parent','B']
    prepared2,token2,kw2,_=runner_fixture(tmp_path/'x' if (tmp_path/'x').mkdir() is None else tmp_path,(False,False));prepared2['evaluated_checkpoints']=['parent','A']
    with pytest.raises(ValueError,match='authorized scope'):ae.run_checkpoints(prepared2,token2,out=tmp_path/'eval2',**kw2)
    assert FakeBackend.log==[]


def test_runner_rechecks_the_authorized_source_closure_before_every_model(tmp_path):
    src=tmp_path/'extra.py';src.write_text('x=1\n')
    FakeBackend.log=[];scope=eval_scope(tmp_path,source_sha256={**eval_scope(tmp_path)['source_sha256'],str(src):ar.file_sha256(src)});human,review=sign(scope)
    token=ar.authorize_evaluation(scope,human,review,consumed=ar.consume_once(tmp_path/'led'))
    prepared={'parent':{'path':'p'},'arms':{a:{'path':a,'alias_of_parent':False} for a in 'AB'},'evaluated_checkpoints':['parent','A','B']}
    def runner(backend,p,label,checkpoint,rows,counter,before):
        FakeBackend.log.append(label);src.write_text('x=2\n');return {}
    with pytest.raises(ValueError,match='source changed'):
        ae.run_checkpoints(prepared,token,backend_factory=lambda l,cp:FakeBackend(l),runner=runner,before=None,counter=None,releaser=lambda run,resident,directory:run(),load_state=lambda *a:None,out=tmp_path/'eval')
    assert FakeBackend.log==['parent']


# ------------------------------------------------------------------ analyzer + stop rule
def row(task,truth,answer,means):
    return {'id':f'{task}-{truth}-{answer}-{means}','task':task,'truth':truth,'greedy':{'text':f'<think>x</think><answer>\\boxed{{{answer}}}</answer>'},
            'candidates':{'per_label':[{'label':l,'mean':v,'sum':2*v,'first':v} for l,v in zip(ev.LABELS[task],means)]},
            'costs':{'greedy_calls':1,'candidate_score_API_calls':3,'generated_tokens':2,'teacherforced_input_positions':10}}


def rows_for(flair_correct,t1_correct,dallas_correct):
    rows=[]
    for i in range(12):
        ok=i<flair_correct;rows.append(row('mri_sequence','FLAIR','FLAIR' if ok else 'contrast-enhanced T1',(-.1,-.9) if ok else (-.9,-.1)))
    for i in range(12):
        ok=i<t1_correct;rows.append(row('mri_sequence','contrast-enhanced T1','contrast-enhanced T1' if ok else 'FLAIR',(-.9,-.1) if ok else (-.1,-.9)))
    for i in range(16):
        truth='same' if i<8 else 'different';ok=i<dallas_correct if i<8 else (i-8)<dallas_correct
        answer=truth if ok else ('different' if truth=='same' else 'same');rows.append(row('mri_visit_correspondence',truth,answer,(-.1,-.9) if answer=='different' else (-.9,-.1)))
    return rows


def test_endpoint_vector_from_saved_rows_and_exact_inventory():
    v=ae.endpoint_vector(rows_for(12,3,6))
    assert v['FLAIR_recall']==1. and v['T1_recall']==.25 and v['sequence_balanced_accuracy']==.625 and v['Dallas_accuracy']==.75 and v['Dallas_same_recall']==.75
    with pytest.raises(ValueError):ae.endpoint_vector(rows_for(12,3,6)[:-1])


def test_verdict_requires_B_over_A_benefit_and_no_flair_or_dallas_degradation():
    parent=ae.endpoint_vector(rows_for(12,3,6));a=parent
    better=ae.endpoint_vector(rows_for(12,6,6));assert ae.verdict(parent,a,better)['decision']=='CONTINUE_CANDIDATE_REQUIRES_NEW_AUTHORIZATION'
    assert ae.verdict(parent,a,parent)['decision']=='STOP_CANDIDATE'                                          # no B-A benefit
    flair_drop=ae.endpoint_vector(rows_for(10,9,6));out=ae.verdict(parent,a,flair_drop);assert out['decision']=='STOP_CANDIDATE' and out['degrade']['FLAIR_recall_below_parent']
    dallas_drop=ae.endpoint_vector(rows_for(12,6,4));out=ae.verdict(parent,a,dallas_drop);assert out['decision']=='STOP_CANDIDATE' and out['degrade']['Dallas_accuracy_below_parent']
    assert out['training_authorized'] is False and out['generalization_proven'] is False


def test_analyze_uses_parent_rows_for_an_alias_arm_and_never_needs_its_own_rows(tmp_path):
    root=tmp_path/'eval'
    for label,args in (('parent',(12,3,6)),('B',(12,6,6))):
        directory=root/f'{label}_PRIVATE/rows_PRIVATE';directory.mkdir(parents=True)
        for i,r in enumerate(rows_for(*args)):(directory/f'{i}_PRIVATE.json').write_text(json.dumps(r))
        (directory/'restored_state_PRIVATE.json').write_text('{}')
    arms={'A':{'alias_of_parent':True},'B':{'alias_of_parent':False}}
    analysis=ae.analyze(root,arms)
    assert analysis['aliases_of_parent']==['A'] and analysis['vectors']['A']==analysis['vectors']['parent'] and analysis['verdict']['decision']=='CONTINUE_CANDIDATE_REQUIRES_NEW_AUTHORIZATION'


# ------------------------------------------------------------------ CPU preparation
ENCODE_TABLE={PREFIX:[9,8],'FLAIR':[1,2],'contrast-enhanced T1':[3,4,5,6,7],'different':[11,12],'same':[13]}


def encode(text):
    if text in ENCODE_TABLE:return ENCODE_TABLE[text]
    for label in ('FLAIR','contrast-enhanced T1','different','same'):
        if text==PREFIX+label:return ENCODE_TABLE[PREFIX]+ENCODE_TABLE[label]
    raise ValueError(text)


def prep_world(tmp_path,monkeypatch):
    run,anchor,frozen=run_and_anchor(tmp_path);R,W=c2.make_repo(tmp_path/'repo');(R/'local_data/accuracy_recovery_v1').mkdir(parents=True)
    parent,receipt,_=world(tmp_path);(tmp_path/'cache').mkdir();cert=tmp_path/'cert.json';cert.write_text('{}')
    (tmp_path/'runtime.json').write_text(json.dumps({'parent_checkpoint':str(parent),'parent_receipt_sha256':receipt,'cache_dir':str(tmp_path/'cache'),'cache_certificate':str(cert),
        'cache_certificate_sha256':inherited.digest(cert),'model':str(tmp_path),'contract':'c','global_lock':str(tmp_path/'lock'),'runtime_source_sha256':{'/x.py':'0'*64}}))
    (tmp_path/'evaluation.json').write_text(json.dumps(m.frozen()))
    orig=ar.adam52_parent_anchor
    monkeypatch.setattr(ar,'adam52_parent_anchor',lambda path,sha:orig(path,sha,frozen=frozen))
    monkeypatch.setattr(inherited,'source_closure',lambda runtime:{'/installed/runtime.py':'1'*64})
    monkeypatch.setattr(inherited,'cache_metadata',lambda runtime:{'expected_binding':{'cache':'fixture'}})
    spec={'runtime_spec':str(tmp_path/'runtime.json'),'evaluation':str(tmp_path/'evaluation.json'),'training_run':str(run),'output':str(R/'local_data/accuracy_recovery_v1/eval1')}
    return spec,W,R


def test_prepare_binds_checkpoints_records_tokens_sources_and_emits_a_scope_the_authorizer_understands(tmp_path,monkeypatch):
    spec,W,R=prep_world(tmp_path,monkeypatch);prepared=ae.prepare(spec,encode=encode,start=W)
    assert prepared['blockers']==[] and prepared['evaluated_checkpoints']==['parent','A','B'] and prepared['budget']==shared_eval.eval_budget(3) and prepared['output']==str(R/'local_data/accuracy_recovery_v1/eval1')
    assert len(prepared['records_PRIVATE'])==40 and len(prepared['aliases_PRIVATE'])==8 and prepared['model_PASS'] is False
    scope=ae.launch_scope(prepared);assert scope['scope_kind']==ar.EVAL_SCOPE_KIND and 'records_PRIVATE' not in scope and scope['evaluated_checkpoints']==['parent','A','B']
    assert ae.prepare(spec,encode=encode,start=W)==prepared                          # deterministic: the authorized scope can be re-derived at launch


@pytest.mark.parametrize('case',['worktree_output','elsewhere_output','forbidden_images','forbidden_spec'])
def test_prepare_refuses_non_canonical_outputs_and_forbidden_evaluation_inputs(tmp_path,monkeypatch,case):
    spec,W,R=prep_world(tmp_path,monkeypatch)
    if case=='worktree_output':spec['output']=str(W/'local_data/accuracy_recovery_v1/eval1')
    elif case=='elsewhere_output':spec['output']=str(tmp_path/'eval1')
    elif case=='forbidden_images':
        data=json.loads((tmp_path/'evaluation.json').read_text())
        for r in data['records_PRIVATE']:r['row_PRIVATE']['images']=[r['row_PRIVATE']['images'][0].replace('/opaque/','/opaque/new16/')]
        (tmp_path/'evaluation.json').write_text(json.dumps(data))
    elif case=='forbidden_spec':spec['evaluation']=str(tmp_path/'final100_records.json')
    with pytest.raises(ValueError):ae.prepare(spec,encode=encode,start=W)


def test_prepare_reports_pending_inputs_without_fabricating_and_run_needs_artifacts(tmp_path):
    prepared=ae.prepare({'runtime_spec':str(tmp_path/'none.json'),'evaluation':str(tmp_path/'e.json'),'training_run':str(tmp_path/'r')})
    assert prepared['blockers']==['RUNTIME_SPEC_NOT_PRESENT'] and prepared['model_PASS'] is False
    with pytest.raises(ValueError,match='pending'):ae.launch_scope(prepared)
    from types import SimpleNamespace as NS
    with pytest.raises(ValueError,match='NOT_AUTHORIZED'):ae.execute(NS(spec=tmp_path/'s.json'))
    import sys;assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)


def test_swap_alias_image_dependence_is_computed_from_reused_real_outputs_and_gates_the_verdict():
    plan=ae.matched.compile_records(m.frozen())
    rows=[{'id':r['id'],'task':r['row_PRIVATE']['ability'],'truth':r['row_PRIVATE']['reward_model']['ground_truth'],
           'greedy':{'text':'<think>x</think><answer>\\boxed{'+r['row_PRIVATE']['reward_model']['ground_truth']+'}</answer>'}} for r in plan['real']]
    summary=ae.swap_summaries(rows,plan['aliases']);assert summary['comparisons']==8 and summary['valid_opposite_class_changes']==8
    parent=ae.endpoint_vector(rows_for(12,3,6));better=ae.endpoint_vector(rows_for(12,6,6))
    swaps={'A':{'valid_opposite_class_changes':8},'B':{'valid_opposite_class_changes':3}}
    out=ae.verdict(parent,parent,better,swaps);assert out['decision']=='STOP_CANDIDATE' and out['degrade']['swap_valid_opposite_class_changes_below_A']
