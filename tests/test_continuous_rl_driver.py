"""Real driver with NumPy Adam and RNG: synthetic fixtures, no model evidence."""
import importlib.util
import json
import random
from pathlib import Path
import numpy as np
import pytest


def module():
    path = Path(__file__).parents[1] / 'src/mri_rl/continuous_rl.py'
    assert path.exists(), 'Continuous RL driver is not implemented'
    spec = importlib.util.spec_from_file_location('continuous_rl_test', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class NumericBackend:
    """Non-collinear Adam updates with genuine serialized state and RNG."""
    instances = []
    def __init__(self, adapter, config):
        self.weight = np.array([1., 2.]); self.rng = random.Random(117)
        self.state = {'step': np.array(0, dtype=np.uint64), 'learning_rate': np.array(1e-5)}
        self.calls = []; self.closed = False
        if adapter is not None:
            with np.load(adapter, allow_pickle=False) as data: self.weight = data['weight'].copy()
        NumericBackend.instances.append(self)
    def optimizer_state(self): return self.state
    def rng_state(self): return {'python': self.rng.getstate()}
    def restore(self, state, rng): self.state = state; self.rng.setstate(rng['python'])
    def trainable_signature(self): return {'weight': {'shape': [2], 'dtype': 'float64'}}
    def save_arrays(self, path, arrays):
        with Path(path).open('wb') as f: np.savez(f, **arrays)
    def load_arrays(self, path):
        with np.load(path, allow_pickle=False) as data: return {k:data[k].copy() for k in data.files}
    def save_adapter(self, path): self.save_arrays(path, {'weight': self.weight})
    def validate_state(self):
        arrays=[self.state['step'],self.state['learning_rate']]
        moments=self.state.get('weight',{})
        arrays.extend(moments.values())
        return {'parameters_finite':bool(np.isfinite(self.weight).all()),
                'optimizer_finite':all(bool(np.isfinite(x).all()) for x in arrays),
                'second_moments_nonnegative':bool(np.all(moments.get('v',np.array([0.]))>=0))}
    def rollout(self, row, index, directory):
        draw = self.rng.random(); self.calls.append(('rollout', index, draw))
        labels = ['FLAIR', 'contrast-enhanced T1'] if index != 1 else ['FLAIR', 'FLAIR']
        import hashlib
        return {'record_id': row['id'], 'image_sha256':hashlib.sha256(Path(row['images'][0]).read_bytes()).hexdigest(), 'responses': [{'text': '<think>Fixture.</think><answer>\\boxed{'+label+'}</answer>', 'token_ids':[i+1], 'finish_reason':'stop'} for i,label in enumerate(labels)]}
    def update(self, row, group, advantages):
        self.calls.append(('update', row['id']))
        g = np.array([1., -2.]) if int(self.state['step'])==0 else np.array([3., 1.])
        moments=self.state.setdefault('weight', {'m':np.zeros(2),'v':np.zeros(2)})
        moments['m']=.9*moments['m']+.1*g; moments['v']=.999*moments['v']+.001*g*g
        self.state['step']+=1
        self.weight-=self.state['learning_rate']*moments['m']/(np.sqrt(moments['v'])+1e-8)
        return {'loss_terms':[.2,.3], 'gradient_l2':float(np.linalg.norm(g)), 'old_current_logp_max_abs_errors':[0.,0.]}
    def close(self): self.closed=True


@pytest.fixture
def setup(tmp_path):
    NumericBackend.instances=[]
    image=tmp_path/'image.png';image.write_bytes(b'SYNTHETIC NOT A REAL MRI')
    rows=[{'id':f'synthetic-{i}','ability':'mri_sequence','data_source':'OpenNeuro_ds007045', 'images':[str(image)],'reward_model':{'ground_truth':'FLAIR'},'extra_info':{'reward_type':'mri_sequence','internal_split':'train','patient_lineage':'BIDS_SUBJECT_VERIFIED','patient_key':f'fixture-{i}','question':'Fixture only'}} for i in range(3)]
    return rows, {'model_sha256':'fixture','contract_id':'fixture','source_revision':'fixture'},tmp_path


def run(m, setup, directory, count, resume=False, authorize=None, factory=NumericBackend):
    rows,bindings,_=setup
    return m.run(output=directory, schedule=rows, bindings=bindings, backend_factory=factory,
                 authorize=authorize or (lambda context: None), max_new_groups=count, resume=resume)


def test_driver_continuous_and_restored_match_including_inactive_rng(setup):
    m=module();_,_,tmp=setup
    full=run(m,setup,tmp/'full',3);one=NumericBackend.instances[-1]
    assert len(NumericBackend.instances)==1 and one.closed
    run(m,setup,tmp/'split',1)
    split=run(m,setup,tmp/'split',2,resume=True);last=NumericBackend.instances[-1]
    np.testing.assert_array_equal(one.weight,last.weight)
    for key in ['m','v']:np.testing.assert_array_equal(one.state['weight'][key],last.state['weight'][key])
    assert one.rng.random()==last.rng.random()
    assert full['completed_groups']==split['completed_groups']==3
    assert full['optimizer_updates']==split['optimizer_updates']==2
    assert last.calls[0][1]==1 and [x for x in last.calls if x[0]=='update']==[('update','synthetic-2')]


def test_inactive_skip_keeps_moments_weights_and_step(setup):
    m=module();_,_,tmp=setup
    run(m,setup,tmp/'run',1);first=NumericBackend.instances[-1]
    before=first.weight.copy();moments={k:v.copy() for k,v in first.state['weight'].items()}
    result=run(m,setup,tmp/'run',1,resume=True);second=NumericBackend.instances[-1]
    np.testing.assert_array_equal(second.weight,before)
    for k in moments:np.testing.assert_array_equal(second.state['weight'][k],moments[k])
    assert int(second.state['step'])==1 and result['completed_groups']==2
    assert not any(x[0]=='update' for x in second.calls)


def test_completed_resume_does_not_repeat_update_or_load_backend(setup):
    m=module();_,_,tmp=setup
    run(m,setup,tmp/'run',3);n=len(NumericBackend.instances)
    result=run(m,setup,tmp/'run',1,resume=True)
    assert result['completed_groups']==3 and len(NumericBackend.instances)==n


def test_authorization_and_corruption_reject_before_backend(setup):
    m=module();_,_,tmp=setup
    def deny(context):raise ValueError('review denied')
    with pytest.raises(ValueError,match='review denied'):run(m,setup,tmp/'deny',1,authorize=deny)
    assert NumericBackend.instances==[]
    run(m,setup,tmp/'run',1);n=len(NumericBackend.instances)
    (tmp/'run/boundary_0001/adapter.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='digest'):run(m,setup,tmp/'run',1,resume=True)
    assert len(NumericBackend.instances)==n


def test_crash_before_commit_does_not_publish_partial_update(setup):
    m=module();_,_,tmp=setup
    class Crash(NumericBackend):
        def save_adapter(self,path):
            if int(self.state['step']):raise RuntimeError('simulated interrupted save')
            return super().save_adapter(path)
    with pytest.raises(RuntimeError,match='interrupted save'):run(m,setup,tmp/'run',1,factory=Crash)
    assert (tmp/'run/boundary_0000/receipt.json').exists()
    assert not (tmp/'run/boundary_0001').exists()
    run(m,setup,tmp/'run',3,resume=True)
    assert int(NumericBackend.instances[-1].state['step'])==2
    assert list((tmp/'run').glob('.pending-*')), 'Keep interrupted evidence'


def test_schedule_tamper_rejected_before_restore(setup):
    m=module();rows,bindings,tmp=setup
    run(m,setup,tmp/'run',1);n=len(NumericBackend.instances)
    rows[2]['reward_model']['ground_truth']='contrast-enhanced T1'
    with pytest.raises(ValueError,match='binding'):run(m,setup,tmp/'run',1,resume=True)
    assert len(NumericBackend.instances)==n

@pytest.mark.parametrize('field,value', [('schema','unknown'),('record_id','synthetic-2'),('updated',1)])
def test_bad_completed_receipt_rejected_before_backend(setup,field,value):
    m=module();_,_,tmp=setup
    run(m,setup,tmp/'run',1);n=len(NumericBackend.instances)
    path=tmp/'run/boundary_0001/receipt.json';receipt=json.loads(path.read_text())
    receipt[field]=value;path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):run(m,setup,tmp/'run',1,resume=True)
    assert len(NumericBackend.instances)==n


def test_concurrent_writer_and_later_denial_preserve_committed_state(setup):
    import fcntl
    m=module();_,_,tmp=setup
    run(m,setup,tmp/'run',1);n=len(NumericBackend.instances)
    with (tmp/'run/.writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='writer'):run(m,setup,tmp/'run',1,resume=True)
    assert len(NumericBackend.instances)==n
    def gate(context):
        if context['phase']=='group':raise ValueError('resource changed')
    with pytest.raises(ValueError,match='resource changed'):
        run(m,setup,tmp/'run',1,resume=True,authorize=gate)
    assert NumericBackend.instances[-1].closed
    assert not (tmp/'run/boundary_0002').exists()


def test_corrupt_skipped_backend_cannot_publish_and_restart_replays(setup):
    m=module();_,_,tmp=setup
    run(m,setup,tmp/'run',1)
    class BadSkip(NumericBackend):
        def rollout(self,*args):
            result=super().rollout(*args);self.state['weight']['m']+=.1;return result
    with pytest.raises(ValueError,match='Inactive group changed optimizer'):
        run(m,setup,tmp/'run',1,resume=True,factory=BadSkip)
    assert not (tmp/'run/boundary_0002').exists()
    result=run(m,setup,tmp/'run',2,resume=True)
    assert result['optimizer_updates']==2 and result['completed_groups']==3

@pytest.mark.parametrize('bad', ['image','parameter_nan','moment_nan','negative_variance'])
def test_independent_review_invalid_input_or_state_never_commits(setup,bad):
    m=module();rows,_,tmp=setup
    class Bad(NumericBackend):
        def update(self,*args):
            result=super().update(*args)
            if bad=='parameter_nan':self.weight[0]=np.nan
            if bad=='moment_nan':self.state['weight']['m'][0]=np.nan
            if bad=='negative_variance':self.state['weight']['v'][0]=-1.
            return result
    def gate(context):
        if context['phase']=='group' and bad=='image':Path(rows[0]['images'][0]).write_bytes(b'UNAPPROVED PIXELS')
    with pytest.raises(ValueError):run(m,setup,tmp/'run',1,factory=Bad,authorize=gate)
    assert not (tmp/'run/boundary_0001').exists()


def test_post_generation_hold_keeps_raw_and_no_update(setup):
    m=module();_,_,tmp=setup
    def gate(context):
        if context['phase']=='post_rollout':raise ValueError('post-generation HOLD')
    with pytest.raises(ValueError,match='post-generation HOLD'):
        run(m,setup,tmp/'run',1,authorize=gate)
    assert list((tmp/'run').glob('.pending-*/rollout_PRIVATE.json'))
    assert not (tmp/'run/boundary_0001').exists()
    assert not any(call[0]=='update' for call in NumericBackend.instances[-1].calls)
