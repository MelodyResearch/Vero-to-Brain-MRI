"""CPU reproduction using selected EXACT installed MLX optimizer AST methods.

NumPy supplies CPU array operations. This is not MLX/native model execution.
"""
import ast,copy,hashlib,json
from pathlib import Path
from typing import Callable
import numpy as np
import pytest
from mri_rl.safe_micro_restore_v1 import detach_containers,restore_detached,DetachedRestoreMixin,optimizer_alias_evidence

ROOT=Path(__file__).resolve().parents[3]
OPT=ROOT/'work/accuracy-recovery-venv-v1/lib/python3.13/site-packages/mlx/optimizers/optimizers.py'
requires_source=pytest.mark.skipif(not OPT.is_file(),reason='Exact local installed MLX source is optional in portable CI; no substitute source')


def source_optimizer():
    evidence=optimizer_alias_evidence(OPT)
    assert evidence['source_hash_matches_pin'] and evidence['state_setter_aliases_argument']
    tree=ast.parse(OPT.read_text());classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    selected={'Optimizer':{'__init__','update','init','apply_gradients','state','step','learning_rate','_maybe_schedule'},
              'Adam':{'__init__','init_single','apply_single'}}
    body=[]
    for name in ('Optimizer','Adam'):
        cls=copy.deepcopy(classes[name]);cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in selected[name]]
        for node in ast.walk(cls):
            if isinstance(node,ast.FunctionDef):
                node.returns=None
                for arg in node.args.args+node.args.kwonlyargs:arg.annotation=None
        body.append(cls)
    class CPU:
        array=staticmethod(np.array);uint64=np.uint64
        square=staticmethod(np.square);sqrt=staticmethod(np.sqrt)
    def tree_map(fn,first,*others):
        if isinstance(first,dict):return {k:tree_map(fn,v,*(other[k] for other in others)) for k,v in first.items()}
        if isinstance(first,list):return [tree_map(fn,v,*(other[i] for other in others)) for i,v in enumerate(first)]
        if isinstance(first,tuple):return tuple(tree_map(fn,v,*(other[i] for other in others)) for i,v in enumerate(first))
        return fn(first,*others)
    ns={'mx':CPU,'Callable':Callable,'tree_map':tree_map}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'<installed-MLX-optimizer-AST-CPU-fixture>','exec'),ns)
    return ns['Adam']


def parent_state():
    return {'step':np.array(52,dtype=np.uint64),'learning_rate':np.array(1e-5),
            'weight':{'m':np.array([.01,-.02]),'v':np.array([.03,.04])}}


def identical(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(identical(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    return np.array_equal(a,b)


class CPUModel:
    def __init__(self,parameters):self.parameters=parameters
    def update(self,parameters):self.parameters=parameters
    def __getitem__(self,key):return self.parameters[key]


@requires_source
def test_installed_source_proves_mutable_container_alias_without_native_import():
    ev=optimizer_alias_evidence(OPT)
    assert ev['state_setter_aliases_argument']
    assert ev['apply_gradients_replaces_step_entry']
    assert ev['adam_replaces_m_and_v_entries']


@requires_source
def test_old_assignment_mutates_saved_adam_snapshot_after_one_real_cpu_formula_update():
    Adam=source_optimizer();opt=Adam(1e-5,betas=[.9,.999],eps=1e-8,bias_correction=False)
    snapshot=parent_state();before=copy.deepcopy(snapshot);opt.state=snapshot
    model=CPUModel({'weight':np.array([.5,-.2])})
    opt.update(model,{'weight':np.array([.3,-.4])})
    assert opt.state is snapshot
    assert int(snapshot['step'])==53 and not identical(snapshot,before)
    assert not np.array_equal(snapshot['weight']['m'],before['weight']['m'])
    assert not np.array_equal(snapshot['weight']['v'],before['weight']['v'])
    opt.state=snapshot
    assert int(opt.state['step'])==53  # attempted rollback cannot recover52


@requires_source
def test_detached_restore_keeps_parent_full_moments_step_rng_repeatable_for_A_B():
    Adam=source_optimizer();opt=Adam(1e-5,betas=[.9,.999],eps=1e-8,bias_correction=False)
    snapshot={'parameters':{'weight':np.array([.5,-.2])},'optimizer':parent_state(),
              'rng':{'python':('CPU',52,('saved',)),'mlx':[np.array([7,9],dtype=np.uint32)]}}
    expected=copy.deepcopy(snapshot);model=CPUModel({'weight':np.zeros(2)});live_rng={}
    def raw_restore(state,rng):
        opt.state=state;live_rng.clear();live_rng.update(rng)
    endpoints=[]
    for _arm in ('A','B'):
        model.update(detach_containers(snapshot['parameters']))
        restore_detached(raw_restore,snapshot['optimizer'],snapshot['rng'])
        assert opt.state is not snapshot['optimizer'] and opt.state['weight'] is not snapshot['optimizer']['weight']
        assert identical(opt.state,expected['optimizer']) and identical(live_rng,expected['rng'])
        opt.update(model,{'weight':np.array([.3,-.4])})
        endpoints.append((copy.deepcopy(model.parameters),copy.deepcopy(opt.state)))
        live_rng['mlx'][0]=np.array([99,99],dtype=np.uint32)
        model.update(detach_containers(snapshot['parameters']))
        restore_detached(raw_restore,snapshot['optimizer'],snapshot['rng'])
        assert identical(snapshot,expected)
        assert identical(model.parameters,expected['parameters'])
        assert identical(opt.state,expected['optimizer']) and int(opt.state['step'])==52
        assert identical(live_rng,expected['rng'])
    assert identical(endpoints[0][0],endpoints[1][0]) and identical(endpoints[0][1],endpoints[1][1])


def test_mixin_detaches_both_transaction_and_two_argument_restore_paths():
    class Base:
        def restore(self,state,rng=None):
            if rng is None:
                self.parameters=state['parameters']
                self.restore(state['optimizer'],state['rng'])
            else:self.state,self.rng=state,rng
    class Fixed(DetachedRestoreMixin,Base):pass
    saved={'parameters':{'weight':[1.]},'optimizer':{'step':52,'nested':[{'m':.1}]},'rng':{'python':(1,),'mlx':[[7,9]]}}
    b=Fixed();b.restore(saved)
    b.state['step']=53;b.state['nested'][0]['m']=4.;b.parameters['weight'][0]=99;b.rng['mlx'][0][0]=99
    assert saved['optimizer']['step']==52 and saved['optimizer']['nested'][0]['m']==.1
    assert saved['parameters']['weight']==[1.] and saved['rng']['mlx']==[[7,9]]


def test_leaf_alias_boundary_is_explicit_and_optional_leaf_copy_is_available():
    value=np.array([1.,2.]);tree={'m':[value]};borrowed=detach_containers(tree)
    assert borrowed is not tree and borrowed['m'] is not tree['m'] and borrowed['m'][0] is value
    copied=detach_containers(tree,copy_leaf=lambda x:x.copy() if isinstance(x,np.ndarray) else x)
    copied['m'][0][0]=9
    assert value[0]==1


@requires_source
def test_source_pin_rejects_different_optimizer_source(tmp_path):
    p=tmp_path/'optimizers.py';p.write_text(OPT.read_text()+'\n# changed\n')
    with pytest.raises(ValueError,match='pinned'):optimizer_alias_evidence(p)
