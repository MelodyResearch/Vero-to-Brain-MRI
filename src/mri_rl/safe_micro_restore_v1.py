"""Future-only container-detached restore; CPU/source verified, NOT native PASS.

Installed MLX assigns the caller's state dict directly and replaces nested
step/m/v entries during update. Handing a saved snapshot's containers to that
setter lets a later update corrupt the saved rollback target. This module only
detaches Python containers; it neither changes Adam nor copies model weights by
default. Array-leaf immutability is NOT asserted. An explicitly supplied leaf
copier is required for a backend that mutates array leaves in place.
"""
from pathlib import Path
import ast,hashlib

OPTIMIZER_SOURCE_SHA256='a649eeb65133d1a21eec61ab83529badd0ec50938ee6476b1e59f7ee36def1f2'
HISTORICAL_NATIVE_SOURCE_SHA256='aceda9c4e212667d512e747447f3ef5533b91565f6cebc5a928b8a805c0b96e3'
HISTORICAL_TRANSACTION_SOURCE_SHA256='4343fe861ac4d060afd8509939578b3c87f3fc1d92bfd38983ed09e11086ebb8'


def detach_containers(value,*,copy_leaf=None):
    """Clone dict/list/tuple containers, retaining leaves unless explicitly copied."""
    if copy_leaf is not None and not callable(copy_leaf):raise TypeError('Explicit leaf copier must be callable')
    active=set()
    def clone(item):
        if isinstance(item,(dict,list,tuple)):
            ident=id(item)
            if ident in active:raise ValueError('Cyclic state container cannot be restored')
            active.add(ident)
            try:
                if isinstance(item,dict):return {k:clone(v) for k,v in item.items()}
                if isinstance(item,list):return [clone(v) for v in item]
                return tuple(clone(v) for v in item)
            finally:active.remove(ident)
        return item if copy_leaf is None else copy_leaf(item)
    return clone(value)


def restore_detached(restore,state,rng,*,copy_leaf=None):
    """Call an existing restore with detached containers; no optimizer step."""
    if not callable(restore):raise TypeError('Existing backend restore callback required')
    return restore(detach_containers(state,copy_leaf=copy_leaf),detach_containers(rng,copy_leaf=copy_leaf))


class DetachedRestoreMixin:
    """Future MicroNativeBackend mixin, preceding it in the MRO.

    Both `restore(snapshot)` and `restore(optimizer,RNG)` detach their mutable
    containers. No existing native class is monkeypatched or silently changed.
    A future exact scope/native verification must explicitly select this mixin.
    """
    restore_copy_leaf=None
    def restore(self,state,rng=None):
        copier=self.restore_copy_leaf
        if rng is None:return super().restore(detach_containers(state,copy_leaf=copier))
        return restore_detached(super().restore,state,rng,copy_leaf=copier)


def optimizer_alias_evidence(path):
    """Read exact installed Python source, not MLX runtime or model payloads."""
    path=Path(path);data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    if digest!=OPTIMIZER_SOURCE_SHA256:raise ValueError('Installed optimizer differs from pinned source')
    tree=ast.parse(data);classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    opt,adam=classes['Optimizer'],classes['Adam']
    setter=next(n for n in opt.body if isinstance(n,ast.FunctionDef) and n.name=='state' and
                any(isinstance(d,ast.Attribute) and d.attr=='setter' for d in n.decorator_list))
    gradients=next(n for n in opt.body if isinstance(n,ast.FunctionDef) and n.name=='apply_gradients')
    single=next(n for n in adam.body if isinstance(n,ast.FunctionDef) and n.name=='apply_single')
    aliases=any(isinstance(n,ast.Assign) and isinstance(n.value,ast.Name) and n.value.id=='state' and
                any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='_state' for t in n.targets)
                for n in ast.walk(setter))
    def assigned_keys(node,base):
        return {t.slice.value for n in ast.walk(node) if isinstance(n,ast.Assign) for t in n.targets
                if isinstance(t,ast.Subscript) and ast.unparse(t.value)==base and isinstance(t.slice,ast.Constant)}
    return {'evidence':'EXACT_PINNED_SOURCE_AST_CPU_ONLY','source_hash_matches_pin':True,'source_sha256':digest,
            'state_setter_aliases_argument':aliases,'apply_gradients_replaces_step_entry':'step' in assigned_keys(gradients,'self.state'),
            'adam_replaces_m_and_v_entries':{'m','v'}<=assigned_keys(single,'state'),
            'line_numbers':{'state_setter':setter.lineno,'apply_gradients':gradients.lineno,'adam_apply_single':single.lineno},
            'native_execution':False,'optimizer_updates_executed':0,
            'limitation':'Source and CPU formula proof; actual stopped model failure attribution and future native repair remain separately unverified.'}
