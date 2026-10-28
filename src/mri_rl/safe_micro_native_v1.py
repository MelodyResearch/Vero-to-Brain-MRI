"""Lazy native bridge for the SAFE MICRO transaction. SOURCE_IMPLEMENTED_NOT_NATIVE_PASS.

Importing this module imports no MLX and loads nothing. It supplies (a) a native backend whose constructor runs the UNCHANGED,
hash-bound `NativeGSPOBackend.__init__` AST in a distinct globals namespace whose COPIED PACED_POLICY carries the 12 GiB memory
limit / 256 MiB cache limit (the historical module global, 18 GiB, is never mutated and no old source is edited), (b) the gradient /
archive helpers `safe_micro_training_v1.execute_micro` needs, and (c) a factory that refuses to construct anything unless a genuine
exact authorization issued by the root entry plus the matching model-B resource gate are supplied. Source readiness is NOT native PASS.
"""
from __future__ import annotations
import ast
import copy
import gc
import hashlib
import importlib
import importlib.util
from pathlib import Path
from mri_rl import safe_micro_training_v1 as micro
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl.accuracy_stage_native_v1 import StageNativeBackend, restore_parent

STATUS = 'SOURCE_IMPLEMENTED_NOT_NATIVE_PASS'
require = shared.require
SOURCE_MODULE = 'mri_rl.mlx_continuous_gspo'
SOURCE_FILE_SHA256 = 'aceda9c4e212667d512e747447f3ef5533b91565f6cebc5a928b8a805c0b96e3'
MEMORY_GUIDELINE_GIB, CACHE_LIMIT_MIB = 12, 256
ENTRY_MODULE = 'mri_rl.safe_micro_entry_v1'
MODEL_PHASE = 'MICRO_TRAIN'


class PendingAuthorization(RuntimeError):
    """Fail-closed: no genuine exact authorization class/instance is available. Never self-authorize."""


# ---------------------------------------------------------------- capped constructor (AST-identical, distinct globals)
def _original_init_node(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeGSPOBackend')
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')


def capped_init_binding():
    """Compile the unchanged historical __init__ AST into a COPIED-globals namespace with PACED_POLICY 12 GiB / 256 MiB."""
    module = importlib.import_module(SOURCE_MODULE)
    path = Path(module.__file__)
    require(shared.file_sha256(path) == SOURCE_FILE_SHA256, 'Historical native source differs from the pinned reviewed source; refusing')
    node = _original_init_node(path.read_text())
    namespace = dict(vars(module))
    policy = copy.deepcopy(module.PACED_POLICY)
    policy.update(memory_guideline_gib=MEMORY_GUIDELINE_GIB, cache_limit_mib=CACHE_LIMIT_MIB)
    namespace['PACED_POLICY'] = policy
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<unchanged-NativeGSPOBackend.__init__-capped-12GiB>', 'exec'), namespace)
    return {'init': namespace['__init__'], 'ast_sha256': hashlib.sha256(ast.dump(node).encode()).hexdigest(), 'source_sha256': SOURCE_FILE_SHA256,
            'policy': policy, 'node': node}


class MicroNativeBackend(StageNativeBackend):
    """StageNativeBackend whose single construction path applies the local 12 GiB / 256 MiB limits BEFORE any materialization."""
    evidence_kind = micro.NATIVE_KIND

    def __init__(self, adapter, config, *, model_path, contract, seed, before_unit, cache=None):
        capped_init_binding()['init'](self, adapter, config, model_path=model_path, contract=contract, seed=seed, before_unit=before_unit, cache=cache)
        # The historical loader installs mlx_vlm.generate as an instance field; it must not shadow generate(unit, lengths).
        self._legacy_generation_api = self.__dict__.pop('generate', None)

    # ---- helpers the backend-neutral transaction requires (float32, serial, one tree at a time)
    def optimizer_step(self):
        return int(self.optimizer_state()['step'].item())

    def scale_gradient(self, tree, weight):
        value = self.tree_map(lambda g: g.astype(self.mx.float32) * weight, tree)
        self.mx.eval(value)
        return value

    def add_scaled(self, total, tree, weight):
        value = self.tree_map(lambda a, g: a + g.astype(self.mx.float32) * weight, total, tree)
        self.mx.eval(value)
        return value

    def gradient_l2(self, tree):
        return self._checked_gradient(tree)

    def gradient_dtypes(self, tree):
        return {str(v.dtype).split('.')[-1] for _, v in self.tree_flatten(tree)}

    def save_gradient(self, path, tree):
        require(str(path).endswith('.safetensors'), 'Gradient archive must be a .safetensors path')
        self.save_arrays(path, dict(self.tree_flatten(tree)))

    def release_graphs(self):
        """Drop the prepared image context/graph before the next task or image graph starts."""
        self._active = self._active_modality = None
        gc.collect()
        self.mx.clear_cache()


# ---------------------------------------------------------------- authorization-gated factory / execution
def _entry_authorization_class():
    if importlib.util.find_spec(ENTRY_MODULE) is None:
        raise PendingAuthorization('PENDING_GENUINE_AUTHORIZATION: the root entry module ' + ENTRY_MODULE + ' is absent')
    cls = getattr(importlib.import_module(ENTRY_MODULE), 'Authorization', None)
    if not isinstance(cls, type):
        raise PendingAuthorization('PENDING_GENUINE_AUTHORIZATION: the root entry exposes no Authorization class')
    return cls


def check_authority(authorization, resource_gate):
    """Exact issued authorization for the model-B MICRO_TRAIN phase and the resource gate built from that same authorization."""
    from mri_rl import safe_micro_resources_v1 as resources
    cls = _entry_authorization_class()
    require(isinstance(authorization, cls), 'Authorization must be an instance issued by the root entry')
    scope = getattr(authorization, 'scope', None)
    require(isinstance(scope, dict) and scope.get('resource_stage') == 'B_MICRO' and scope.get('execution_stage') == MODEL_PHASE,
            'Authorization is not the exact model-B MICRO_TRAIN phase')
    require(isinstance(resource_gate, resources.ResourceGate) and resource_gate.stage == 'B_MICRO' and resource_gate.authorization is authorization and
            resource_gate.scope == scope, 'Resource gate must be the model-B gate built from this exact authorization')
    authorization.check('micro_native_authority')
    return True


def make_native_backend(*, authorization, resource_gate, adapter, config, model_path, contract, seed, cache):
    """Construct the single resident native backend. Requires the issued exact authorization; the resource gate is the unit guard."""
    check_authority(authorization, resource_gate)
    require(isinstance(cache, dict) and set(cache) == {'dir'}, 'Exact certified cache directory required')
    return MicroNativeBackend(adapter, config, model_path=model_path, contract=contract, seed=seed, before_unit=resource_gate, cache=cache)


def configure_and_restore_parent(backend, *, grader, expected_images, prompt_lengths, work, binding, parent_dir, anchor):
    """Reuse the historical stage configuration and Adam52/504 parent restore; return the trusted common-parent identity."""
    require(isinstance(backend, MicroNativeBackend), 'Native micro backend required')
    backend.configure_stage(grader=grader, expected_images=expected_images, prompt_lengths=prompt_lengths, work=work, binding=binding)
    restore_parent(backend, parent_dir, anchor)
    return {'policy': backend.policy_sha(), 'adam': backend.adam_digest(), 'rng': backend.rng_digest()}


def execute_native_micro(*, authorization, resource_gate, backend, units, prompt_lengths, reward_policy_sha256, output, expected_parent,
                         checkpoint_receipt_sha256=None, verify_parent_disk=None):
    """Run the micro transaction on an authorized native backend. The guard at every boundary is authorization + resource gate."""
    check_authority(authorization, resource_gate)
    require(isinstance(backend, MicroNativeBackend), 'Native micro backend required')
    return micro.execute_micro(backend=backend, units=units, prompt_lengths=prompt_lengths, reward_policy_sha256=reward_policy_sha256, output=output,
                               before=resource_gate, expected_parent=expected_parent, checkpoint_receipt_sha256=checkpoint_receipt_sha256,
                               verify_parent_disk=verify_parent_disk)
