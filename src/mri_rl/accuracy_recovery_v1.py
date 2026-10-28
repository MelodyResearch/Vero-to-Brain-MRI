"""VERO_TO_MRI_ACCURACY_RECOVERY_V1: backend-neutral CPU core. IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED.

Independently versioned preparation/launch layer for ONE matched shared-rollout experiment from the AUTHENTIC
existing P_calibrated13 Adam52 / boundary114 parent (NOT the lost Adam60). The original shared8 bundle, GSPO
loss, weights (1/8 per admitted response), arm membership (A: F4,T4; B: F2T2,F2T2) and atomic checkpoint
infrastructure are REUSED from shared_rollout_gspo_core_v1 and are not modified. New here: (1) an anchor that
authenticates the ORIGINAL continuous-rl boundary schema (no nested Adam60 offsets are invented), (2) one
combined human scope (integrated parent-restore preflight + one F4+T4 rollout + own-old parity + two arms of at
most one update each), (3) full per-arm gradient archives written BEFORE each step with independent norm/cosine/
Adam-delta recompute, (4) a fail-closed Authorization token with no qualification phase and no approval flag.
The model context keeps the explicit narrow informative group gate in BOTH arms: a LOCAL Vero-style recipe,
not the complete official Vero. No MLX import here; nothing in this module authenticates real model outputs.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl.optimizer_checkpoint import pack_tree,unpack_tree

STATUS=shared.STATUS
EXPERIMENT='VERO_TO_MRI_ACCURACY_RECOVERY_V1'
EXECUTION_SUBTYPE='SHARED_ROLLOUT_ONE_STEP_FROM_AUTHENTIC_ADAM52_NOT_SEQUENTIAL_ON_POLICY'
BUNDLE_EXPERIMENT=shared.EXPERIMENT          # the reused immutable bundle schema keeps its original experiment name
POLICY=shared.POLICY
RECIPE_LABEL='LOCAL_VERO_STYLE_RECIPE_NARROW_INFORMATIVE_GATE_IN_BOTH_ARMS_NOT_COMPLETE_OFFICIAL_VERO'
ACTION='accuracy_recovery_v1_integrated_adam52_rollout_two_arm_one_step'
SCOPE_KIND='FUTURE_EXACT_ACCURACY_RECOVERY_V1_INTEGRATED_RUN_SCOPE'
REVIEWER_KINDS=shared.REVIEWER_KINDS
FROZEN_PARENT={'adapter_sha256':'c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6',
               'receipt_sha256':'e0696a8186d413b2bb35034b5c5eb2568ddc0b05e1b719778de42fa491cc882a','groups':114,'updates':52,'boundary':'boundary_0114'}
PARENT_SCHEMA='ORIGINAL_NON_NESTED_CONTINUOUS_RL_BOUNDARY_V1'
REQUIRED_PARENT_FILES=('adapter.safetensors','adapter_config.json','optimizer/manifest.json','optimizer/state.safetensors')
APPROVAL_KEYS=('approved','authorized','training_authorized','native_approved','READY_FOR_TRAINING','human_authorized')
GRADIENT_SCHEMA='accuracy-recovery-gradient-archive-v1'
OWN_SOURCES=('src/mri_rl/accuracy_recovery_v1.py','src/mri_rl/accuracy_recovery_driver_v1.py','src/mri_rl/accuracy_recovery_entry_v1.py','scripts/run_accuracy_recovery_v1.py',
             'src/mri_rl/shared_rollout_gspo_core_v1.py','src/mri_rl/shared_rollout_gspo_backend_v1.py','src/mri_rl/paired_prompt_groups_v1.py','src/mri_rl/class_margin_aux_v2.py',
             'src/mri_rl/vero_mri_reward.py','src/mri_rl/answer_likelihood.py','src/mri_rl/optimizer_checkpoint.py','src/mri_rl/continuous_rl.py',
             'src/mri_rl/paired_z045_continuation_driver_v1.py')

# Unchanged shared infrastructure, re-exported explicitly (no monkeypatching of the shared modules).
require=shared.require
sha=shared.sha
file_sha256=shared.file_sha256
identity=shared.identity
canonical=shared.canonical
publish_once=shared.publish_once
replace_atomic=shared.replace_atomic
sync_dir=shared.sync_dir
Counters=shared.Counters
CHARGEABLE=shared.CHARGEABLE
GROUP_SIZE,MAX_TOKENS,RESPONSES,WEIGHT=shared.GROUP_SIZE,shared.MAX_TOKENS,shared.RESPONSES,shared.WEIGHT
ROLLOUT_TOLERANCE,OWN_OLD_TOLERANCE=shared.ROLLOUT_TOLERANCE,shared.OWN_OLD_TOLERANCE
RESPONSE_IDS=shared.RESPONSE_IDS
freeze_bundle,load_bundle,arm_plan,gate_report=shared.freeze_bundle,shared.load_bundle,shared.arm_plan,shared.gate_report
validate_bundle_against_run=shared.validate_bundle_against_run
check_old_shapes,parity_report,update_decision=shared.check_old_shapes,shared.parity_report,shared.update_decision
snapshot_identity,scope_digest,consume_once=shared.snapshot_identity,shared.scope_digest,shared.consume_once
MODES=('run',)
EVAL_ACTION='accuracy_recovery_v1_adam52_parent_A_B_readonly_matched_f32_evaluation'
EVAL_SCOPE_KIND='FUTURE_EXACT_ACCURACY_RECOVERY_V1_EVALUATION_SCOPE'
EVAL_OWN_SOURCES=('src/mri_rl/accuracy_recovery_eval_v1.py','scripts/run_accuracy_recovery_eval_v1.py','src/mri_rl/paired_z045_matched_f32_eval_v1.py',
                  'src/mri_rl/paired_z045_evaluation_v1.py','src/mri_rl/shared_rollout_gspo_eval_v1.py')
# Original parent metadata, verified by Codex from the genuine Adam52 receipt (not inferred from the Adam60 chain).
PARENT_ARM={'name':'P_calibrated','reward':'medgrpo_eq3','sampling':'proportional'}
PARENT_SEED=13
PARENT_CALIBRATION_SHA256='0d8564c5373e10b9f7fb3c004d42a45d7c4a690e972b2686309e225b43d1a136'
PARENT_ACTIVE_GROUP_RULE='all-full-format_task-varies_equal-overlong'
PRIVATE_SUBDIR='local_data/accuracy_recovery_v1'


# ---------------------------------------------------------------- canonical PRIVATE output root (survives worktree cleanup)
def canonical_project_root(start=None):
    """The project root R owning the git COMMON dir (<R>/.git), even when running inside <R>/.worktrees/<name>."""
    here=Path(start if start is not None else __file__).resolve()
    for candidate in (here,*here.parents):
        git=candidate/'.git'
        if git.is_dir():return candidate
        if git.is_file():
            text=git.read_text().strip();require(text.startswith('gitdir:'),'Unreadable worktree .git file')
            gitdir=Path(text[len('gitdir:'):].strip());gitdir=(gitdir if gitdir.is_absolute() else candidate/gitdir).resolve()
            marker=gitdir/'commondir';common=(gitdir/marker.read_text().strip()).resolve() if marker.is_file() else gitdir
            require(common.name=='.git' and common.is_dir(),'Git common dir is not <project>/.git');return common.parent
    raise ValueError('No git project root found')


def output_root(start=None):
    return canonical_project_root(start)/PRIVATE_SUBDIR


def require_canonical_output(path,*,start=None):
    """Actual outputs/private payloads live ONLY under <canonical R>/local_data/accuracy_recovery_v1: no worktree copy, no symlink, no traversal."""
    project=canonical_project_root(start);root=project/PRIVATE_SUBDIR
    require(not (project/'local_data').is_symlink() and not root.is_symlink(),'Private output root must not be a symlink')
    raw=os.path.abspath(path);resolved=Path(path).resolve()
    require(raw==str(resolved),'Output path must be canonical (no symlink, no traversal)')
    require(resolved.is_relative_to(root) and resolved!=root,'Output must be strictly inside '+str(root));return resolved


# ---------------------------------------------------------------- authentic ORIGINAL-schema Adam52 parent
def adam52_parent_anchor(path,expected_receipt_sha256,*,frozen=FROZEN_PARENT):
    """Authenticate the original (non-nested) continuous-rl boundary114/Adam52 parent from metadata + the four required files.

    The existing Adam60 validator assumes receipt['binding']['experiment']['adaptive_parent'] and absolute_* offsets;
    the original chain has neither, so nothing is converted or fabricated: counters are the receipt's own 114/52.
    `frozen` exists for fixtures only; production callers (entry/driver) never pass it and the authorization
    scope must carry the module-level frozen identity."""
    path=Path(path)
    require(expected_receipt_sha256==frozen['receipt_sha256'],'Receipt digest is not the frozen Adam52 receipt')
    require(path.is_dir() and not path.is_symlink() and path.name==frozen['boundary'],'Frozen Adam52 boundary directory required')
    receipt_path=path/'receipt.json'
    require(receipt_path.is_file() and not receipt_path.is_symlink() and file_sha256(receipt_path)==frozen['receipt_sha256'],'Parent receipt digest changed')
    receipt=json.loads(receipt_path.read_text())
    require(receipt.get('schema')=='continuous-rl-boundary-v1' and type(receipt.get('completed_groups')) is int and type(receipt.get('optimizer_updates')) is int and
            receipt['completed_groups']==frozen['groups'] and receipt['optimizer_updates']==frozen['updates'],'Original receipt counters differ from frozen group114/Adam52')
    require('absolute_completed_groups' not in receipt and 'absolute_optimizer_updates' not in receipt,'Nested/absolute counters are not the original Adam52 schema')
    binding=receipt.get('binding');experiment=binding.get('experiment') if isinstance(binding,dict) else None
    require(isinstance(experiment,dict) and 'adaptive_parent' not in experiment,'Nested adaptive-parent binding is not the original Adam52 chain')
    files=receipt.get('files_sha256');require(isinstance(files,dict) and all(name in files for name in REQUIRED_PARENT_FILES),'Receipt must list all four required parent files')
    for name in REQUIRED_PARENT_FILES:
        item=path/name;require(item.is_file() and not item.is_symlink() and file_sha256(item)==files[name],'Parent artifact digest changed: '+name)
    require(files['adapter.safetensors']==frozen['adapter_sha256'],'Parent adapter is not the frozen Adam52 adapter')
    manifest=json.loads((path/'optimizer/manifest.json').read_text())
    require(manifest.get('completed_groups')==frozen['groups'] and manifest.get('optimizer_updates')==frozen['updates'] and manifest.get('bindings',{}).get('run')==binding and
            manifest['bindings'].get('adapter_sha256')==files['adapter.safetensors'],'Saved native manifest does not preserve the original parent counters/binding')
    return {'receipt_sha256':frozen['receipt_sha256'],'adapter_sha256':files['adapter.safetensors'],'optimizer_manifest_sha256':files['optimizer/manifest.json'],
            'optimizer_state_sha256':files['optimizer/state.safetensors'],'adapter_config_sha256':files['adapter_config.json'],'groups':frozen['groups'],'updates':frozen['updates'],
            'parent_schema':PARENT_SCHEMA,'required_files_sha256':{name:files[name] for name in REQUIRED_PARENT_FILES},'original_binding':binding}


def parent_disk_snapshot(path):
    """Name/size/mtime/sha256 of EVERY file in the parent directory; symlinks are refused. Compared again after the run."""
    path=Path(path);require(path.is_dir() and not path.is_symlink(),'Parent directory required');inventory={}
    for item in sorted(path.rglob('*')):
        require(not item.is_symlink(),'Parent directory must not contain symlinks')
        if item.is_file():
            stat=item.stat();inventory[str(item.relative_to(path))]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':file_sha256(item)}
    return inventory


def require_parent_unchanged(before,path):
    require(parent_disk_snapshot(path)==before,'Parent checkpoint disk changed (parent must stay immutable)')
    return True


# ---------------------------------------------------------------- budget / counters
def derive_budget(prompt_lengths):
    """The shared one-rollout/two-arm one-step budget (generation charged to THIS run; nothing adopted) + zero-cost integrated preflight."""
    base=shared.derive_budget(prompt_lengths,'train')
    return {**base,'experiment':EXPERIMENT,'execution_subtype':EXECUTION_SUBTYPE,'mode':'accuracy_recovery_run','recipe':RECIPE_LABEL,
            'integrated_preflight':{'forwards':0,'generation':0,'backward':0,'optimizer_updates':0,
                'checks':['restore frozen Adam52 adapter+Adam+RNG','optimizer counter==52','504 trainable signature','restore-twice digest equality','adapter readback hash recorded']},
            'gradient_archive_before_each_step':True,'qualification_phase':False}


def counter_limits(budget):
    require(isinstance(budget,dict) and budget.get('experiment')==EXPERIMENT and budget==derive_budget(budget.get('prompt_lengths')),'Exact derived accuracy-recovery budget required')
    return {key:budget[key] for key in CHARGEABLE}


# ---------------------------------------------------------------- fail-closed authorization token
_ISSUER=object()


def _flag_free(value):
    if isinstance(value,dict):return all(not (k in APPROVAL_KEYS and v is True) and _flag_free(v) for k,v in value.items())
    if isinstance(value,list):return all(_flag_free(v) for v in value)
    return True


class Authorization:
    """Issued only by authorize_launch. Binds the exact scope digest, input hash, budget and the WHOLE reviewed source closure."""
    def __init__(self,token,scope):
        require(token is _ISSUER,'Authorization can only be issued by authorize_launch')
        self.scope=json.loads(canonical(scope));self.scope_sha256=scope_digest(self.scope);self.input_sha256=scope['input_sha256']
        self.source_sha256=dict(scope['source_sha256']);self.budget=json.loads(canonical(scope['budget']));self.parent=dict(scope['parent'])
    def check(self,phase):
        """Called before backend construction and before EVERY phase: any reviewed source file change stops the run."""
        require(scope_digest(self.scope)==self.scope_sha256,'Authorized scope mutated')
        for path,expected in self.source_sha256.items():
            require(Path(path).is_file() and file_sha256(path)==expected,f'Reviewed source changed before {phase}: {path}')
        return True
    def matches(self,*,scope_sha256,source_sha256,input_sha256,budget,parent):
        return (self.scope_sha256==scope_sha256 and self.source_sha256==source_sha256 and self.input_sha256==input_sha256 and self.budget==budget and
                self.parent=={'receipt_sha256':parent['receipt_sha256'],'adapter_sha256':parent['adapter_sha256']})


def _authorize(scope,human,review,*,consumed,frozen,action,scope_kind,budget_ok,required_sources):
    require(callable(consumed),'A once-only human-instruction ledger is mandatory')
    require(isinstance(scope,dict) and scope.get('scope_kind')==scope_kind and 'mode' not in scope,'Preparation is not an exact launch scope')
    require(_flag_free(scope),'A local approval flag cannot authorize')
    require(scope.get('experiment')==EXPERIMENT and scope.get('execution_subtype')==EXECUTION_SUBTYPE and scope.get('numerical_policy')==POLICY and scope.get('recipe_label')==RECIPE_LABEL,
            'Experiment/subtype/policy/recipe differs')
    require(scope.get('parent')=={'receipt_sha256':frozen['receipt_sha256'],'adapter_sha256':frozen['adapter_sha256']},'Scope parent is not the frozen Adam52 identity')
    budget=scope.get('budget');require(isinstance(budget,dict) and budget_ok(budget),'Scope budget is not the exact derived budget')
    closure=scope.get('source_sha256')
    require(isinstance(closure,dict) and all(sha(v) for v in closure.values()) and all(any(p.endswith(name) for p in closure) for name in required_sources),'Complete source closure including every own source required')
    require(all(Path(p).is_file() and file_sha256(p)==v for p,v in closure.items()),'Reviewed source closure differs from current source')
    require(sha(scope.get('input_sha256')) and scope.get('implemented_by') in REVIEWER_KINDS,'Input hash and implementing agent kind required')
    binding=scope_digest(scope)
    require(isinstance(human,dict) and human.get('authorized') is True and human.get('action')==action and human.get('scope_sha256')==binding and sha(human.get('human_instruction_sha256')),
            'Fresh exact human authorization for this action and scope required')
    require(isinstance(review,dict) and review.get('scope_sha256')==binding and review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('reviewer_id') and
            review.get('reviewer_kind') in REVIEWER_KINDS and review.get('reviewer_kind')!=scope['implemented_by'] and review.get('source_sha256')==closure and
            review.get('input_sha256')==scope['input_sha256'],'Independent exact source/input/scope review by a different agent kind required')
    consumed(human['human_instruction_sha256'])         # last: a refused authorization consumes nothing
    return Authorization(_ISSUER,scope)


def authorize_launch(scope,human,review,*,consumed,frozen=FROZEN_PARENT):
    """Exact scope + FRESH human action + independent exact review, all digest-bound. No qualification proof (none exists for this
    parent); no mode; a local approval flag anywhere in the scope never authorizes. `consumed` (once-only ledger) is mandatory."""
    return _authorize(scope,human,review,consumed=consumed,frozen=frozen,action=ACTION,scope_kind=SCOPE_KIND,
                      budget_ok=lambda b:b==derive_budget(b.get('prompt_lengths')),required_sources=OWN_SOURCES)


def authorize_evaluation(scope,human,review,*,consumed,frozen=FROZEN_PARENT):
    """Same fail-closed contract for the read-only Parent52/A/B evaluation; its own action and ledger, so ONE genuine human instruction
    may bind both exact scopes without either scope ever authorizing the other. No old qualification proof is consulted."""
    from mri_rl import shared_rollout_gspo_eval_v1 as shared_eval
    return _authorize(scope,human,review,consumed=consumed,frozen=frozen,action=EVAL_ACTION,scope_kind=EVAL_SCOPE_KIND,
                      budget_ok=lambda b:b.get('checkpoints') in (1,2,3) and b==shared_eval.eval_budget(b['checkpoints']),required_sources=(*OWN_SOURCES,*EVAL_OWN_SOURCES))


# ---------------------------------------------------------------- gradient archive (BEFORE any optimizer step)
def _fsync(path):
    with Path(path).open('rb') as handle:os.fsync(handle.fileno())


def archive_gradient(*,backend,directory,arm,planned,gradient,evidence,bundle_sha256,parent_identity,anchor,parent_path,profile,decision,audit):
    """Persist the arm's FULL actual weighted-sum gradient arrays plus metadata sufficient for independent L2/cosine/Adam-delta recompute.

    gradient None (no admitted group) is recorded explicitly: no tensors file, no invented gradient or advantage."""
    require(arm in shared.ARMS and sha(bundle_sha256) and sha(parent_identity),'Arm, bundle and parent identity required')
    require(gradient is not None or not decision.get('update'),'A gradient-less archive cannot coexist with an update decision')
    require(gradient is not None or planned.admitted_groups==0,'Admitted groups require an actual gradient')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    admitted=set(planned.admitted_response_ids);items=planned.plan.items
    record={'schema':GRADIENT_SCHEMA,'arm':arm,'experiment':EXPERIMENT,'bundle_sha256':bundle_sha256,'parent_identity':parent_identity,
        'parent':{k:anchor[k] for k in ('receipt_sha256','adapter_sha256','optimizer_manifest_sha256','optimizer_state_sha256','groups','updates')},'parent_checkpoint':str(parent_path),
        'optimizer_profile':profile['optimizer'],'adam_step_before_update':anchor['updates'],'decision':decision,'admitted_response_ids':sorted(admitted),
        'advantages':{i.response.response_id:(i.advantage if i.response.response_id in admitted else None) for i in items},
        'weights':{i.response.response_id:(i.weight if i.response.response_id in admitted else 0.) for i in items},
        'pre_restore_audit':audit,'weighted_loss':evidence.get('weighted_loss'),'backward_calls':evidence.get('backward_calls'),'gradient_l2_reported':evidence.get('gradient_l2')}
    meta=directory/f'arm_{arm}_gradient_meta_PRIVATE.json'
    if gradient is None:
        record.update(gradient=None,tensors_file=None,tensors_sha256=None,layout=None,layout_sha256=None,tensor_count=0,reason=decision.get('reason'))
    else:
        layout,arrays=pack_tree(gradient);target=directory/f'arm_{arm}_gradient_PRIVATE.safetensors'
        require(arrays,'Nonempty gradient tree required')
        if target.exists() or meta.exists():raise FileExistsError('Gradient archive already exists; never overwritten')
        backend.save_arrays(target,arrays);_fsync(target)
        record.update(gradient='ARRAYS_IN_TENSORS_FILE',tensors_file=target.name,tensors_sha256=file_sha256(target),layout=layout,layout_sha256=identity(layout),
                      tensor_count=len(arrays),tensor_bytes=sum(int(getattr(v,'nbytes',0)) for v in arrays.values()),reason=decision.get('reason'))
    publish_once(meta,record)       # the metadata link is the archive commit marker; FileExistsError on a second publish
    return record


def _flatten(tree,prefix=''):
    if isinstance(tree,dict):
        for key,value in tree.items():yield from _flatten(value,f'{prefix}.{key}' if prefix else key)
    elif isinstance(tree,(list,tuple)):
        for index,value in enumerate(tree):yield from _flatten(value,f'{prefix}.{index}' if prefix else str(index))
    else:yield prefix,tree


def load_gradient_archive(meta_path,load_arrays):
    """Verify tensors/layout hashes and return {flat_name: float array}; None for an explicitly gradient-less (inactive) arm."""
    meta_path=Path(meta_path);record=json.loads(meta_path.read_text())
    require(record.get('schema')==GRADIENT_SCHEMA,'Unknown gradient archive')
    if record['gradient'] is None:
        require(record['tensors_file'] is None and record['tensors_sha256'] is None,'Gradient-less record must not name tensors');return None
    path=meta_path.with_name(record['tensors_file'])
    require(file_sha256(path)==record['tensors_sha256'],'Gradient tensors hash differs from the archive record')
    require(identity(record['layout'])==record['layout_sha256'],'Gradient layout hash differs')
    flat=dict(_flatten(unpack_tree(record['layout'],load_arrays(path))))
    require(len(flat)==record['tensor_count'],'Gradient tensor inventory differs');return {k:np_array(v) for k,v in flat.items()}


def np_array(value):
    import numpy as np
    return np.asarray(value)


def gradient_l2(flat):
    return math.sqrt(sum(float((v.astype('float64')**2).sum()) for v in flat.values()))


def cosine(left,right):
    """Cosine of two archived arm gradients over the shared flat inventory; None if either arm is gradient-less or a norm is zero."""
    if left is None or right is None:return None
    require(set(left)==set(right),'Gradient inventories differ')
    dot=sum(float((left[k].astype('float64')*right[k].astype('float64')).sum()) for k in left);scale=gradient_l2(left)*gradient_l2(right)
    return dot/scale if scale>0 else None


def adam_delta(flat_state,flat_gradient,optimizer):
    """Independent recompute of ONE no-bias-correction Adam step from the PARENT moments: delta=-lr*m'/(sqrt(v')+eps), m',v' updated with g."""
    if flat_gradient is None:return None
    require(optimizer.get('name')=='Adam' and optimizer.get('bias_correction') is False,'Only the frozen no-bias-correction Adam profile is recomputable here')
    import numpy as np
    lr,(b1,b2),eps=optimizer['learning_rate'],optimizer['betas'],optimizer['eps'];delta={}
    for name,g in flat_gradient.items():
        m=np.asarray(flat_state[name+'.m'],dtype='float64');v=np.asarray(flat_state[name+'.v'],dtype='float64');g=g.astype('float64')
        m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g*g;delta[name]=-lr*m/(np.sqrt(v)+eps)
    return delta


def load_gradient_tree(meta_path,load_arrays):
    """(record, backend-array tree) read back from disk for the optimizer step; (record, None) for an explicitly gradient-less arm.
    Authenticates tensors/layout hashes and re-derives the L2 norm against the value recorded when the gradient was computed."""
    meta_path=Path(meta_path);record=json.loads(meta_path.read_text());require(record.get('schema')==GRADIENT_SCHEMA,'Unknown gradient archive')
    if record['gradient'] is None:
        require(record['tensors_file'] is None and record['tensors_sha256'] is None,'Gradient-less record must not name tensors');return record,None
    path=meta_path.with_name(record['tensors_file'])
    require(file_sha256(path)==record['tensors_sha256'] and identity(record['layout'])==record['layout_sha256'],'Gradient tensors/layout hash differs from the archive record')
    tree=unpack_tree(record['layout'],load_arrays(path));flat={k:np_array(v) for k,v in _flatten(tree)}
    reported=record.get('gradient_l2_reported');require(len(flat)==record['tensor_count'] and isinstance(reported,(int,float)),'Gradient inventory/norm record incomplete')
    actual=gradient_l2(flat);require(abs(actual-reported)<=1e-4*max(abs(reported),1e-12),'Archived gradient norm differs from the recorded norm')
    return record,tree


def comparison_record(metas,load_arrays):
    """Independent A/B comparison over the AUTHENTICATED archives: norms, cosine, admitted sets. None is explicit, never invented."""
    require(set(metas)=={'A','B'},'Both arm archives required');records={};flats={}
    for arm,path in metas.items():
        records[arm]=json.loads(Path(path).read_text());flats[arm]=load_gradient_archive(path,load_arrays)
    require(records['A']['bundle_sha256']==records['B']['bundle_sha256'] and records['A']['parent_identity']==records['B']['parent_identity'],'Arm gradients belong to different bundles/parents')
    cosine_value=cosine(flats['A'],flats['B']);reasons=[f"{a}:{records[a]['reason']}" for a in 'AB' if flats[a] is None]
    if not reasons and cosine_value is None:reasons=['ZERO_NORM']
    return {'schema':'accuracy-recovery-gradient-comparison-v1','bundle_sha256':records['A']['bundle_sha256'],'parent_identity':records['A']['parent_identity'],
            'norms':{a:(gradient_l2(flats[a]) if flats[a] is not None else None) for a in 'AB'},'cosine':cosine_value,'cosine_reason':';'.join(reasons) or None,
            'admitted_response_ids':{a:records[a]['admitted_response_ids'] for a in 'AB'},
            'archive_sha256':{a:{'meta':file_sha256(metas[a]),'tensors':records[a]['tensors_sha256']} for a in 'AB'}}


def adam_next_state(flat_state,flat_gradient,optimizer):
    """Independent recurrence from the PARENT Adam state: m',v',step+1 and the parameter delta. None for a gradient-less arm (no momentum-only step)."""
    if flat_gradient is None:return None
    delta=adam_delta(flat_state,flat_gradient,optimizer);import numpy as np
    b1,b2=optimizer['betas'];result={'step':np.asarray(flat_state['step'])+1,'delta':delta}
    for name,g in flat_gradient.items():
        g=g.astype('float64');result[name+'.m']=b1*np.asarray(flat_state[name+'.m'],dtype='float64')+(1-b1)*g
        result[name+'.v']=b2*np.asarray(flat_state[name+'.v'],dtype='float64')+(1-b2)*g*g
    return result


def compare_adam_after(expected,after,*,rtol=1e-4,atol=1e-12):
    import numpy as np
    for key,value in expected.items():
        if key=='delta':continue
        require(key in after and np.allclose(np.asarray(after[key],dtype='float64'),np.asarray(value,dtype='float64'),rtol=rtol,atol=atol),'Adam recurrence differs from the saved after-state: '+key)
    return True
