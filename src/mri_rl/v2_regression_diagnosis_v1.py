"""V2_REGRESSION_DIAGNOSIS_V1: CPU-only saved-evidence diagnosis of the Accuracy Pilot V2 negative result.

Author: Claude Code (claude-opus-5-5), 2026-10-21. STATUS: IMPLEMENTED_CPU_PENDING_INDEPENDENT_EXECUTION.

Strictly READ-ONLY over the immutable V2 run. No MLX/torch/JAX import, no model load, no generation, no backward,
no optimizer step, no new gradient. Float32 tensors are streamed one parameter at a time with
safetensors(framework='np'); tokenizer vocabulary files (optional) are read as plain JSON only.

Re-derived, never trusted (each compared against what was saved):
  * every one of 4 macro-updates (2 arms x 2 segments) / 8 units per arm: unit bundles are rebuilt from the raw
    rollouts (segment 0) or re-validated from the saved own-policy bundles (segment 1) and must hash to the bundle
    SHA-256s in the self-hashed, journal-bound segment receipts; groups, the inherited gate, advantages and 1/8
    weights are recomputed with the UNCHANGED planners (ms.arm_plan_for -> shared.arm_plan / ppg.make_plan);
  * rewards are optionally re-graded from response text with the production grader + calibration;
  * every Adam step is recomputed on CPU with the installed formula (float32 emulation) against the saved moments
    and adapters, with rounding tolerances DECLARED BEFORE COMPUTATION (TOLERANCES).
Measured but explicitly limited:
  * only JOINT gradients were saved per macro-update. Task-specific MRI/Dallas gradients are NOT inferred; their
    cosines and Adam projections are NOT_VERIFIED unless a segment is task-pure by construction (no admitted
    groups of the other task, then the joint IS that task's gradient) or a two-unit norm identity applies.
  * g . delta, momentum-only (g=0) baseline, and exact actual-denominator splits are JOINT quantities, not MRI
    class-margin derivatives. Greedy-path selected-token log-probabilities are not class margins.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path

from mri_rl import accuracy_pilot_v2_core as core
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl import paired_prompt_groups_v1 as ppg
from mri_rl.mlx_gspo_smoke import advantages_from_rewards, overlong_penalty
from mri_rl.vero_mri_reward import _answer_and_format

SCHEMA = 'v2-regression-diagnosis-v1'
STATUS = 'IMPLEMENTED_CPU_PENDING_INDEPENDENT_EXECUTION'
require = shared.require
identity, canonical = ppg.identity, ppg.canonical
LABELS = core.LABELS
UNIT_WEIGHT = core.UNIT_WEIGHT
ARMS = ('A', 'B')
ZERO = 1e-12
FORBIDDEN_MODULES = ('mlx', 'mlx_vlm', 'mlx_lm', 'torch', 'jax', 'jaxlib', 'tensorflow')
ACCURACY_WEIGHT, FORMAT_WEIGHT = .8, .2          # vero_mri_reward / two_task_training callback: score = .8*acc + .2*format
BOUNDARIES = {'model_loads': 0, 'MLX_imports': 0, 'GPU': 0, 'new_gradients_or_backward': 0, 'optimizer_updates': 0, 'new_generation': 0,
              'installs': 0, 'old_final100_or_sealed_new16_outputs': 0, 'writes_inside_run_directory': 0}
EVIDENCE_TIERS = ('OBSERVED', 'REPRODUCED', 'DERIVED', 'INFERRED', 'PROPOSED', 'NOT_VERIFIED')

# ---------------------------------------------------------------- Adam profile (installed formula) and tolerances
ADAM_PROFILE = {'name': 'Adam', 'beta1': .9, 'beta2': .999, 'eps': 1e-8, 'bias_correction': False,
                'learning_rate': 'read from every saved optimizer state (float32 scalar); must be identical before/after',
                'formula': 'installed mlx.optimizers.Adam.apply_single, bias_correction=False: m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g^2; '
                           'w=w-lr*m/(sqrt(v)+eps); Optimizer.apply_gradients increments step before apply_single',
                'dtype': 'float32 arrays; Python scalars b1,(1-b1),b2,(1-b2),eps cast to float32; lr cast to gradient dtype'}
ADAM_CONSTRUCTOR_PIN = ('src/mri_rl/mlx_continuous_gspo.py', 'optim.Adam(learning_rate=1e-5,betas=[.9,.999],eps=1e-8,bias_correction=False)')
MLX_ADAM_SOURCE_PINS = ('m = b1 * m + (1 - b1) * gradient', 'v = b2 * v + (1 - b2) * mx.square(gradient)',
                        'return parameter - lr * m / (mx.sqrt(v) + eps)', 'self.state["step"] = self.step + 1')
F32_EPS = 2. ** -23
TOLERANCES = {
    'declared_before_computation': True,
    'moment_elementwise': {'relative': 4 * F32_EPS, 'absolute': 1e-30,
                           'rule': '|pred-saved| <= relative*max(|pred|,|saved|) + absolute (float32 emulation; allows FMA/rounding-order differences)'},
    'parameter_elementwise': {'spacing_multiple': 2., 'update_relative': 8 * F32_EPS,
                              'rule': '|theta_pred-theta_saved| <= 2*spacing32(|theta_saved|) + 8*eps32*|lr*m/(sqrt(v)+eps)| using SAVED m,v'},
    'beta_fit_absolute': 1e-5, 'advantage_absolute': 1e-12, 'reward_absolute': 1e-12,
    'decomposition_residual_relative': 1e-4, 'gradient_norm_relative': 1e-4,
    'PASS_rule': 'An elementwise recomputation PASSES iff zero elements exceed its declared tolerance; counts/max errors reported either way.'}
TOLERANCES_SHA256 = identity(TOLERANCES)

REUSED_SOURCES = ('src/mri_rl/accuracy_pilot_v2_core.py', 'src/mri_rl/multistep_gspo_core_v1.py', 'src/mri_rl/shared_rollout_gspo_core_v1.py',
                  'src/mri_rl/paired_prompt_groups_v1.py', 'src/mri_rl/uniform_balance_continuation_gate_v1.py', 'src/mri_rl/mlx_gspo_smoke.py',
                  'src/mri_rl/vero_mri_reward.py', 'src/mri_rl/paired_z045_evaluation_v1.py', 'src/mri_rl/two_task_training.py',
                  'src/mri_rl/reward_replay.py', 'src/mri_rl/visit_correspondence.py', 'src/mri_rl/mlx_continuous_gspo.py',
                  'src/mri_rl/optimizer_checkpoint.py')


# ================================================================ isolation guard
class _ImportBlocker:
    """Meta-path finder refusing MLX/torch/JAX/TensorFlow for the whole analysis process."""
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in FORBIDDEN_MODULES:
            raise ImportError('blocked by v2_regression_diagnosis_v1 CPU-only guard: ' + name)
        return None


def install_import_guard():
    loaded = sorted(m for m in sys.modules if m.split('.')[0] in FORBIDDEN_MODULES)
    require(not loaded, 'Forbidden heavyweight modules already imported: ' + ','.join(loaded))
    if not any(isinstance(f, _ImportBlocker) for f in sys.meta_path):
        sys.meta_path.insert(0, _ImportBlocker())
    return True


def forbidden_modules_loaded():
    return sorted(m for m in sys.modules if m.split('.')[0] in FORBIDDEN_MODULES)


# ================================================================ authentication helpers
def sha256(path):
    return shared.file_sha256(path)


def require_file_sha(path, expected, label):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), label + ': regular non-symlink file required')
    require(shared.sha(expected), label + ': expected SHA-256 missing')
    require(sha256(path) == expected, label + ': SHA-256 changed (mutated or wrong file)')
    return expected


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_self_hash(record, field='receipt_sha256', label='receipt'):
    require(isinstance(record, dict) and shared.sha(record.get(field)), label + ': self-hash field missing')
    require(record[field] == identity({k: v for k, v in record.items() if k != field}), label + ': self-hash differs (mutated record)')
    return record[field]


JOURNAL_COMMIT_FIELDS = {'value', 'files', 'reservation_sha256', 'record_sha256'}


def verify_journal_commit(commit_path, required, *, value=None):
    """Re-check one safe_micro_journal_v1 commit exactly as Journal.get does, read-only (the immutable format is not rewritten):
    record_sha256 self-hash, reservation file hash + key + binding hash, EVERY file-map entry, then the required files and value.
    A locally coherent re-hash still passes HERE; it is rejected by the external anchors (verify_stage_anchors / receipt links)."""
    commit_path = Path(commit_path)
    commit = load_json(commit_path)
    require(isinstance(commit, dict) and set(commit) == JOURNAL_COMMIT_FIELDS, 'Journal commit schema differs: ' + commit_path.parent.name)
    require(commit['record_sha256'] == identity({k: v for k, v in commit.items() if k != 'record_sha256'}),
            'Journal commit record_sha256 differs (mutated commit): ' + commit_path.parent.name)
    claim = commit_path.parent / 'reservation_PRIVATE.json'
    require_file_sha(claim, commit['reservation_sha256'], 'journal reservation ' + commit_path.parent.name)
    reserved = load_json(claim)
    require(reserved.get('key') == commit_path.parent.name, 'Journal reservation key differs from its operation directory')
    binding_path = commit_path.parent.parent / 'binding_PRIVATE.json'
    require(binding_path.is_file() and not binding_path.is_symlink(), 'Journal binding record required')
    binding_sha = identity(load_json(binding_path))
    require(reserved.get('binding_sha256') == binding_sha, 'Journal reservation binding differs: ' + commit_path.parent.name)
    files = commit['files']
    require(isinstance(files, dict) and files, 'Journal commit file map required')
    for name, h in files.items():
        require_file_sha(name, h, 'journal-committed ' + Path(name).name)
    for p in required:
        key = str(Path(p).resolve())
        require(key in files, 'File absent from its journal commit: ' + Path(p).name)
        require_file_sha(p, files[key], 'journal-bound ' + Path(p).name)
    if value is not None:
        require(commit.get('value') == value, 'Journal commit value differs from the saved record')
    return {'commit_sha256': sha256(commit_path), 'record_sha256': commit['record_sha256'], 'reservation_sha256': commit['reservation_sha256'],
            'binding_sha256': binding_sha, 'committed_files_verified': len(files)}


STAGES = ('P', 'C', 'A', 'B', 'EP', 'EA', 'EB')


def verify_stage_anchors(run_dir, results):
    """External anchor: the PUBLIC committed result's evidence.all_stage_hashes fixes every stage terminal and closed exit record.
    Each stage's verified marker must name exactly those hashes and an exit file inside the run directory."""
    run_dir = Path(run_dir).resolve()
    stages = results['evidence']['all_stage_hashes']
    require(set(stages) == set(STAGES), 'Committed result does not anchor exactly the seven V2 stages')
    out = {}
    for stage in STAGES:
        h = stages[stage]
        terminal_path = run_dir / f'stage{stage}_PRIVATE' / 'terminal_PRIVATE.json'
        require_file_sha(terminal_path, h['terminal_sha256'], f'stage {stage} terminal (committed-result anchor)')
        verified = load_json(run_dir / f'stage{stage}_verified_PRIVATE.json')
        require(verified.get('terminal_sha256') == h['terminal_sha256'] and verified.get('exit_sha256') == h['exit_sha256']
                and verified.get('scope_file_sha256') == results['scope_file_sha256'], f'stage {stage}: verified marker differs from the committed anchors')
        exit_path = Path(verified['exit_path']).resolve()
        require(exit_path.is_relative_to(run_dir), f'stage {stage}: exit record outside the run directory')
        require_file_sha(exit_path, h['exit_sha256'], f'stage {stage} exit (committed-result anchor)')
        terminal = load_json(terminal_path)
        require(terminal.get('status') == 'PASS' and terminal.get('stage', stage) == stage, f'stage {stage}: terminal is not this stage PASS')
        out[stage] = terminal
    return out


def link_terminals(terminals, receipts, dev_rows):
    """Link the journal-verified records to the anchored terminals where the native terminal schema carries them."""
    links = {}
    for arm in ARMS:
        t = terminals[arm]
        want = [receipts[(arm, 0)]['receipt_sha256'], receipts[(arm, 1)]['receipt_sha256']]
        require(t['terminal']['receipts_sha256'] == want and t['arm']['receipt_sha256'] == want, f'stage {arm}: segment receipts differ from the anchored terminal')
        links[arm] = 'SEGMENT_RECEIPTS_MATCH_ANCHORED_TERMINAL'
    for label, stage in (('Parent', 'EP'), ('A', 'EA'), ('B', 'EB')):
        require(identity(terminals[stage]['metrics']) == identity(core.greedy_summary(dev_rows[label])),
                f'stage {stage}: DEV rows do not reproduce the anchored terminal metrics')
        links[stage] = 'DEV_ROWS_REPRODUCE_ANCHORED_TERMINAL_METRICS'
    links['C'] = 'TERMINAL_HASH_ANCHORED_ONLY: common record linked through segment-0 receipts (common_receipt_sha256/bundle_sha256)'
    links['P'] = 'TERMINAL_HASH_ANCHORED; prompt condition compared with the committed result'
    links['old_independent_digest_evidence'] = 'NOT_VERIFIED: not re-read in this pass'
    return links


def source_pins(worktree, scope, modules):
    """Reused planner/reward sources must be byte-identical to the scope-bound native-run sources; imported modules must be them."""
    worktree = Path(worktree).resolve()
    bound = scope.get('source_sha256', {})
    report = {}
    for rel in REUSED_SOURCES:
        path = worktree / rel
        actual = sha256(path) if path.is_file() else None
        scoped = bound.get(str(path))
        status = 'MISSING' if actual is None else ('NOT_IN_SCOPE' if scoped is None else ('MATCH' if scoped == actual else 'DRIFT'))
        report[rel] = {'sha256': actual, 'scope_sha256': scoped, 'status': status}
    drift = [k for k, v in report.items() if v['status'] in ('DRIFT', 'MISSING')]
    require(not drift, 'Reused source drifted from the native-run scope: ' + ','.join(drift))
    for module in modules:
        require(Path(module.__file__).resolve().is_relative_to(worktree / 'src'), 'Imported module is not the pinned worktree source: ' + module.__name__)
    return report


def adam_formula_pins(worktree, mlx_optimizer_source=None):
    """Read (never import) the Adam constructor and the installed MLX Adam source text."""
    rel, text = ADAM_CONSTRUCTOR_PIN
    path = Path(worktree) / rel
    constructor = {'path': rel, 'sha256': sha256(path), 'present': text in path.read_text()}
    require(constructor['present'], 'Pinned Adam constructor not found in the native source')
    if mlx_optimizer_source is None or not Path(mlx_optimizer_source).is_file():
        installed = {'status': 'NOT_VERIFIED', 'reason': 'installed MLX optimizer source text not supplied'}
    else:
        body = Path(mlx_optimizer_source).read_text()
        hits = {pin: pin in body for pin in MLX_ADAM_SOURCE_PINS}
        installed = {'status': 'MATCH' if all(hits.values()) else 'MISMATCH', 'sha256': sha256(mlx_optimizer_source), 'pins': hits,
                     'read_as_text_only': True}
    return {'constructor': constructor, 'installed_mlx_adam': installed, 'profile': ADAM_PROFILE}


# ================================================================ reward / advantage semantics
def reward_parser_label(text):
    """The training reward's own parser (format 0 -> INVALID_FORMAT). Strict DEV parsing uses core.strict_label instead."""
    answer, fmt = _answer_and_format(text)
    if fmt == 0:
        return 'INVALID_FORMAT'
    low = answer.strip().lower()
    for label in LABELS:
        if low == label.lower():
            return label
    return 'OTHER_ANSWER'


def advantage_components(grades, epsilon=1e-6):
    """Exact linear split of the unchanged group-relative advantage into accuracy/format/overlong parts.

    A_i = (r_i-mean r)/(sd+eps) with r = .8*acc + .2*format + overlong (identity checked per response).
    """
    rewards = [float(g['total']) for g in grades]
    advantages = advantages_from_rewards(rewards, epsilon)
    n = len(rewards)
    mean = sum(rewards) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in rewards) / (n - 1))
    denom = sd + epsilon
    means = {k: sum(float(g[k]) for g in grades) / n for k in ('accuracy', 'format', 'overlong')}
    out = []
    for g, a in zip(grades, advantages):
        identity_ok = abs(float(g['total']) - (ACCURACY_WEIGHT * g['accuracy'] + FORMAT_WEIGHT * g['format'] + g['overlong'])) <= TOLERANCES['reward_absolute']
        acc = ACCURACY_WEIGHT * (g['accuracy'] - means['accuracy']) / denom
        fmt = FORMAT_WEIGHT * (g['format'] - means['format']) / denom
        ovl = (g['overlong'] - means['overlong']) / denom
        out.append({'advantage': a, 'accuracy_part': acc if identity_ok else None, 'format_part': fmt if identity_ok else None,
                    'overlong_part': ovl if identity_ok else None, 'reward_identity_holds': identity_ok,
                    'split_residual': (a - acc - fmt - ovl) if identity_ok else None})
    return out


def group_contrast(members):
    """members: [{'context','raw_accuracy','accuracy','format','total'}]. Which reward differences the group can contrast.

    Judged on FORMAT-VALID responses only (an invalid response has raw accuracy 0 by construction, so validity must not
    masquerade as a class contrast):
      WITHIN_IMAGE_CLASS   some single image has both correct and wrong valid answers (genuine image-conditioned contrast);
      BETWEEN_IMAGE_ONLY   valid answers differ in correctness only ACROSS images (paired groups: prompt/class difficulty);
      FORMAT_OR_VALIDITY_ONLY  no correctness difference among valid answers; reward varies through format/validity only.
    """
    totals = [m['total'] for m in members]
    fmt_vary = len({m['format'] for m in members}) > 1
    valid = {}
    for m in members:
        if m['format'] == 1:
            valid.setdefault(m['context'], set()).add(m['raw_accuracy'])
    within = any(len(v) > 1 for v in valid.values())
    between = len(set().union(*valid.values())) > 1 if valid else False
    suffix = '_WITH_FORMAT' if fmt_vary else ''
    if max(totals) - min(totals) <= ZERO:
        return 'NO_REWARD_VARIANCE'
    if within:
        return 'WITHIN_IMAGE_CLASS' + suffix
    if between:
        return 'BETWEEN_IMAGE_ONLY' + suffix
    return 'FORMAT_OR_VALIDITY_ONLY'


def within_image_flags(members):
    """Both readings of 'within-image correctness contrast', reported separately (no single definition is privileged):
      all_response_within  some image has raw-correct and raw-wrong responses counting INVALID-format ones as wrong;
      format_valid_within  some image has correct and wrong answers among FORMAT-VALID responses only.
    BETWEEN-image-only contrast (paired groups) is the intended Paired-Prompt mechanism and is not flagged as invalid here."""
    def within(rows):
        by = {}
        for m in rows:
            by.setdefault(m['context'], set()).add(m['raw_accuracy'])
        return any(len(v) > 1 for v in by.values())
    return {'all_response_within': within(members), 'format_valid_within': within([m for m in members if m['format'] == 1])}


def sign_class(raw_accuracy, advantage):
    if abs(advantage) <= ZERO:
        return 'ZERO'
    return ('CORRECT' if raw_accuracy == 1 else 'WRONG') + ('_POSITIVE' if advantage > 0 else '_NEGATIVE')


# ================================================================ bundles: rebuild (segment 0) / revalidate (segment 1)
def rebuild_common_blocks(units4, raws, *, parent_policy, reward_sha, parent_receipt):
    require(len(units4) == len(raws) == 4, 'Four common units required')
    return [ms.build_block(u, r, policy_sha=parent_policy, reward_policy_sha256=reward_sha, checkpoint_receipt_sha256=parent_receipt)
            for u, r in zip(units4, raws)]


def dallas_block_from_saved(unit, value, *, policy_sha, reward_sha):
    """Re-validate a saved Dallas block exactly as ms.build_block bound it (context identity, reward receipts, body hash)."""
    require(value.get('schema') == 'multistep-dallas-block-v1' and value.get('experiment') == ms.EXPERIMENT, 'Unknown Dallas block')
    require(identity({k: v for k, v in value.items() if k != 'bundle_sha256'}) == value.get('bundle_sha256'), 'Dallas block hash differs')
    require(value['policy_sha256'] == policy_sha, 'Dallas block policy differs from the generating policy')
    keys = [c['key'] for c in unit['contexts']]
    require(set(value['contexts']) == set(keys), 'Dallas contexts differ from the unit')
    contexts = {}
    for k, c in zip(keys, unit['contexts']):
        rec = value['contexts'][k]
        ctx = ms.DContext(c['row']['id'], identity({'dallas_row': c['row']['id']}), 'D', rec['prompt_text_sha256'], rec['image_sha256'],
                          policy_sha, identity(shared.POLICY), reward_sha)
        require(rec['context_sha256'] == ctx.sha256 and rec['row_id'] == c['row']['id'], 'Dallas context identity differs')
        contexts[k] = ctx
    order = [f'{k}_{i}' for k in keys for i in range(shared.GROUP_SIZE)]
    require([r['response_id'] for r in value['responses']] == order, 'Dallas response inventory/order differs')
    responses = []
    for item in value['responses']:
        ctx = contexts[item['context_key']]
        grade = shared._grade_dict(item['grade'])
        token_sha, logps_sha = identity(tuple(item['token_ids'])), identity(item['raw_selected_logps'])
        text_sha = hashlib.sha256(item['text'].encode()).hexdigest()
        require(item['reward_receipt_sha256'] == shared.reward_receipt(item['response_id'], ctx.sha256, token_sha, text_sha, logps_sha, grade),
                'Dallas reward receipt differs (mutated grade/text/tokens)')
        g = ppg.Grade(accuracy=grade['accuracy'], format=grade['format'], total=grade['total'], overlong=grade['overlong'], raw_accuracy=grade['raw_accuracy'])
        responses.append(ppg.Response(item['response_id'], ctx, tuple(item['token_ids']), tuple(float(v) for v in item['raw_selected_logps']), g,
                                      item['reward_receipt_sha256']))
    return ms.UnitBlock(ms.DALLAS_KIND, value, tuple(responses), value['bundle_sha256'], policy_sha, None)


def load_own_blocks(units4, saved, *, policy_sha, reward_sha, previous_receipt):
    blocks = []
    for unit, value in zip(units4, saved):
        if unit['kind'] == ms.SEQ_KIND:
            bundle = shared.load_bundle(value)
            require(value['parent'] == {'receipt_sha256': previous_receipt, 'adapter_sha256': policy_sha},
                    'Own-policy SEQ bundle parent differs from the segment-0 receipt/policy')
            for key, c in zip(('F', 'T'), unit['contexts']):
                ctx = value['contexts'][key]
                require(ctx['row_id'] == c['row']['id'] and ctx['reward_policy_sha256'] == reward_sha, 'SEQ bundle context differs from the scheduled row/reward')
            blocks.append(ms.UnitBlock(ms.SEQ_KIND, value, bundle.responses, bundle.sha256, policy_sha, bundle))
        else:
            blocks.append(dallas_block_from_saved(unit, value, policy_sha=policy_sha, reward_sha=reward_sha))
    return blocks


# ================================================================ plan re-derivation
def _image_class(key):
    return {'F': 'FLAIR_IMAGE', 'T': 'T1_IMAGE'}.get(key, 'DALLAS')


def plan_records(block, unit, arm, *, grader=None):
    """Recompute gate/advantages/weights with the unchanged planner and describe every response and group."""
    planned = ms.arm_plan_for(block, arm)
    raw = {r['response_id']: r for r in block.raw['responses']}
    rows = {c['key']: c['row'] for c in unit['contexts']}
    task = 'MRI' if unit['kind'] == ms.SEQ_KIND else 'DALLAS'
    groups = {}
    for item in planned.plan.items:
        groups.setdefault(item.group_index, []).append(item)
    responses, group_records = [], []
    for gi in sorted(groups):
        members = groups[gi]
        grades = [asdict(m.response.grade) for m in members]
        comps = advantage_components(grades)
        keyed = []
        for m, comp in zip(members, comps):
            rid = m.response.response_id
            key = raw[rid].get('context_key') or rid[0]
            keyed.append({'context': key, 'raw_accuracy': m.response.grade.raw_accuracy, 'accuracy': m.response.grade.accuracy,
                          'format': m.response.grade.format, 'total': m.response.grade.total})
            require(abs(comp['advantage'] - m.advantage) <= TOLERANCES['advantage_absolute'], 'Planner advantage differs from the component recomputation')
            text = raw[rid]['text']
            record = {'unit': unit['index'], 'task': task, 'response_id': rid, 'context_key': key, 'image_class': _image_class(key), 'group': gi,
                      'admitted': bool(m.admitted), 'weight': m.weight, 'joint_weight': UNIT_WEIGHT * m.weight, 'advantage': m.advantage,
                      'advantage_accuracy_part': comp['accuracy_part'], 'advantage_format_part': comp['format_part'],
                      'advantage_overlong_part': comp['overlong_part'], 'reward_identity_holds': comp['reward_identity_holds'],
                      'reward_total': m.response.grade.total, 'calibrated_accuracy': m.response.grade.accuracy, 'format': m.response.grade.format,
                      'raw_accuracy': m.response.grade.raw_accuracy, 'overlong': m.response.grade.overlong, 'tokens': len(m.response.token_ids),
                      'length_cap_hit': len(m.response.token_ids) >= core.MAX_TOKENS,
                      'predicted': reward_parser_label(text) if task == 'MRI' else None,
                      'sign_class': sign_class(m.response.grade.raw_accuracy, m.advantage)}
            if grader is not None:
                record['regrade'] = regrade_one(grader, rows[key], text, len(m.response.token_ids), raw[rid]['grade'])
            responses.append(record)
        valid = [k for k in keyed if k['format'] == 1]
        group_records.append({'unit': unit['index'], 'task': task, 'group': gi, 'members': [m.response.response_id for m in members],
                              'contexts': sorted({k['context'] for k in keyed}), 'admitted': bool(members[0].admitted),
                              'contrast': group_contrast(keyed), **within_image_flags(keyed), 'rewards': [k['total'] for k in keyed],
                              'correct': sum(k['raw_accuracy'] == 1 for k in keyed), 'format_valid': len(valid),
                              'valid_subset_accuracy_values': sorted({k['accuracy'] for k in valid}),
                              'abs_advantage_accuracy_mass': sum(abs(c['accuracy_part'] or 0.) for c in comps),
                              'abs_advantage_format_mass': sum(abs(c['format_part'] or 0.) for c in comps)})
    return {'unit': unit['index'], 'kind': unit['kind'], 'task': task, 'grouping': planned.plan.grouping, 'effective_groups': planned.admitted_groups,
            'admitted_response_ids': list(planned.admitted_response_ids), 'responses': responses, 'groups': group_records}


def regrade_one(grader, row, text, tokens, saved_grade):
    grade = dict(grader(row, text, None))
    grade['overlong'] = overlong_penalty(tokens, core.MAX_TOKENS, 'direct_vero')
    grade['total'] = grade['score'] + grade['overlong']
    diffs = {k: abs(float(grade[k]) - float(saved_grade[k])) for k in ('accuracy', 'format', 'raw_accuracy', 'overlong', 'total')}
    return {'match': all(v <= TOLERANCES['reward_absolute'] for v in diffs.values()), 'max_abs_diff': max(diffs.values())}


def compare_with_saved_evidence(recomputed, saved):
    """Unit-level and per-response agreement between the re-derived plan and the native gradient_evidence record."""
    problems = []
    if recomputed['unit'] != saved['unit'] or recomputed['kind'] != saved['kind']:
        problems.append('unit/kind')
    if recomputed['effective_groups'] != saved['effective_groups']:
        problems.append('effective_groups')
    if sorted(recomputed['admitted_response_ids']) != sorted(saved['admitted_response_ids']):
        problems.append('admitted_response_ids')
    if recomputed['grouping'] != saved['grouping']:
        problems.append('grouping')
    saved_items = {i['response_id']: i for i in saved['items']}
    if set(saved_items) != {r['response_id'] for r in recomputed['responses']}:
        problems.append('response_inventory')
    max_adv = 0.
    for r in recomputed['responses']:
        s = saved_items.get(r['response_id'])
        if s is None:
            continue
        max_adv = max(max_adv, abs(r['advantage'] - s['advantage']))
        for mine, theirs in (('admitted', 'admitted'), ('weight', 'weight'), ('format', 'format'), ('raw_accuracy', 'raw_accuracy'),
                             ('tokens', 'tokens'), ('group', 'group')):
            if r[mine] != s[theirs]:
                problems.append(f"{r['response_id']}:{mine}")
        if abs(r['reward_total'] - s['reward']) > TOLERANCES['reward_absolute']:
            problems.append(f"{r['response_id']}:reward")
    if max_adv > TOLERANCES['advantage_absolute']:
        problems.append('advantage')
    return {'match': not problems, 'problems': problems, 'max_abs_advantage_diff': max_adv,
            'saved_unit_gradient_l2_native': saved.get('gradient_l2'), 'saved_weighted_loss_native': saved.get('weighted_loss')}


# ================================================================ allocation / counts
def _bucket(target, key, w, a):
    cell = target.setdefault(key, {'signed': 0., 'abs': 0., 'n': 0})
    cell['signed'] += w * a
    cell['abs'] += w * abs(a)
    cell['n'] += 1


def allocation_summary(records):
    """Descriptive SEQUENCE-LEVEL joint-weighted advantage allocation of ADMITTED responses (weight>0). Not a gradient."""
    by_task, by_image, by_cell, by_correct, mass = {}, {}, {}, {}, {}
    admitted = [r for r in records if r['weight'] > 0]
    for r in admitted:
        w, a = r['joint_weight'], r['advantage']
        _bucket(by_task, r['task'], w, a)
        _bucket(by_image, r['image_class'], w, a)
        _bucket(by_correct, r['task'] + ('_CORRECT' if r['raw_accuracy'] == 1 else '_WRONG'), w, a)
        if r['task'] == 'MRI':
            _bucket(by_cell, r['image_class'] + '|' + r['predicted'], w, a)
        m = mass.setdefault(r['task'], {'accuracy': 0., 'format': 0., 'overlong': 0., 'unsplit': 0.})
        if r['reward_identity_holds']:
            m['accuracy'] += w * abs(r['advantage_accuracy_part'])
            m['format'] += w * abs(r['advantage_format_part'])
            m['overlong'] += w * abs(r['advantage_overlong_part'])
        else:
            m['unsplit'] += w * abs(a)
    s = lambda key: by_cell.get(key, {}).get('signed', 0.)
    prior = (s('FLAIR_IMAGE|FLAIR') + s('T1_IMAGE|FLAIR')) - (s('FLAIR_IMAGE|contrast-enhanced T1') + s('T1_IMAGE|contrast-enhanced T1'))
    conditioned = (s('FLAIR_IMAGE|FLAIR') - s('FLAIR_IMAGE|contrast-enhanced T1')) - (s('T1_IMAGE|FLAIR') - s('T1_IMAGE|contrast-enhanced T1'))
    total_abs = sum(v['abs'] for v in by_task.values())
    return {'admitted_responses': len(admitted), 'by_task': by_task, 'by_image_class': by_image, 'by_image_and_reward_parser_answer': by_cell,
            'by_task_correctness': by_correct, 'abs_advantage_mass_by_component': mass,
            'MRI_share_of_abs_joint_weighted_advantage': (by_task.get('MRI', {}).get('abs', 0.) / total_abs) if total_abs > 0 else None,
            'sequence_level_push_toward_FLAIR_answers_regardless_of_image': prior,
            'sequence_level_image_conditioned_push': conditioned,
            'definition': 'sums of UNIT_WEIGHT*weight*advantage over admitted responses; "push" signs describe which complete sampled sequences '
                          'were up/down-weighted, NOT a parameter-space or class-margin derivative'}


def sign_counts(records):
    out = {}
    for scope, rows in (('admitted', [r for r in records if r['weight'] > 0]), ('all', records)):
        counts = {k: 0 for k in ('CORRECT_POSITIVE', 'CORRECT_NEGATIVE', 'WRONG_POSITIVE', 'WRONG_NEGATIVE', 'ZERO')}
        for r in rows:
            counts[r['sign_class']] += 1
        counts['WRONG_POSITIVE_FORMAT_DRIVEN'] = sum(r['sign_class'] == 'WRONG_POSITIVE' and r['reward_identity_holds'] and r['advantage_format_part'] > ZERO
                                                     and r['advantage_accuracy_part'] <= ZERO for r in rows)
        counts['CORRECT_NEGATIVE_FORMAT_DRIVEN'] = sum(r['sign_class'] == 'CORRECT_NEGATIVE' and r['reward_identity_holds'] and r['advantage_format_part'] < -ZERO
                                                       for r in rows)
        out[scope] = counts
    return out


def raw_reward_counterfactual(units):
    """SCALAR-ONLY counterfactual: plans, group membership, admission and weights held FIXED at the actual values; only the
    within-group advantage is recomputed from r_raw = .8*raw_accuracy + .2*format + overlong (calibrated accuracy replaced by raw).
    Describes advantage SIGNS/MASS of sampled sequences; it is NOT a gradient influence, NOT an accuracy effect and NOT a reward
    recommendation. Plan occurrences (one per group membership) and unique responses are counted separately."""
    items, unique = [], {}
    for u in units:
        groups = {}
        for r in u['responses']:
            groups.setdefault(r['group'], []).append(r)
        for members in groups.values():
            raw_adv = advantages_from_rewards([ACCURACY_WEIGHT * r['raw_accuracy'] + FORMAT_WEIGHT * r['format'] + r['overlong'] for r in members])
            for r, a in zip(members, raw_adv):
                items.append((r, a))
                unique.setdefault((u['unit'], r['response_id']), r)
    admitted = [(r, a) for r, a in items if r['weight'] > 0]
    out = {'plan_occurrences': len(items), 'admitted_plan_occurrences': len(admitted), 'unique_responses': len(unique),
           'unique_admitted_responses': len({(r['unit'], r['response_id']) for r, _ in admitted}), 'by_task': {}}
    for task in ('MRI', 'DALLAS'):
        rows = [(r, a) for r, a in admitted if r['task'] == task]
        flips = [(r, a) for r, a in rows if r['advantage'] * a < 0]
        mass = sum(r['joint_weight'] * abs(r['advantage']) for r, _ in rows)
        wrong_pos = sum(r['joint_weight'] * abs(r['advantage']) for r, _ in rows if r['raw_accuracy'] != 1 and r['advantage'] > ZERO)
        out['by_task'][task] = {
            'admitted_plan_occurrences': len(rows),
            'advantage_sign_changes_actual_to_raw': len(flips),
            'sign_changes_by_kind': {k: sum(1 for r, _ in flips if (('CORRECT' if r['raw_accuracy'] == 1 else 'WRONG') + ('_FORMAT_VALID' if r['format'] == 1 else '_FORMAT_INVALID')) == k)
                                     for k in ('CORRECT_FORMAT_VALID', 'CORRECT_FORMAT_INVALID', 'WRONG_FORMAT_VALID', 'WRONG_FORMAT_INVALID')},
            'raw_correct_positive_under_raw': sum(1 for r, a in rows if r['raw_accuracy'] == 1 and a > ZERO),
            'raw_correct_total': sum(1 for r, _ in rows if r['raw_accuracy'] == 1),
            'raw_wrong_negative_under_raw': sum(1 for r, a in rows if r['raw_accuracy'] != 1 and a < -ZERO),
            'raw_wrong_total': sum(1 for r, _ in rows if r['raw_accuracy'] != 1),
            'actual_wrong_positive_share_of_abs_joint_weighted_advantage_mass': _div(wrong_pos, mass)}
    out['label'] = 'SCALAR advantage counterfactual with fixed plans/admission/weights; not gradient influence or accuracy causation'
    return out


def segment_summary(units):
    """units: list of plan_records outputs for one arm segment."""
    records = [r for u in units for r in u['responses']]
    groups = [g for u in units for g in u['groups']]
    contrast, within = {}, {}
    for g in groups:
        c = contrast.setdefault(g['task'], {}).setdefault(g['contrast'], {'groups': 0, 'admitted': 0})
        c['groups'] += 1
        c['admitted'] += int(g['admitted'])
        w = within.setdefault(g['task'], {'admitted_groups': 0, 'admitted_all_response_within': 0, 'admitted_format_valid_within': 0})
        if g['admitted']:
            w['admitted_groups'] += 1
            w['admitted_all_response_within'] += int(g['all_response_within'])
            w['admitted_format_valid_within'] += int(g['format_valid_within'])
    sampled = {}
    for r in records:
        s = sampled.setdefault(r['task'] + ':' + r['image_class'], {'responses': 0, 'correct': 0, 'format_invalid': 0, 'length_cap': 0, 'answers': {}})
        s['responses'] += 1
        s['correct'] += int(r['raw_accuracy'] == 1)
        s['format_invalid'] += int(r['format'] == 0)
        s['length_cap'] += int(r['length_cap_hit'])
        if r['predicted'] is not None:
            s['answers'][r['predicted']] = s['answers'].get(r['predicted'], 0) + 1
    return {'responses': len(records), 'admitted_groups_by_task': {t: sum(g['admitted'] for g in groups if g['task'] == t) for t in ('MRI', 'DALLAS')},
            'groups_by_task': {t: sum(g['task'] == t for g in groups) for t in ('MRI', 'DALLAS')},
            'contrast_types': contrast, 'within_image_contrast_counts': within, 'raw_reward_counterfactual': raw_reward_counterfactual(units),
            'sampled_outcomes': sampled, 'sign_counts': sign_counts(records), 'allocation': allocation_summary(records),
            'regrade': None if not any('regrade' in r for r in records) else
            {'checked': sum('regrade' in r for r in records), 'mismatches': sum(('regrade' in r) and not r['regrade']['match'] for r in records)}}


def shared_segment0_comparison(plans):
    """A vs B on the IDENTICAL 32 shared parent responses (only grouping differs)."""
    a = {(u['unit'], r['response_id']): r for u in plans['A'] for r in u['responses']}
    b = {(u['unit'], r['response_id']): r for u in plans['B'] for r in u['responses']}
    require(set(a) == set(b), 'Segment-0 response inventories differ between arms')
    both = only_a = only_b = flips = 0
    rows = []
    for key in sorted(a):
        ra, rb = a[key], b[key]
        ia, ib = ra['weight'] > 0, rb['weight'] > 0
        both += ia and ib
        only_a += ia and not ib
        only_b += ib and not ia
        flip = ia and ib and ra['advantage'] * rb['advantage'] < 0
        flips += flip
        rows.append({'unit': key[0], 'response_id': key[1], 'task': ra['task'], 'image_class': ra['image_class'], 'raw_accuracy': ra['raw_accuracy'],
                     'format': ra['format'], 'A_admitted': ia, 'B_admitted': ib, 'A_advantage': ra['advantage'], 'B_advantage': rb['advantage']})
    return {'responses': len(rows), 'admitted_in_both': both, 'admitted_only_A': only_a, 'admitted_only_B': only_b,
            'advantage_sign_flips_among_admitted_in_both': flips, 'rows': rows,
            'Dallas_units_identical_plans': all(a[k]['advantage'] == b[k]['advantage'] and a[k]['weight'] == b[k]['weight'] for k in a if a[k]['task'] == 'DALLAS')}


# ================================================================ task-gradient inventory (no invention)
def task_gradient_inventory(run_dir, segments):
    """segments: {(arm, seg): {'units': plan_records list, 'saved_unit_l2': [...] }}. Lists retained gradient files and states what they identify."""
    run_dir = Path(run_dir)
    files = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob('*.safetensors') if 'gradient' in p.name)
    per_task = [f for f in files if re.search(r'(mri|dallas|task|unit)', Path(f).name, re.I) and 'joint' not in Path(f).name]
    report = {'gradient_files': files, 'per_task_or_per_unit_gradient_files': per_task, 'segments': {}}
    for (arm, seg), value in sorted(segments.items()):
        mri = sum(u['effective_groups'] for u in value['units'] if u['task'] == 'MRI')
        dal = sum(u['effective_groups'] for u in value['units'] if u['task'] == 'DALLAS')
        if mri == 0 and dal > 0:
            composition = 'DALLAS_ONLY_EXACT_BY_GATE'
        elif dal == 0 and mri > 0:
            composition = 'MRI_ONLY_EXACT_BY_GATE'
        elif mri == dal == 0:
            composition = 'EMPTY'
        else:
            composition = 'MIXED_NOT_SEPARABLE_FROM_JOINT'
        report['segments'][f'{arm}{seg}'] = {'joint_saved': True, 'per_task_saved': bool(per_task), 'MRI_admitted_groups': mri, 'Dallas_admitted_groups': dal,
                                              'joint_task_composition': composition, 'saved_native_unit_gradient_l2': value['saved_unit_l2'],
                                              'unit_tasks': [u['task'] for u in value['units']]}
    report['MRI_vs_Dallas_task_cosine'] = task_cosine(None)
    return report


def task_cosine(task_gradients):
    """MRI-vs-Dallas cosine ONLY from separately retained task gradients ({name: flat array}); never inferred from a joint sum."""
    missing = [t for t in ('MRI', 'DALLAS') if not task_gradients or task_gradients.get(t) is None]
    if missing:
        return {'status': 'NOT_VERIFIED', 'cosine': None,
                'reason': 'Task-specific gradient(s) not retained: ' + ','.join(missing) + '; a joint sum plus unit norms does not identify task vectors'}
    import numpy as np
    a = np.asarray(task_gradients['MRI'], dtype=np.float64).ravel()
    b = np.asarray(task_gradients['DALLAS'], dtype=np.float64).ravel()
    require(a.shape == b.shape, 'Task gradient shapes differ')
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {'status': 'MEASURED', 'cosine': float(a @ b) / (na * nb) if na > 0 and nb > 0 else None, 'MRI_l2': na, 'DALLAS_l2': nb}


def joint_norm_constraint(joint_l2, unit_l2, unit_tasks, weight=UNIT_WEIGHT):
    """J = weight*sum_u g_u. From ||J|| and every ||g_u||: sum_{u<v} <g_u,g_v> = (||J||^2/w^2 - sum ||g_u||^2)/2 exactly.
    A pairwise cosine is identified ONLY when exactly two units are nonzero."""
    nonzero = [(i, n, t) for i, (n, t) in enumerate(zip(unit_l2, unit_tasks)) if n > 0]
    cross = ((joint_l2 / weight) ** 2 - sum(n * n for _, n, _ in nonzero)) / 2
    result = {'nonzero_units': [i for i, _, _ in nonzero], 'pairwise_cross_inner_product_sum': cross, 'identifies_pairwise_cosine': len(nonzero) == 2}
    if len(nonzero) == 2:
        (i, a, ta), (j, b, tb) = nonzero
        result.update(unit_pair=[i, j], pair_tasks=[ta, tb], derived_cosine=cross / (a * b), kind='DERIVED_FROM_NORMS_EXACT_TWO_TERM_IDENTITY',
                      MRI_vs_Dallas=ta != tb)
    else:
        result['reason'] = 'more than two nonzero unit gradients: one equation, several unknown cross products; no task cosine identified'
    return result


# ================================================================ streaming tensors (safetensors numpy only)
class TensorFile:
    """Read-only access to ONE tensor per call via safetensors(framework='np'); no MLX/torch."""
    def __init__(self, path):
        from safetensors import safe_open
        self.path = Path(path)
        handle = safe_open(str(self.path), framework='np')
        enter = getattr(handle, '__enter__', None)
        self._handle = enter() if callable(enter) else handle
        self._keys = frozenset(self._handle.keys())

    def keys(self):
        return set(self._keys)

    def get(self, key, dtype='float32'):
        import numpy as np
        require(key in self._keys, f'{self.path.name}: tensor {key} absent')
        value = self._handle.get_tensor(key)
        require(str(value.dtype) == dtype, f'{self.path.name}:{key}: {dtype} required, found {value.dtype}')
        return np.array(value, copy=True)


def layout_index(layout):
    """Flat name -> tensor node of an mri-optimizer-boundary-v1 layout, without loading tensors."""
    out = {}

    def visit(node, path):
        kind = node['kind']
        if kind == 'tensor':
            require(path not in out, 'Duplicate layout path')
            out[path] = node
        elif kind == 'dict':
            for key, value in node['items']:
                visit(value, f'{path}.{key}' if path else key)
        elif kind in ('list', 'tuple'):
            for index, value in enumerate(node['items']):
                visit(value, f'{path}.{index}' if path else str(index))
        else:
            require(kind == 'scalar', 'Unknown layout node')
    visit(layout, '')
    return out


class Checkpoint:
    """Adapter + full Adam state of one saved boundary (hash-authenticated, streamed)."""
    def __init__(self, directory, *, label, adapter_sha256=None):
        d = Path(directory)
        self.label = label
        manifest = load_json(d / 'optimizer' / 'manifest.json')
        require(manifest.get('schema') == 'mri-optimizer-boundary-v1', label + ': unknown optimizer manifest')
        require_file_sha(d / 'optimizer' / 'state.safetensors', manifest['tensors_sha256'], label + ' optimizer state')
        if adapter_sha256 is not None:
            require_file_sha(d / 'adapter.safetensors', adapter_sha256, label + ' adapter')
        index = layout_index(manifest['layout'])
        require('optimizer.step' in index and 'optimizer.learning_rate' in index, label + ': step/learning_rate absent')
        self.step_key, self.lr_key = index['optimizer.step']['key'], index['optimizer.learning_rate']['key']
        moments = {}
        for path, node in index.items():
            if path.startswith('optimizer.') and path.rsplit('.', 1)[-1] in ('m', 'v'):
                moments.setdefault(path[len('optimizer.'):-2], {})[path[-1]] = node['key']
        require(moments and all(set(v) == {'m', 'v'} for v in moments.values()), label + ': incomplete Adam moments')
        self.moment_keys = moments
        self.adapter = TensorFile(d / 'adapter.safetensors')
        self.state = TensorFile(d / 'optimizer' / 'state.safetensors')
        require(self.adapter.keys() == set(moments), label + ': adapter/moment inventory differs')
        self.updates = manifest['optimizer_updates']
        require(self.step() == self.updates, label + ': saved step differs from manifest optimizer_updates')

    def names(self):
        return sorted(self.moment_keys)

    def step(self):
        return int(self.state.get(self.step_key, 'uint64'))

    def lr(self):
        return self.state.get(self.lr_key)

    def param(self, name):
        return self.adapter.get(name)

    def moments(self, name):
        k = self.moment_keys[name]
        return self.state.get(k['m']), self.state.get(k['v'])


def _locate(name):
    m = re.search(r'layers\.(\d+)\.(.+)\.(lora_[ab])$', name)
    return (int(m.group(1)), m.group(2)) if m else ('other', name)


class Sums:
    def __init__(self):
        self.total, self.by_layer, self.by_module = {}, {}, {}

    def add(self, name, **values):
        layer, module = _locate(name)
        for target in (self.total, self.by_layer.setdefault(layer, {}), self.by_module.setdefault(module, {})):
            for k, v in values.items():
                target[k] = target.get(k, 0.) + float(v)


class Check:
    def __init__(self, label):
        self.label, self.n, self.violations, self.bitwise_equal, self.max_abs, self.max_rel = label, 0, 0, 0, 0., 0.
        self.nonfinite = 0

    def add(self, pred, saved, bound):
        """An element PASSES only if pred, saved and bound are all finite, bound >= 0 and |pred-saved| <= bound.
        (A bare `err > bound` comparison is False for NaN error or NaN/Inf bound and would silently PASS.)"""
        import numpy as np
        p, s = np.asarray(pred).astype(np.float64), np.asarray(saved).astype(np.float64)
        b = np.broadcast_to(np.asarray(bound, dtype=np.float64), p.shape)
        err = np.abs(p - s)
        ok = np.isfinite(p) & np.isfinite(s) & np.isfinite(b) & (b >= 0) & (err <= b)
        self.n += err.size
        self.nonfinite += int((~(np.isfinite(p) & np.isfinite(s) & np.isfinite(b))).sum())
        self.violations += int((~ok).sum())
        self.bitwise_equal += int((pred == saved).sum())
        finite = np.isfinite(err)
        if finite.any():
            self.max_abs = max(self.max_abs, float(err[finite].max()))
            scale = np.maximum(np.abs(s[finite]), 1e-30)
            self.max_rel = max(self.max_rel, float((err[finite] / scale).max()))

    def report(self):
        return {'elements': self.n, 'violations': self.violations, 'nonfinite_elements': self.nonfinite, 'bitwise_equal': self.bitwise_equal,
                'max_abs_error': self.max_abs, 'max_rel_error': self.max_rel, 'PASS': self.violations == 0 and self.n > 0}


def _div(a, b):
    return None if b == 0 else a / b


def _cos(dot, aa, bb):
    return None if aa <= 0 or bb <= 0 else dot / math.sqrt(aa * bb)


def _derive_step(s):
    r = lambda k: math.sqrt(max(s.get(k, 0.), 0.))
    dd = s.get('dd', 0.)
    return {'delta_l2': r('dd'), 'joint_gradient_l2': r('gg'), 'relative_change_l2': _div(r('dd'), r('th0')),
            'cos_descent_vs_actual_delta': _cos(-s.get('gd', 0.), s.get('gg', 0.), dd),
            'first_order_surrogate_change_g_dot_delta': s.get('gd'),
            'momentum_only_g0_delta_l2': r('mm'), 'cos_actual_delta_vs_momentum_only': _cos(s.get('dm', 0.), dd, s.get('mm', 0.)),
            'cos_descent_vs_momentum_only': _cos(-s.get('gm', 0.), s.get('gg', 0.), s.get('mm', 0.)),
            'increment_l2': r('ii'), 'increment_over_delta_l2': _div(r('ii'), r('dd')),
            'cos_descent_vs_increment': _cos(-s.get('gi', 0.), s.get('gg', 0.), s.get('ii', 0.)), 'g_dot_increment': s.get('gi'),
            'actual_denominator_split': {'momentum_share_along_delta': _div(s.get('cmd', 0.), dd), 'gradient_share_along_delta': _div(s.get('cgd', 0.), dd),
                                         'momentum_term_l2_over_delta_l2': _div(r('cmcm'), r('dd')), 'gradient_term_l2_over_delta_l2': _div(r('cgcg'), r('dd')),
                                         'cos_gradient_term_vs_delta': _cos(s.get('cgd', 0.), s.get('cgcg', 0.), dd),
                                         'cos_momentum_term_vs_gradient_term': _cos(s.get('cmcg', 0.), s.get('cmcm', 0.), s.get('cgcg', 0.)),
                                         'residual_l2_over_delta_l2': _div(r('res'), r('dd'))},
            'previous_momentum_l2': r('m0m0'), 'cos_previous_momentum_vs_gradient': _cos(s.get('m0g', 0.), s.get('m0m0', 0.), s.get('gg', 0.)),
            'sign_agreement_delta_vs_descent_abs_mass_fraction': _div(s.get('agree', 0.), s.get('absd', 0.))}


def _derive_group(s, total_dd):
    return {'delta_l2': math.sqrt(s.get('dd', 0.)), 'share_of_total_delta_sq': _div(s.get('dd', 0.), total_dd),
            'cos_descent_vs_actual_delta': _cos(-s.get('gd', 0.), s.get('gg', 0.), s.get('dd', 0.)),
            'momentum_share_along_delta': _div(s.get('cmd', 0.), s.get('dd', 0.)), 'gradient_share_along_delta': _div(s.get('cgd', 0.), s.get('dd', 0.)),
            'cos_actual_delta_vs_momentum_only': _cos(s.get('dm', 0.), s.get('dd', 0.), s.get('mm', 0.))}


def adam_step_report(prev, nxt, grad, *, profile=ADAM_PROFILE, tolerances=TOLERANCES):
    """Recompute ONE saved optimizer step on CPU and decompose the actual parameter delta. Never calls an optimizer."""
    import numpy as np
    names = prev.names()
    require(nxt.names() == names and grad.keys() == set(names), 'Parameter/moment/gradient inventories differ')
    s0, s1 = prev.step(), nxt.step()
    require(s1 == s0 + 1, 'Exactly one optimizer step between checkpoints required')
    lr32 = prev.lr()
    require(lr32.tobytes() == nxt.lr().tobytes(), 'Learning rate changed across the step')
    b1, b2, eps = profile['beta1'], profile['beta2'], profile['eps']
    f = np.float32
    B1, C1, B2, C2, EPS = f(b1), f(1 - b1), f(b2), f(1 - b2), f(eps)
    lr = float(lr32)
    mt, pt = tolerances['moment_elementwise'], tolerances['parameter_elementwise']
    checks = {'m': Check('m'), 'v': Check('v'), 'theta': Check('theta')}
    sums = Sums()
    for name in names:
        th0, th1, g = prev.param(name), nxt.param(name), grad.get(name)
        m0, v0 = prev.moments(name)
        m1, v1 = nxt.moments(name)
        require(th0.shape == th1.shape == g.shape == m0.shape == v0.shape == m1.shape == v1.shape, name + ': shape mismatch')
        require(all(bool(np.isfinite(x).all()) for x in (th0, th1, g, m0, v0, m1, v1)), name + ': nonfinite saved tensor')
        require(bool((v0 >= 0).all()) and bool((v1 >= 0).all()), name + ': negative saved Adam second moment (sqrt undefined)')
        m_pred = B1 * m0 + C1 * g
        v_pred = B2 * v0 + C2 * np.square(g)
        for key, pred, saved in (('m', m_pred, m1), ('v', v_pred, v1)):
            bound = mt['relative'] * np.maximum(np.abs(pred.astype(np.float64)), np.abs(saved.astype(np.float64))) + mt['absolute']
            checks[key].add(pred, saved, bound)
        u32 = (lr32 * m1) / (np.sqrt(v1) + EPS)
        th_pred = th0 - u32
        bound = pt['spacing_multiple'] * np.spacing(np.abs(th1)).astype(np.float64) + pt['update_relative'] * np.abs(u32.astype(np.float64))
        checks['theta'].add(th_pred, th1, bound)
        G, D = g.astype(np.float64), th1.astype(np.float64) - th0.astype(np.float64)
        M0, V0, M1, V1 = (x.astype(np.float64) for x in (m0, v0, m1, v1))
        den1 = np.sqrt(V1) + eps
        cm, cg = -lr * b1 * M0 / den1, -lr * (1 - b1) * G / den1
        mom = -lr * b1 * M0 / (np.sqrt(b2 * V0) + eps)
        inc = D - mom
        e1, e2 = M1 - G, M0 - G
        q1, q2 = V1 - G * G, V0 - G * G
        sums.add(name, dd=np.vdot(D, D), gg=np.vdot(G, G), gd=np.vdot(G, D), mm=np.vdot(mom, mom), dm=np.vdot(D, mom),
                 gm=np.vdot(G, mom), ii=np.vdot(inc, inc), gi=np.vdot(G, inc), cmcm=np.vdot(cm, cm), cgcg=np.vdot(cg, cg), cmd=np.vdot(cm, D),
                 cgd=np.vdot(cg, D), cmcg=np.vdot(cm, cg), res=np.vdot(cm + cg - D, cm + cg - D), m0m0=np.vdot(M0, M0), m0g=np.vdot(M0, G),
                 th0=np.vdot(th0.astype(np.float64), th0.astype(np.float64)),
                 agree=float(np.abs(D)[np.sign(D) == -np.sign(G)].sum()), absd=float(np.abs(D).sum()),
                 b1n=np.vdot(e1, e2), b1d=np.vdot(e2, e2), b1e=np.vdot(e1, e1), b2n=np.vdot(q1, q2), b2d=np.vdot(q2, q2), b2e=np.vdot(q1, q1))
    t = sums.total
    fit = {}
    for k, nominal in (('b1', b1), ('b2', b2)):
        beta = _div(t[k + 'n'], t[k + 'd'])
        resid = None if beta is None else math.sqrt(max(t[k + 'e'] - 2 * beta * t[k + 'n'] + beta * beta * t[k + 'd'], 0.)) / max(math.sqrt(t[k + 'e']), 1e-300)
        fit[k] = {'fitted': beta, 'nominal': nominal, 'abs_diff': None if beta is None else abs(beta - nominal),
                  'relative_fit_residual': resid, 'PASS': beta is not None and abs(beta - nominal) <= tolerances['beta_fit_absolute']}
    metrics = _derive_step(t)
    return {'step_before': s0, 'step_after': s1, 'learning_rate_float32': lr, 'parameters': len(names),
            'formula_checks': {k: v.report() for k, v in checks.items()},
            'beta_fit_from_saved_moments_and_saved_joint_gradient': fit,
            'saved_gradient_is_the_applied_gradient': all(v['PASS'] for v in fit.values()) and checks['m'].violations == 0 and checks['v'].violations == 0,
            'metrics': metrics,
            'by_layer': {str(k): _derive_group(v, t['dd']) for k, v in sorted(sums.by_layer.items(), key=lambda kv: (str(type(kv[0])), kv[0]))},
            'by_module': {k: _derive_group(v, t['dd']) for k, v in sorted(sums.by_module.items())},
            'label': 'JOINT (MRI+Dallas) quantities; not MRI class-margin derivatives'}


def two_step_report(parent, mid, final, g0, g1, *, profile=ADAM_PROFILE):
    """Exact linear split of theta54-theta52 into parent-history momentum (m52), first and second joint gradients,
    using the SAVED denominators D53=sqrt(v53)+eps and D54=sqrt(v54)+eps."""
    import numpy as np
    names = parent.names()
    require(mid.names() == names == final.names() and g0.keys() == set(names) == g1.keys(), 'Inventories differ')
    require(mid.step() == parent.step() + 1 and final.step() == mid.step() + 1, 'Two consecutive steps required')
    lr, b1, eps = float(parent.lr()), profile['beta1'], profile['eps']
    sums = Sums()
    for name in names:
        th0, th1, th2 = (c.param(name).astype(np.float64) for c in (parent, mid, final))
        m52 = parent.moments(name)[0].astype(np.float64)
        v53, v54 = mid.moments(name)[1].astype(np.float64), final.moments(name)[1].astype(np.float64)
        G0, G1 = g0.get(name).astype(np.float64), g1.get(name).astype(np.float64)
        require(all(bool(np.isfinite(x).all()) for x in (th0, th1, th2, m52, v53, v54, G0, G1)), name + ': nonfinite saved tensor')
        require(bool((v53 >= 0).all()) and bool((v54 >= 0).all()), name + ': negative saved Adam second moment (sqrt undefined)')
        d53, d54 = np.sqrt(v53) + eps, np.sqrt(v54) + eps
        k = 1 / d53 + b1 / d54
        ch, c0, c1 = -lr * b1 * m52 * k, -lr * (1 - b1) * G0 * k, -lr * (1 - b1) * G1 / d54
        D, D1, D2 = th2 - th0, th1 - th0, th2 - th1
        sums.add(name, dd=np.vdot(D, D), chch=np.vdot(ch, ch), c0c0=np.vdot(c0, c0), c1c1=np.vdot(c1, c1), chd=np.vdot(ch, D), c0d=np.vdot(c0, D),
                 c1d=np.vdot(c1, D), chc0=np.vdot(ch, c0), chc1=np.vdot(ch, c1), c0c1=np.vdot(c0, c1), res=np.vdot(ch + c0 + c1 - D, ch + c0 + c1 - D),
                 d1d1=np.vdot(D1, D1), d2d2=np.vdot(D2, D2), d1d2=np.vdot(D1, D2))
    t = sums.total
    dd = t['dd']
    share = {k: _div(t[k + 'd'], dd) for k in ('ch', 'c0', 'c1')}
    return {'total_delta_l2': math.sqrt(dd), 'step1_delta_l2': math.sqrt(t['d1d1']), 'step2_delta_l2': math.sqrt(t['d2d2']),
            'cos_step1_vs_step2_delta': _cos(t['d1d2'], t['d1d1'], t['d2d2']),
            'shares_along_total_delta': {'parent_history_momentum_m52': share['ch'], 'segment0_joint_gradient': share['c0'], 'segment1_joint_gradient': share['c1']},
            'term_l2_over_total_l2': {'parent_history_momentum_m52': math.sqrt(t['chch'] / dd), 'segment0_joint_gradient': math.sqrt(t['c0c0'] / dd),
                                      'segment1_joint_gradient': math.sqrt(t['c1c1'] / dd)},
            'cos_term_vs_total': {'parent_history_momentum_m52': _cos(t['chd'], t['chch'], dd), 'segment0_joint_gradient': _cos(t['c0d'], t['c0c0'], dd),
                                  'segment1_joint_gradient': _cos(t['c1d'], t['c1c1'], dd)},
            'cos_between_terms': {'m52_vs_g0': _cos(t['chc0'], t['chch'], t['c0c0']), 'm52_vs_g1': _cos(t['chc1'], t['chch'], t['c1c1']),
                                  'g0_vs_g1': _cos(t['c0c1'], t['c0c0'], t['c1c1'])},
            'residual_l2_over_total_l2': math.sqrt(t['res'] / dd),
            'by_layer_total_delta_l2': {str(k): math.sqrt(v['dd']) for k, v in sorted(sums.by_layer.items(), key=lambda kv: (str(type(kv[0])), kv[0]))},
            'residual_within_declared_tolerance': math.sqrt(t['res'] / dd) <= TOLERANCES['decomposition_residual_relative'],
            'label': ('Algebraic split of two no-bias-correction Adam steps using the SAVED v53/v54 denominators, evaluated in float64 on float32 '
                      'saved tensors: holds up to float32 rounding (see residual), NOT a bitwise-exact decomposition. Shares are signed projections '
                      'onto the total delta of JOINT terms: NOT causal MRI-damage fractions and NOT norm shares')}


CROSS_VECTORS = ('dA53', 'dA54', 'dAtot', 'dB53', 'dB54', 'dBtot', 'thetaA54_minus_thetaB54', 'gA0', 'gA1', 'gB0', 'gB1', 'gB0_minus_gA0', 'm52')


def cross_arm_report(parent, A53, A54, B53, B54, gA0, gA1, gB0, gB1):
    """Full Gram matrix of parameter deltas, joint gradients and the inherited Adam52 momentum across arms."""
    import numpy as np
    names = parent.names()
    for c in (A53, A54, B53, B54):
        require(c.names() == names, 'Checkpoint inventories differ')
    for g in (gA0, gA1, gB0, gB1):
        require(g.keys() == set(names), 'Gradient inventories differ')
    gram = {}
    for name in names:
        th = {k: c.param(name).astype(np.float64) for k, c in (('P', parent), ('A53', A53), ('A54', A54), ('B53', B53), ('B54', B54))}
        g = {k: f.get(name).astype(np.float64) for k, f in (('gA0', gA0), ('gA1', gA1), ('gB0', gB0), ('gB1', gB1))}
        v = {'dA53': th['A53'] - th['P'], 'dA54': th['A54'] - th['A53'], 'dAtot': th['A54'] - th['P'], 'dB53': th['B53'] - th['P'],
             'dB54': th['B54'] - th['B53'], 'dBtot': th['B54'] - th['P'], 'thetaA54_minus_thetaB54': th['A54'] - th['B54'], **g,
             'gB0_minus_gA0': g['gB0'] - g['gA0'], 'm52': parent.moments(name)[0].astype(np.float64)}
        for i, a in enumerate(CROSS_VECTORS):
            for b in CROSS_VECTORS[i:]:
                gram[(a, b)] = gram.get((a, b), 0.) + float(np.vdot(v[a], v[b]))
    norm = {k: math.sqrt(gram[(k, k)]) for k in CROSS_VECTORS}
    cos = {f'{a}|{b}': _cos(gram[(a, b)], gram[(a, a)], gram[(b, b)]) for i, a in enumerate(CROSS_VECTORS) for b in CROSS_VECTORS[i + 1:]}
    return {'l2': norm, 'cosine': cos,
            'final_A_minus_B_over_total_A_l2': _div(norm['thetaA54_minus_thetaB54'], norm['dAtot']),
            'final_A_minus_B_over_total_B_l2': _div(norm['thetaA54_minus_thetaB54'], norm['dBtot']),
            'gB0_minus_gA0_note': 'Segment-0 Dallas units are the same bundles, grouping and parent in both arms; if their native gradients are '
                                  'deterministic, gB0-gA0 = 0.25*(MRI unit differences) only (DERIVED_UNDER_DETERMINISM_ASSUMPTION)',
            'label': 'JOINT gradients/updates; parent momentum m52 is the inherited Adam52 first moment'}


def lowrank_inner(terms_x, terms_y):
    """<sum c_i a_i b_i, sum d_j a_j b_j>_F using only r x r Gram blocks: tr((a_i^T a_j)(b_j b_i^T))."""
    total = 0.
    for c, a, b in terms_x:
        for d, a2, b2 in terms_y:
            total += c * d * float(((a.T @ a2) * (b2 @ b.T).T).sum())
    return total


LORA_PAIRS = (('P', 'A53'), ('A53', 'A54'), ('P', 'A54'), ('P', 'B53'), ('B53', 'B54'), ('P', 'B54'), ('B54', 'A54'))


def lora_effective_report(checkpoints, scale):
    """Effective LoRA weight change ||scale*(a'b'-ab)||_F per module, layer and type (r x r Gram identities, streamed per module)."""
    import numpy as np
    names = checkpoints['P'].names()
    modules = sorted({n[:-len('.lora_a')] for n in names if n.endswith('.lora_a')})
    require(all(m + '.lora_b' in names for m in modules) and len(names) == 2 * len(modules), 'Each LoRA module needs exactly lora_a/lora_b')
    total, by_layer, by_type = {}, {}, {}
    for mod in modules:
        f = {k: (c.param(mod + '.lora_a').astype(np.float64), c.param(mod + '.lora_b').astype(np.float64)) for k, c in checkpoints.items()}
        delta = {f'{p}->{q}': [(1., *f[q]), (-1., *f[p])] for p, q in LORA_PAIRS}
        values = {k: lowrank_inner(t, t) for k, t in delta.items()}
        values['W_parent'] = lowrank_inner([(1., *f['P'])], [(1., *f['P'])])
        values['dA_tot_dot_dB_tot'] = lowrank_inner(delta['P->A54'], delta['P->B54'])
        layer, kind = _locate(mod + '.lora_a')
        for target in (total, by_layer.setdefault(str(layer), {}), by_type.setdefault(kind, {})):
            for k, v in values.items():
                target[k] = target.get(k, 0.) + v
    def out(s):
        r = {k: scale * math.sqrt(max(s[k], 0.)) for k in s if k != 'dA_tot_dot_dB_tot'}
        r['cos_effective_dA_tot_vs_dB_tot'] = _cos(s['dA_tot_dot_dB_tot'], s['P->A54'], s['P->B54'])
        r['relative_P->A54'] = _div(math.sqrt(s['P->A54']), math.sqrt(s['W_parent']))
        r['relative_P->B54'] = _div(math.sqrt(s['P->B54']), math.sqrt(s['W_parent']))
        return r
    return {'scale': scale, 'modules': len(modules), 'total': out(total), 'by_layer': {k: out(v) for k, v in by_layer.items()},
            'by_type': {k: out(v) for k, v in by_type.items()}, 'label': 'Frobenius norms of scale*a@b differences; vision tower frozen (no LoRA)'}


# ================================================================ DEV greedy-path evidence
def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + list(range(ord('\xae'), ord('\xff') + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class TokenDecoder:
    """Byte-level BPE id->bytes from vocab.json (+ tokenizer.json added tokens) read as JSON. No tokenizer library, no model."""
    def __init__(self, vocab, added=()):
        self.inverse = {i: t for t, i in vocab.items()}
        self.added = {a['id']: a['content'] for a in added}
        self.byte_of = bytes_to_unicode()

    @classmethod
    def from_files(cls, vocab_json, tokenizer_json=None):
        added = load_json(tokenizer_json).get('added_tokens', []) if tokenizer_json else []
        return cls(load_json(vocab_json), added)

    def piece(self, token_id):
        if token_id in self.added:
            return self.added[token_id].encode()
        text = self.inverse.get(token_id)
        require(text is not None, f'Token id {token_id} absent from vocabulary')
        try:
            return bytes(self.byte_of[ch] for ch in text)
        except KeyError:                       # not byte-level encoded: keep literal UTF-8 (reported via decoded_text_check)
            return text.encode()


def decision_token_index(token_ids, decoder, marker=b'\\boxed{'):
    """Index of the token holding the first label byte after the LAST '\\boxed{'; None if absent."""
    if decoder is None:
        return None
    spans, cursor = [], 0
    for t in token_ids:
        p = decoder.piece(t)
        spans.append((cursor, cursor + len(p)))
        cursor += len(p)
    full = b''.join(decoder.piece(t) for t in token_ids)
    at = full.rfind(marker)
    if at < 0:
        return None
    first = at + len(marker)
    for i, (a, b) in enumerate(spans):
        if a <= first < b:
            return i
    return None


def selected_logp_comparison(ref, child, decoder=None):
    """Greedy-path comparison of saved selected-token log-probs. NOT a class margin (no alternative-class logit saved)."""
    t0, l0 = ref['token_ids'], ref['raw_selected_logps']
    t1, l1 = child['token_ids'], child['raw_selected_logps']
    require(len(t0) == len(l0) and len(t1) == len(l1), 'Token/logp lengths differ')
    n = 0
    while n < min(len(t0), len(t1)) and t0[n] == t1[n]:
        n += 1
    deltas = [b - a for a, b in zip(l0[:n], l1[:n])]
    k0, k1 = decision_token_index(t0, decoder), decision_token_index(t1, decoder)
    out = {'identical_token_sequence': t0 == t1, 'common_prefix_tokens': n, 'first_divergence_position': None if t0 == t1 else n,
           'ref_tokens': len(t0), 'child_tokens': len(t1), 'prefix_logp_sum_delta_child_minus_ref': sum(deltas),
           'prefix_max_token_logp_drop': min(deltas) if deltas else None, 'ref_min_selected_logp': min(l0), 'child_min_selected_logp': min(l1),
           'ref_decision_token_index': k0, 'child_decision_token_index': k1,
           'label': 'GREEDY_PATH_SELECTED_TOKEN_LOGPROB_NOT_CLASS_MARGIN'}
    if k0 is not None:
        out['ref_decision_token_logp'] = l0[k0]
        out['ref_decision_token_prob'] = math.exp(l0[k0])
        out['divergence_relative_to_decision'] = ('IDENTICAL' if t0 == t1 else 'BEFORE_DECISION' if n < k0 else 'AT_DECISION' if n == k0 else 'AFTER_DECISION')
        if n > k0:
            out['child_logp_of_same_decision_token_same_prefix'] = l1[k0]
            out['decision_token_logp_delta_child_minus_ref'] = l1[k0] - l0[k0]
    if k1 is not None:
        out['child_decision_token_logp'] = l1[k1]
        out['child_decision_token_prob'] = math.exp(l1[k1])
    return out


def dev_case_analysis(records, rows, decoder=None):
    """records: frozen DEV24; rows: {'Parent'|'A'|'B': [24 verified rows]}. Pseudonymous case/pair indices only."""
    pairs = core.dev_pairs(records)
    partner = {}
    for p, (f, t) in enumerate(pairs):
        partner[f], partner[t] = (p, t), (p, f)
    truth = [LABELS.index(r['row_PRIVATE']['reward_model']['ground_truth']) for r in records]
    preds = {m: [core.strict_label(r['text']) for r in rows[m]] for m in rows}
    cases = []
    for i in range(len(records)):
        p, j = partner[i]
        pf, pt = pairs[p]
        resp = {m: rows[m][i]['actual_PRIVATE']['response'] for m in rows}
        case = {'case': i, 'pair': p, 'modality': 'FLAIR' if truth[i] == 0 else 'T1', 'partner_case': j,
                'correct': {m: preds[m][i] == truth[i] for m in rows}, 'invalid': {m: preds[m][i] < 0 for m in rows},
                'partner_correct': {m: preds[m][j] == truth[j] for m in rows},
                'pair_category': {m: core.pair_category(preds[m][pf], preds[m][pt]) for m in rows},
                'text_identical': {'A_vs_Parent': rows['A'][i]['text'] == rows['Parent'][i]['text'], 'B_vs_Parent': rows['B'][i]['text'] == rows['Parent'][i]['text'],
                                   'A_vs_B': rows['A'][i]['text'] == rows['B'][i]['text']},
                'greedy_path': {'A_vs_Parent': selected_logp_comparison(resp['Parent'], resp['A'], decoder),
                                'B_vs_Parent': selected_logp_comparison(resp['Parent'], resp['B'], decoder),
                                'B_vs_A': selected_logp_comparison(resp['A'], resp['B'], decoder)}}
        for m in ('A', 'B'):
            a, b = case['correct']['Parent'], case['correct'][m]
            case['transition_' + m] = {(True, True): 'CORRECT_KEPT', (True, False): 'CORRECT_TO_WRONG', (False, True): 'WRONG_TO_CORRECT', (False, False): 'WRONG_KEPT'}[(a, b)]
        cases.append(case)
    return cases


def dev_summary(cases):
    out = {}
    for m in ('A', 'B'):
        changed = [c for c in cases if c['transition_' + m] in ('CORRECT_TO_WRONG', 'WRONG_TO_CORRECT')]
        out[m] = {'transitions': {k: sum(c['transition_' + m] == k for c in cases) for k in ('CORRECT_KEPT', 'CORRECT_TO_WRONG', 'WRONG_TO_CORRECT', 'WRONG_KEPT')},
                  'sign_test_two_sided_p_DESCRIPTIVE': core.binomial_two_sided(sum(c['transition_' + m] == 'WRONG_TO_CORRECT' for c in cases),
                                                                               sum(c['transition_' + m] == 'CORRECT_TO_WRONG' for c in cases)),
                  'texts_identical_to_Parent': sum(c['text_identical'][m + '_vs_Parent'] for c in cases),
                  'token_sequences_identical_to_Parent': sum(c['greedy_path'][m + '_vs_Parent']['identical_token_sequence'] for c in cases),
                  'changed_cases': [{'case': c['case'], 'pair': c['pair'], 'modality': c['modality'], 'transition': c['transition_' + m],
                                     'pair_category_parent': c['pair_category']['Parent'], 'pair_category_child': c['pair_category'][m],
                                     'partner_correct_parent': c['partner_correct']['Parent'], 'partner_correct_child': c['partner_correct'][m],
                                     'first_divergence_position': c['greedy_path'][m + '_vs_Parent']['first_divergence_position'],
                                     'divergence_relative_to_decision': c['greedy_path'][m + '_vs_Parent'].get('divergence_relative_to_decision'),
                                     'parent_decision_token_prob': c['greedy_path'][m + '_vs_Parent'].get('ref_decision_token_prob'),
                                     'child_decision_token_prob': c['greedy_path'][m + '_vs_Parent'].get('child_decision_token_prob')} for c in changed]}
        same = [c['greedy_path'][m + '_vs_Parent'] for c in cases if 'decision_token_logp_delta_child_minus_ref' in c['greedy_path'][m + '_vs_Parent']]
        by_mod = {}
        for c in cases:
            g = c['greedy_path'][m + '_vs_Parent']
            if 'decision_token_logp_delta_child_minus_ref' in g:
                key = c['modality'] + ('_parent_correct' if c['correct']['Parent'] else '_parent_wrong')
                by_mod.setdefault(key, []).append(g['decision_token_logp_delta_child_minus_ref'])
        out[m]['same_prefix_decision_token_logp_delta'] = {'cases': len(same), 'by_modality_and_parent_correctness':
                                                            {k: {'n': len(v), 'mean': sum(v) / len(v), 'min': min(v), 'max': max(v), 'negative': sum(x < 0 for x in v)}
                                                             for k, v in sorted(by_mod.items())},
                                                            'label': 'p(parent greedy label-first token | identical prefix) under parent vs child; NOT a class margin'}
    probs = [(c['greedy_path']['A_vs_Parent'].get('ref_decision_token_prob'), c['case']) for c in cases]
    known = sorted((p, i) for p, i in probs if p is not None)
    out['parent_decision_confidence_rank_of_changed_cases'] = {
        m: [{'case': x['case'], 'rank_lowest_first': next((r for r, (_, i) in enumerate(known) if i == x['case']), None), 'of': len(known)}
            for x in out[m]['changed_cases']] for m in ('A', 'B')}
    out['A_B_identical_predictions'] = all(c['correct']['A'] == c['correct']['B'] and c['invalid']['A'] == c['invalid']['B'] for c in cases)
    return out


# ================================================================ inventories and privacy
def evaluation_inventory(run_dir):
    run_dir = Path(run_dir)
    rows = {p.parent.name: len(list(p.glob('*_row_*_PRIVATE.json'))) for p in run_dir.glob('stage*_PRIVATE/rows_PRIVATE')}
    restores = {}
    for p in sorted(run_dir.glob('stage*_PRIVATE/*_resident_*_PRIVATE/actual_checkpoint_full_restore_PRIVATE.json')):
        value = load_json(p)
        restores.setdefault(p.parts[-3], []).append({'optimizer_step': value.get('optimizer_step'), 'policy': value.get('policy')})
    margin = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob('*') if p.is_file() and re.search(r'margin|candidate', p.name, re.I))
    steps = sorted({r['optimizer_step'] for v in restores.values() for r in v})
    return {'row_files_by_stage': rows, 'evaluated_checkpoint_restores': restores, 'evaluated_optimizer_steps_besides_parent': steps,
            'intermediate_Adam53_DEV_outputs': 'AVAILABLE' if 53 in steps else 'NOT_AVAILABLE',
            'candidate_or_margin_files': margin, 'DEV_class_margins': 'AVAILABLE' if margin else 'NOT_AVAILABLE',
            'when_degradation_occurred': 'NOT_VERIFIED: no DEV measurement exists for Adam53 A/B checkpoints' if 53 not in steps else 'MEASURABLE'}


def privacy_check(public, forbidden):
    text = canonical(public)
    require('/Users/' not in text and '/home/' not in text, 'Public aggregate contains an absolute local path')
    hits = sorted({s for s in forbidden if isinstance(s, str) and len(s) >= 6 and s in text})
    require(not hits, f'Public aggregate contains {len(hits)} forbidden private identifier/text value(s)')
    return {'checked_values': len(forbidden), 'absolute_paths': 0, 'hits': 0, 'status': 'PASS'}


def write_exclusive(path, value):
    path = Path(path)
    with path.open('x') as handle:
        handle.write(canonical(value) + '\n')
    return sha256(path)


# ================================================================ orchestration over the real immutable run
def _segment_paths(run_dir, arm, seg):
    base = Path(run_dir) / f'stage{arm}_PRIVATE'
    return base / 'arm_PRIVATE' / f'segment_{seg}', base / 'v2_journal_PRIVATE' / f'segment{seg}' / 'commit_PRIVATE.json'


def authenticate(*, root, worktree, run_dir, results_path, expected_scope_sha256=None):
    """Bind the immutable run to the committed public result, the exact scope, spec, inputs, parent and every journal commit."""
    root, worktree, run_dir = Path(root).resolve(), Path(worktree).resolve(), Path(run_dir).resolve()
    results = load_json(results_path)
    scope_sha = results['scope_file_sha256']
    require(expected_scope_sha256 in (None, scope_sha), 'Expected scope SHA differs from the committed result')
    scope_path = root / 'local_data/accuracy_pilot_v2/PILOT_V2_EXECUTION_SCOPE_PRIVATE.json'
    require_file_sha(scope_path, scope_sha, 'execution scope')
    scope = load_json(scope_path)
    require(Path(scope['output']).resolve() == run_dir, 'Run directory is not the scope-bound output')
    bound = scope['file_sha256']
    spec_path = root / 'local_data/accuracy_pilot_v2/PILOT_V2_SPEC_PRIVATE.json'
    require_file_sha(spec_path, bound[str(spec_path)], 'spec')
    spec = load_json(spec_path)
    for key in ('stage_input', 'dev24', 'runtime_spec'):
        require_file_sha(spec[key], bound[spec[key]], key)
    runtime = load_json(spec['runtime_spec'])
    require_file_sha(runtime['calibration'], bound[runtime['calibration']], 'reward calibration')
    parent = Path(runtime['parent_checkpoint'])
    require(runtime['parent_receipt_sha256'] == scope['parent']['receipt_sha256'], 'Parent receipt differs from scope')
    require_file_sha(parent / 'receipt.json', scope['parent']['receipt_sha256'], 'parent receipt')
    require_file_sha(parent / 'adapter.safetensors', scope['parent']['adapter_sha256'], 'parent adapter')
    require_file_sha(parent / 'optimizer' / 'manifest.json', bound[str(parent / 'optimizer' / 'manifest.json')], 'parent optimizer manifest')
    top = {name: require_file_sha(run_dir / name, h, name) for name, h in results['evidence']['artifact_sha256'].items()}
    terminals = verify_stage_anchors(run_dir, results)
    condition = terminals['P']['prompt_decision']['condition']
    require(condition == results['prompt_decision']['condition'], 'Prompt condition differs from the committed result')
    common_dir = run_dir / 'stageC_PRIVATE' / 'common_PRIVATE'
    common = load_json(common_dir / 'common_record_PRIVATE.json')
    verify_self_hash(common, label='common record')
    commits = {'common': verify_journal_commit(run_dir / 'stageC_PRIVATE/v2_journal_PRIVATE/common/commit_PRIVATE.json',
                                               [common_dir / 'common_record_PRIVATE.json', common_dir / 'raw_rollout_and_own_old_PRIVATE.json'], value=common)}
    receipts = {}
    for arm in ARMS:
        for seg in (0, 1):
            d, commit = _segment_paths(run_dir, arm, seg)
            r = core.verify_segment_receipt(d)
            required = [d / 'receipt_PRIVATE.json', d / 'gradient_evidence_PRIVATE.json', *(d / 'checkpoint_PRIVATE' / n for n in r['files_sha256'])]
            grad = d / 'joint_gradient_float32_PRIVATE.safetensors'
            require(grad.exists() == (r['effective_groups'] > 0), 'Joint gradient archive presence differs from admitted groups')
            if grad.exists():
                required.append(grad)
            if seg == 1:
                required += [d / f'bundle_unit{u}_PRIVATE.json' for u in core.SEGMENTS[1]] + [d / 'own_old_PRIVATE.json']
            commits[f'{arm}{seg}'] = verify_journal_commit(commit, required, value=r)
            receipts[(arm, seg)] = r
        r0, r1 = receipts[(arm, 0)], receipts[(arm, 1)]
        require(r0['arm'] == r1['arm'] == arm and r1['prev_receipt_sha256'] == r0['receipt_sha256'], 'Broken arm receipt chain')
        require(r0['common_receipt_sha256'] == common['receipt_sha256'] and r0['bundle_sha256'] == common['bundle_sha256'], 'Segment 0 is not the common rollout')
        require((r0['policy_sha_before'], r0['adam_digest_before'], r0['rng_digest_before']) == tuple(common['parent_state'][k] for k in ('policy', 'adam', 'rng')),
                'Segment 0 did not start at the common parent')
        require(r1['policy_sha_before'] == r0['policy_sha_after'] and r1['adam_digest_before'] == r0['adam_digest_after'], 'Segment 1 start differs')
        require([r0['adam_step_before'], r0['adam_step_after'], r1['adam_step_after']] == [52, 52 + r0['updated'], 52 + r0['updated'] + r1['updated']],
                'Adam step chain differs from actual updates')
    dev = load_json(spec['dev24'])
    rows = {}
    for label, stage in (('Parent', 'EP'), ('A', 'EA'), ('B', 'EB')):
        directory = run_dir / f'stage{stage}_PRIVATE' / 'rows_PRIVATE'
        require(len(list(directory.glob(f'{label}_row_*_PRIVATE.json'))) == len(dev['records_PRIVATE']) == 24, 'Exactly 24 DEV rows required: ' + label)
        out = []
        for i, rec in enumerate(dev['records_PRIVATE']):
            key = f'{label}_row_{i:02d}'
            path = directory / f'{key}_PRIVATE.json'
            commits[key] = verify_journal_commit(run_dir / f'stage{stage}_PRIVATE' / 'v2_journal_PRIVATE' / key / 'commit_PRIVATE.json', [path])
            row = load_json(path)
            resp = row['actual_PRIVATE']['response']
            require(row['key'] == key and row['id'] == rec['id'] and row['truth'] == rec['row_PRIVATE']['reward_model']['ground_truth']
                    and row['condition'] == condition and resp['text'] == row['text'] and len(resp['token_ids']) == row['tokens'] == len(resp['raw_selected_logps']),
                    'DEV row differs from its frozen record/condition: ' + key)
            out.append(row)
        rows[label] = out
        if label != 'Parent':
            restores = sorted((run_dir / f'stage{stage}_PRIVATE').glob('*_resident_*_PRIVATE/actual_checkpoint_full_restore_PRIVATE.json'))
            require(len(restores) == 1, 'Exactly one evaluated checkpoint restore required: ' + stage)
            ident = load_json(restores[0])
            r1 = receipts[(label, 1)]
            require((ident['policy'], ident['adam'], ident['rng'], ident['optimizer_step']) ==
                    (r1['policy_sha_after'], r1['adam_digest_after'], r1['rng_digest_after'], r1['adam_step_after']), 'DEV stage evaluated a different checkpoint: ' + stage)
    links = link_terminals(terminals, receipts, rows)
    stage_input = load_json(spec['stage_input'])
    units = core.condition_units(stage_input['units_PRIVATE'], condition)
    core.validate_schedule(units, {r['row_PRIVATE']['extra_info']['patient_key'] for r in dev['records_PRIVATE']})
    modules = [core, ms, shared, ppg, sys.modules['mri_rl.mlx_gspo_smoke'], sys.modules['mri_rl.vero_mri_reward'], sys.modules[__name__]]
    pins = source_pins(worktree, scope, modules)
    return {'root': root, 'worktree': worktree, 'run_dir': run_dir, 'results': results, 'scope': scope, 'spec': spec, 'runtime': runtime, 'parent_dir': parent,
            'reward_sha': sha256(runtime['calibration']), 'parent_receipt': scope['parent']['receipt_sha256'], 'condition': condition, 'units': units,
            'dev': dev, 'dev_rows': rows, 'common': common, 'common_dir': common_dir, 'receipts': receipts,
            'record': {'scope_file_sha256': scope_sha, 'scope_output_matches_run': True, 'spec_sha256': bound[str(spec_path)],
                       'top_level_artifacts_sha256': top, 'journal_commits': commits, 'source_pins': pins,
                       'stage_anchors_from_committed_result': {s: results['evidence']['all_stage_hashes'][s] for s in STAGES}, 'terminal_links': links,
                       'parent_adapter_sha256': scope['parent']['adapter_sha256'], 'reward_calibration_sha256': sha256(runtime['calibration']),
                       'prompt_condition': condition, 'DEV_rows_verified': {k: len(v) for k, v in rows.items()},
                       'segment_receipts': {f'{a}{s}': r['receipt_sha256'] for (a, s), r in receipts.items()}}}


def rederive_arm(units, common_dir, arm_dir, arm, *, reward_sha, parent_receipt, grader=None):
    """Re-derive both segments of one arm from raw rollouts / saved bundles; compare with saved gradient evidence."""
    common_dir, arm_dir = Path(common_dir), Path(arm_dir)
    common = load_json(common_dir / 'common_record_PRIVATE.json')
    verify_self_hash(common, label='common record')
    raw = load_json(common_dir / 'raw_rollout_and_own_old_PRIVATE.json')
    out = {}
    for seg in (0, 1):
        d = arm_dir / f'segment_{seg}'
        r = core.verify_segment_receipt(d)
        require(r['arm'] == arm and r['segment'] == seg, 'Receipt arm/segment differs')
        u4 = [units[i] for i in core.SEGMENTS[seg]]
        if seg == 0:
            ps = common['parent_state']
            blocks = rebuild_common_blocks(u4, raw['raws'], parent_policy=ps['policy'], reward_sha=reward_sha, parent_receipt=parent_receipt)
            for block, old in zip(blocks, raw['olds']):
                ms.check_old(block, old)
        else:
            saved = [load_json(d / f'bundle_unit{u}_PRIVATE.json') for u in core.SEGMENTS[1]]
            blocks = load_own_blocks(u4, saved, policy_sha=r['policy_sha_before'], reward_sha=reward_sha, previous_receipt=r['prev_receipt_sha256'])
            olds = load_json(d / 'own_old_PRIVATE.json')
            for block, old in zip(blocks, olds):
                ms.check_old(block, old)
        require([b.sha256 for b in blocks] == r['bundle_sha256'], f'{arm}{seg}: rebuilt bundle SHA-256s differ from the receipt')
        evidence = load_json(d / 'gradient_evidence_PRIVATE.json')
        plans = [plan_records(b, u, arm, grader=grader) for b, u in zip(blocks, u4)]
        comparison = [compare_with_saved_evidence(p, e) for p, e in zip(plans, evidence)]
        require(sum(p['effective_groups'] for p in plans) == r['effective_groups'], f'{arm}{seg}: effective groups differ from the receipt')
        out[seg] = {'receipt_sha256': r['receipt_sha256'], 'units': plans, 'saved_evidence_comparison': comparison,
                    'saved_evidence_all_match': all(c['match'] for c in comparison), 'saved_unit_l2': [e['gradient_l2'] for e in evidence],
                    'joint_gradient_l2_receipt': r['joint_gradient_l2'], 'updated': r['updated'], 'summary': segment_summary(plans),
                    'rollout_origin': r['rollout_origin']}
    return out


def _forbidden_values(ctx):
    values = set()
    for u in ctx['units']:
        for c in u['contexts']:
            values.add(c['row']['id'])
            values.add(c['row']['extra_info'].get('patient_key'))
            for img in c['row'].get('images', []):
                values.add(str(img))
    for rec in ctx['dev']['records_PRIVATE']:
        values.update({rec['id'], rec['row_PRIVATE']['extra_info'].get('patient_key'), rec.get('image_sha256')})
    for rows in ctx['dev_rows'].values():
        values.update(r['text'] for r in rows if len(r['text']) >= 24)
    for r in ctx['receipts'].values():
        values.update(r['patients'])
    return {v for v in values if isinstance(v, str)}


PUBLIC_FORBIDDEN_KEYS = frozenset({'case', 'cases', 'pair', 'partner_case', 'changed_cases', 'parent_decision_confidence_rank_of_changed_cases',
                                   'members', 'contexts', 'rewards', 'rows', 'response_id', 'admitted_response_ids', 'items', 'token_ids',
                                   'raw_selected_logps', 'text', 'texts', 'patients', 'row_id', 'image_sha256', 'partner_correct',
                                   'first_divergence_position', 'ref_decision_token_prob', 'child_decision_token_prob'})


def public_schema_check(public):
    """STRUCTURAL check that the public candidate carries no per-case/pair/member/context/score arrays (key-level, recursive).
    Together with privacy_check (literal values) this is a necessary screen, NOT a certificate of unlinkability."""
    hits = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in PUBLIC_FORBIDDEN_KEYS:
                    hits.append(path + '.' + k)
                walk(v, path + '.' + k)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                if isinstance(v, dict):
                    walk(v, f'{path}[{i}]')
    walk(public, '')
    require(not hits, 'Public aggregate carries per-case/member-level fields: ' + ','.join(sorted(hits)[:8]))
    return {'forbidden_keys_checked': len(PUBLIC_FORBIDDEN_KEYS), 'status': 'PASS', 'certifies_unlinkability': False}


def _bucket_position(n):
    return 'NONE' if n is None else ('0-8' if n <= 8 else '9-64' if n <= 64 else '65+')


def dev_public_summary(cases):
    """Counts only: accuracy by modality, transitions, pair-category counts (each pair once), and changed-case histograms."""
    out = {}
    for m in ('Parent', 'A', 'B'):
        acc = {mod: {'n': sum(c['modality'] == mod for c in cases), 'correct': sum(c['modality'] == mod and c['correct'][m] for c in cases),
                     'invalid': sum(c['modality'] == mod and c['invalid'][m] for c in cases)} for mod in ('FLAIR', 'T1')}
        seen, cats = set(), {}
        for c in cases:
            if c['pair'] not in seen:
                seen.add(c['pair'])
                cats[c['pair_category'][m]] = cats.get(c['pair_category'][m], 0) + 1
        out[m] = {'by_modality': acc, 'pair_category_counts': cats}
    for m in ('A', 'B'):
        g = lambda c: c['greedy_path'][m + '_vs_Parent']
        changed = [c for c in cases if c['transition_' + m] in ('CORRECT_TO_WRONG', 'WRONG_TO_CORRECT')]
        hist = lambda f: {k: sum(f(c) == k for c in changed) for k in sorted({f(c) for c in changed}, key=str)}
        same = [g(c)['decision_token_logp_delta_child_minus_ref'] for c in cases if 'decision_token_logp_delta_child_minus_ref' in g(c)]
        out[m].update({
            'transitions': {k: sum(c['transition_' + m] == k for c in cases) for k in ('CORRECT_KEPT', 'CORRECT_TO_WRONG', 'WRONG_TO_CORRECT', 'WRONG_KEPT')},
            'transitions_by_modality': {mod: {k: sum(c['modality'] == mod and c['transition_' + m] == k for c in cases)
                                              for k in ('CORRECT_TO_WRONG', 'WRONG_TO_CORRECT')} for mod in ('FLAIR', 'T1')},
            'texts_identical_to_Parent': sum(c['text_identical'][m + '_vs_Parent'] for c in cases),
            'token_sequences_identical_to_Parent': sum(g(c)['identical_token_sequence'] for c in cases),
            'changed_divergence_relative_to_decision_counts': hist(lambda c: g(c).get('divergence_relative_to_decision', 'NOT_VERIFIED')),
            'changed_first_divergence_position_bucket_counts': hist(lambda c: _bucket_position(g(c)['first_divergence_position'])),
            'same_prefix_decision_token_logp_delta': {'n': len(same), 'negative': sum(x < 0 for x in same), 'mean': sum(same) / len(same) if same else None,
                                                      'label': 'NOT a class margin'}})
    out['A_B_identical_predictions'] = all(c['correct']['A'] == c['correct']['B'] and c['invalid']['A'] == c['invalid']['B'] for c in cases)
    return out


def public_aggregate(report):
    """Aggregate-only view: no texts, identifiers, image hashes, paths, or per-case/pair/member/context/score arrays.
    The PRIVATE report retains the full evidence."""
    t = report['training']
    pub = {'schema': SCHEMA + '-aggregate', 'status': report['status'], 'boundaries': report['boundaries'], 'tolerances_sha256': TOLERANCES_SHA256,
           'authentication': {k: report['authentication'][k] for k in ('scope_file_sha256', 'spec_sha256', 'parent_adapter_sha256', 'reward_calibration_sha256',
                                                                         'prompt_condition', 'DEV_rows_verified', 'segment_receipts')},
           'source_pins_status': {k: v['status'] for k, v in report['authentication']['source_pins'].items()},
           'adam_formula_pins': {'constructor_present': report['adam_formula_pins']['constructor']['present'],
                                 'installed_mlx_adam': report['adam_formula_pins']['installed_mlx_adam']['status']},
           'training': {arm: {str(seg): {'saved_evidence_all_match': v['saved_evidence_all_match'], 'updated': v['updated'], 'rollout_origin': v['rollout_origin'],
                                          'joint_gradient_l2_receipt': v['joint_gradient_l2_receipt'], 'saved_unit_l2': v['saved_unit_l2'], 'summary': v['summary']}
                         for seg, v in segs.items()} for arm, segs in t.items()},
           'segment0_A_vs_B_shared_responses': {k: v for k, v in report['segment0_A_vs_B'].items() if k != 'rows'},
           'task_gradient_inventory': {k: v for k, v in report['task_gradient_inventory'].items() if k not in ('gradient_files', 'per_task_or_per_unit_gradient_files')},
           'joint_norm_constraints': report['joint_norm_constraints'],
           'tensor_analysis': report['tensor_analysis'], 'evaluation_inventory': {k: v for k, v in report['evaluation_inventory'].items()
                                                                                 if k in ('intermediate_Adam53_DEV_outputs', 'DEV_class_margins', 'when_degradation_occurred',
                                                                                          'evaluated_optimizer_steps_besides_parent')},
           'dev': dev_public_summary(report['dev']['cases']),
           'measurements_by_hypothesis': report['measurements_by_hypothesis']}
    return pub


HYPOTHESES = {
    'H1_MRI_Dallas_gradient_interference': ['task_gradient_inventory', 'joint_norm_constraints', 'training.*.summary.allocation.MRI_share_of_abs_joint_weighted_advantage'],
    'H2_GSPO_reward_or_advantage_misalignment': ['training.*.summary.sign_counts', 'training.*.summary.contrast_types', 'training.*.summary.allocation'],
    'H3_harmful_or_momentum_dominated_Adam_update': ['tensor_analysis.*.step*.metrics', 'tensor_analysis.*.two_step', 'tensor_analysis.cross_arm'],
    'H4_training_overfitting_to_four_patients': ['training.*.summary.sampled_outcomes (segment 0 vs 1 are DIFFERENT patients: not a before/after)'],
    'H5_text_side_answer_preference_or_class_prior': ['dev.summary', 'training.*.summary.allocation.sequence_level_push_toward_FLAIR_answers_regardless_of_image'],
    'H6_frozen_vision_limitation': ['tensor_analysis.lora_effective (language-model LoRA only); no vision-side measurement exists'],
}


def run_analysis(*, root, worktree, run_dir, out_dir, results_path, expected_scope_sha256=None, vocab_json=None, tokenizer_json=None,
                 mlx_optimizer_source=None, regrade=True):
    install_import_guard()
    ctx = authenticate(root=root, worktree=worktree, run_dir=run_dir, results_path=results_path, expected_scope_sha256=expected_scope_sha256)
    run_dir = ctx['run_dir']
    out_dir = Path(out_dir).resolve()
    require(not out_dir.is_relative_to(run_dir), 'Output must be outside the immutable run directory')
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / n for n in ('v2_regression_diagnosis_PRIVATE.json', 'v2_regression_diagnosis_AGGREGATE_PUBLIC_CANDIDATE.json',
                                     'v2_regression_dev_text_changes_PRIVATE.json')]
    require(not any(p.exists() for p in targets), 'Refusing to overwrite an existing diagnosis output')
    grader = None
    regrade_status = 'NOT_REQUESTED'
    if regrade:
        try:
            from mri_rl.two_task_training import make_grader
            grader = make_grader(load_json(ctx['runtime']['calibration']))
            regrade_status = 'PRODUCTION_GRADER'
        except ImportError as error:
            regrade_status = 'NOT_VERIFIED: ' + type(error).__name__
    arms = {}
    for arm in ARMS:
        arms[arm] = rederive_arm(ctx['units'], ctx['common_dir'], run_dir / f'stage{arm}_PRIVATE' / 'arm_PRIVATE', arm,
                                 reward_sha=ctx['reward_sha'], parent_receipt=ctx['parent_receipt'], grader=grader)
    seg_inputs = {(arm, seg): {'units': arms[arm][seg]['units'], 'saved_unit_l2': arms[arm][seg]['saved_unit_l2']} for arm in ARMS for seg in (0, 1)}
    inventory = task_gradient_inventory(run_dir, seg_inputs)
    # ---- tensors (streamed)
    seg_dir = lambda arm, seg: run_dir / f'stage{arm}_PRIVATE' / 'arm_PRIVATE' / f'segment_{seg}'
    parent = Checkpoint(ctx['parent_dir'], label='Parent_Adam52', adapter_sha256=ctx['scope']['parent']['adapter_sha256'])
    ck = {'P': parent}
    grads = {}
    for arm in ARMS:
        for seg in (0, 1):
            r = ctx['receipts'][(arm, seg)]
            ck[f'{arm}{53 + seg}'] = Checkpoint(seg_dir(arm, seg) / 'checkpoint_PRIVATE', label=f'{arm}_Adam{53 + seg}', adapter_sha256=r['files_sha256']['adapter.safetensors'])
            grads[f'{arm}{seg}'] = TensorFile(seg_dir(arm, seg) / 'joint_gradient_float32_PRIVATE.safetensors')
    tensor = {}
    norm_constraints = {}
    for arm in ARMS:
        steps = {}
        for seg, (a, b) in enumerate((('P', f'{arm}53'), (f'{arm}53', f'{arm}54'))):
            steps[f'step{53 + seg}'] = adam_step_report(ck[a], ck[b], grads[f'{arm}{seg}'])
            jl2 = steps[f'step{53 + seg}']['metrics']['joint_gradient_l2']
            rec = ctx['receipts'][(arm, seg)]['joint_gradient_l2']
            steps[f'step{53 + seg}']['joint_gradient_l2_matches_receipt'] = abs(jl2 - rec) <= TOLERANCES['gradient_norm_relative'] * max(rec, 1e-12)
            norm_constraints[f'{arm}{seg}'] = joint_norm_constraint(jl2, arms[arm][seg]['saved_unit_l2'], [u['task'] for u in arms[arm][seg]['units']])
        steps['two_step'] = two_step_report(ck['P'], ck[f'{arm}53'], ck[f'{arm}54'], grads[f'{arm}0'], grads[f'{arm}1'])
        tensor[arm] = steps
    tensor['cross_arm'] = cross_arm_report(ck['P'], ck['A53'], ck['A54'], ck['B53'], ck['B54'], grads['A0'], grads['A1'], grads['B0'], grads['B1'])
    scale = load_json(ctx['parent_dir'] / 'adapter_config.json')['lora_parameters']['scale']
    tensor['lora_effective'] = lora_effective_report(ck, float(scale))
    # ---- DEV
    decoder, decoder_status = None, 'NOT_SUPPLIED: decision-token fields NOT_VERIFIED'
    if vocab_json and Path(vocab_json).is_file():
        decoder = TokenDecoder.from_files(vocab_json, tokenizer_json if tokenizer_json and Path(tokenizer_json).is_file() else None)
        decoder_status = {'vocab_sha256': sha256(vocab_json), 'tokenizer_json_sha256': sha256(tokenizer_json) if tokenizer_json and Path(tokenizer_json).is_file() else None,
                          'read_as_json_only': True}
    decoded_check = None
    if decoder is not None:
        mism = 0
        for rows in ctx['dev_rows'].values():
            for r in rows:
                text = b''.join(decoder.piece(t) for t in r['actual_PRIVATE']['response']['token_ids']).decode('utf-8', 'replace')
                mism += int(r['text'] not in text and text.strip() not in r['text'])
        decoded_check = {'rows': sum(len(v) for v in ctx['dev_rows'].values()), 'decoded_text_mismatches': mism}
    cases = dev_case_analysis(ctx['dev']['records_PRIVATE'], ctx['dev_rows'], decoder)
    report = {'schema': SCHEMA, 'status': 'EXECUTED_CPU_READ_ONLY', 'implemented_by': 'Claude Code (claude-opus-5-5)', 'boundaries': dict(BOUNDARIES),
              'tolerances': TOLERANCES, 'tolerances_sha256': TOLERANCES_SHA256, 'authentication': ctx['record'],
              'adam_formula_pins': adam_formula_pins(ctx['worktree'], mlx_optimizer_source), 'regrade_status': regrade_status,
              'training': {arm: {seg: v for seg, v in arms[arm].items()} for arm in ARMS},
              'segment0_A_vs_B': shared_segment0_comparison({arm: arms[arm][0]['units'] for arm in ARMS}),
              'task_gradient_inventory': inventory, 'joint_norm_constraints': norm_constraints, 'tensor_analysis': tensor,
              'evaluation_inventory': evaluation_inventory(run_dir), 'tokenizer': decoder_status, 'decoded_text_check': decoded_check,
              'dev': {'cases': cases, 'summary': dev_summary(cases)},
              'measurements_by_hypothesis': {k: {'measurement_keys': v, 'verdict': 'PENDING_INTERPRETATION_AFTER_INDEPENDENT_EXECUTION'} for k, v in HYPOTHESES.items()}}
    changed = {m: [c['case'] for c in cases if c['transition_' + m] != 'CORRECT_KEPT' and c['transition_' + m] != 'WRONG_KEPT'] for m in ('A', 'B')}
    text_notes = {'label': 'PRIVATE qualitative material for changed DEV cases (and their same-pair partners); never publish',
                  'cases': {str(i): {'Parent': ctx['dev_rows']['Parent'][i]['text'], 'A': ctx['dev_rows']['A'][i]['text'], 'B': ctx['dev_rows']['B'][i]['text']}
                            for i in sorted({i for v in changed.values() for i in v} | {cases[i]['partner_case'] for v in changed.values() for i in v})}}
    public = public_aggregate(report)
    forbidden = _forbidden_values(ctx)
    report['public_schema'] = public['public_schema'] = public_schema_check(public)
    report['privacy'] = public['privacy'] = privacy_check(public, forbidden)
    loaded = forbidden_modules_loaded()
    require(not loaded, 'Forbidden module imported during analysis: ' + ','.join(loaded))
    report['forbidden_modules_loaded'] = public['forbidden_modules_loaded'] = loaded
    # inputs unchanged after analysis: re-verify every journal-bound file and parent payload
    reauth = authenticate(root=root, worktree=worktree, run_dir=run_dir, results_path=results_path, expected_scope_sha256=expected_scope_sha256)
    report['inputs_unchanged_after_analysis'] = public['inputs_unchanged_after_analysis'] = reauth['record'] == ctx['record']
    written = {p.name: write_exclusive(p, v) for p, v in zip(targets, (report, public, text_notes))}
    return {'written_sha256': written, 'public': public}
