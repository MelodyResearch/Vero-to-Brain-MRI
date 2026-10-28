"""Finite, gated reconstruction of the ORIGINAL GPU-produced 4-bit base cache (runtime dependency only; no new model).

Why: the CPU rebuild (accuracy_recovery_cpu_cache_v1) reproduced all 406 retained tensors byte-exactly but 0/344 quantized
payloads (mx.quantize on CPU != the historical GPU result). This module re-quantizes ONLY those 344 units with the original
NativeConvertBackend/qc.convert recipe on the default GPU, one tensor at a time, and adopts a unit only if every output part's
payload SHA-256 equals the externally pinned historical 750-unit proof. First mismatch = STOP (no second attempt, no fallback).

Scope: 750 plan units = 406 retained (copied, authenticated, from the failed CPU root, which is only READ) + 344 quantized on GPU.
No Qwen model is built or loaded; zero model forward/backward/generation/optimizer calls. The underlying quantization maximum is
688 = 344 direct mx.quantize + 344 nn.quantize module-path verifications (a temporary single nn.Linear/nn.Embedding per unit used as
a lookup check, exactly as the original converter did; not a model). 406 retained units use 0 GPU operations.

Commands: `plan` (read-only), `adopt-retained` (CPU file copy), `prepare-scope` (CPU: stream-hashes the source shards, writes an
immutable private execution scope), all three never import MLX. `execute` is the only command that can create MLX arrays, and only after:
  1. three SEPARATE files with exact SHA-256 pins: the prepared execution scope, a human authorization (instruction hash, action,
     scope digest) and an independent review (PASS / PASS_WITH_LIMITATIONS, different agent kind than the implementer, bound to the
     scope digest, the whole local source closure and the input digest). The code cannot prove who wrote a file; it requires the
     files, hashes, bindings and a once-only consumption, and consumes nothing when anything is refused;
  2. the canonical shared remaining20 heavy-job flock acquired NON-BLOCKING before MLX is imported (never forced, deleted or modified);
  3. unchanged source closure / inputs / installed MLX fingerprint, checked at admission, after the lock, before the backend and before
     every unit; the strict AC / headroom / pressure / swap-growth / disk / RSS gates.
One-shot: after consumption every outcome is final; there is no resume, no re-quantization of accepted units and no retry on a new root.

Compute/memory limitations (honest): the 8 GiB RSS limit and the disk/swap/pressure checks are SAMPLED at operation boundaries, not
enforced mid-call; the MLX 8 GiB memory limit / 64 MiB cache limit are MLX allocator guidelines inherited from NativeConvertBackend and
were not independently proven to be hard caps. MLX peak/active counters are informational. A thermal sensor that reports
SENSOR_UNAVAILABLE cannot refuse. flock is advisory: it excludes cooperating drivers only. Nothing promises against an OS-level failure.
"""
from __future__ import annotations
import ast
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from mri_rl import accuracy_recovery_cpu_cache_v1 as cc, quant_convert as qc, shared_rollout_gspo_core_v1 as shared
from mri_rl.quant_convert import NativeConvertBackend

PLAN_SHA256 = '71e2ca2709ae295d2b7aa3860701fd6ddc8ab3cd04e1a1cd011c07b420f2a68d'
WORK_ROOT_NAME = 'accuracy-recovery-historical-gpu-cache-v2'
CPU_ROOT_NAME = 'accuracy-recovery-exact-cache-v1'
GLOBAL_LOCK_REL = 'local_data/c_two_task_pool_v1/tier_c_v1/remaining20_bundle_v1/remaining20.lock'     # the existing lock every driver uses
CONSUMED_REL = 'local_data/accuracy_recovery_v1/historical_gpu_cache_v2_consumed'
SCOPE_DIR_REL = 'local_data/accuracy_recovery_v1'
MLX_MEMORY_LIMIT_BYTES = 8 * 1024**3     # must equal NativeConvertBackend.__init__ (test-enforced)
MLX_CACHE_LIMIT_BYTES = 64 * 1024**2
DISK_RESERVE_GIB = 20.0
WINDOW_SECONDS = 1800.0
MAX_GPU_UNIT_ATTEMPTS = MAX_GPU_QUANTIZE_OPS = MAX_GPU_MODULE_PATH_CHECKS = 344
MAX_UNDERLYING_QUANTIZATIONS = MAX_GPU_QUANTIZE_OPS + MAX_GPU_MODULE_PATH_CHECKS      # 688
SCOPE = {
    'name': 'VERO_TO_MRI_ACCURACY_RECOVERY_V1/historical_gpu_cache_v2', 'plan_sha256': PLAN_SHA256,
    'plan_units': 750, 'retained_units_adopted_from_cpu_root_readonly': 406, 'quantized_units_on_gpu': 344, 'retained_units_on_gpu': 0,
    'gpu_unit_attempts_max': MAX_GPU_UNIT_ATTEMPTS, 'gpu_direct_quantize_ops_max': MAX_GPU_QUANTIZE_OPS,
    'gpu_module_path_quantizations_max': MAX_GPU_MODULE_PATH_CHECKS, 'underlying_quantizations_max': MAX_UNDERLYING_QUANTIZATIONS,
    'quantization': dict(qc.QUANT), 'full_model_loaded': False, 'model_forward_calls': 0, 'model_backward_calls': 0, 'generation_calls': 0,
    'optimizer_updates': 0, 'module_construction': 'temporary single nn.Linear/nn.Embedding lookup verification only', 'device': 'gpu',
    'work_root_name': WORK_ROOT_NAME, 'cpu_root_name_read_only': CPU_ROOT_NAME, 'mlx_memory_limit_bytes': MLX_MEMORY_LIMIT_BYTES,
    'mlx_cache_limit_bytes': MLX_CACHE_LIMIT_BYTES, 'rss_envelope_gib': cc.RSS_ENVELOPE_GIB, 'swap_net_growth_cap_gib': cc.SWAP_GROWTH_CAP_GIB,
    'disk_reserve_gib': DISK_RESERVE_GIB, 'operation_start_window_seconds': WINDOW_SECONDS, 'power': 'AC strictly',
    'first_payload_mismatch': 'STOP_no_second_attempt', 'automatic_retry': False, 'one_shot_no_rerun_no_resume_no_new_root_retry': True,
    'final_identity': 'pinned historical cache certificate incl. both shard SHA-256'}
SCOPE_SHA256 = qc.sha_bytes(qc.canonical(SCOPE).encode())
SCOPE_SCHEMA = 'accuracy-recovery-historical-gpu-cache-v2-execution-scope-v1'
SCOPE_KIND = 'FUTURE_EXACT_HISTORICAL_GPU_CACHE_V2_EXECUTION_SCOPE'
ACTION = 'accuracy_recovery_historical_gpu_cache_v2_requantize_344_units'
IMPLEMENTED_BY = 'claude_code'
REVIEWER_KINDS = shared.REVIEWER_KINDS
MAX_AUTH_HOURS = 24
APPROVAL_KEYS = ('approved', 'authorized', 'training_authorized', 'native_approved', 'READY_FOR_TRAINING', 'human_authorized', 'execute')
REQUIRED_CLOSURE = ('accuracy_recovery_gpu_cache_v2.py', 'rebuild_accuracy_recovery_historical_gpu_cache_v2.py', 'accuracy_recovery_cpu_cache_v1.py',
                    'quant_convert.py', 'quant_cache.py', 'quant_probe.py', 'resource_supervisor.py', 'paced_resources.py', 'shared_rollout_gspo_core_v1.py')


class GpuGuardError(RuntimeError):
    """MLX is not on a usable GPU (the CPU would reproduce the wrong bytes); no array was created."""


class HistoricalPayloadMismatch(RuntimeError):
    """A re-quantized part differs from the pinned historical payload SHA-256; STOP, no second attempt."""


class WindowExceeded(RuntimeError):
    """The 1800 s operation-start window elapsed before all quantized units started; STOP with a partial, no rerun."""


class ScopeRefused(RuntimeError):
    """Admission/lock/ledger/pin check refused execution; no MLX array was created."""


def _req(ok, message):
    if not ok:
        raise ScopeRefused(message)


# ---- canonical paths / roots ------------------------------------------------------------------------------------
def canon_path(path):
    """Absolute path with no symlink component: a drifted or symlinked path is refused, never silently followed."""
    a, r = os.path.abspath(path), os.path.realpath(path)
    if a != r:
        raise ScopeRefused('Path is not canonical (symlink or relative drift): %s' % path)
    return Path(r)


def repo_root_from_git(start=None):
    """Canonical project root R = parent of the actual git common dir (works from a worktree)."""
    start = Path(start or Path(__file__).resolve().parent)
    out = subprocess.run(['git', '-C', str(start), 'rev-parse', '--git-common-dir'], capture_output=True, text=True, timeout=10)
    if out.returncode != 0:
        raise ScopeRefused('Cannot derive the canonical root from the git common dir')
    return (start / out.stdout.strip()).resolve().parent


def canonical_roots(repo_root, source_dir=None):
    r = Path(repo_root)
    return {'repo_root': str(r), 'work_root': str(r / 'work' / WORK_ROOT_NAME), 'cpu_root': str(r / 'work' / CPU_ROOT_NAME),
            'global_lock': str(r / GLOBAL_LOCK_REL), 'consumed_dir': str(r / CONSUMED_REL), 'source_dir': str(source_dir) if source_dir else None}


# ---- source closure / installed fingerprint ---------------------------------------------------------------------
def local_closure(entries):
    """Transitive local `mri_rl` import closure (static AST scan incl. lazy in-function imports): {absolute path: sha256}."""
    src = Path(__file__).resolve().parents[1]
    seen, todo = {}, [Path(e).resolve() for e in entries]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen[f] = shared.file_sha256(f)
        for node in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + (['mri_rl.' + a.name for a in node.names] if node.module == 'mri_rl' else [])
            for n in names:
                parts = n.split('.')
                if parts[0] == 'mri_rl' and len(parts) >= 2 and (src / 'mri_rl' / (parts[1] + '.py')).is_file():
                    todo.append((src / 'mri_rl' / (parts[1] + '.py')).resolve())
    init = (src / 'mri_rl' / '__init__.py')
    if init.is_file():
        seen[init.resolve()] = shared.file_sha256(init)
    return {str(p): h for p, h in sorted(seen.items())}


def default_entries():
    return [Path(__file__), Path(__file__).resolve().parents[2] / 'scripts' / 'rebuild_accuracy_recovery_historical_gpu_cache_v2.py']


def tree_sha256(roots):
    h, n = hashlib.sha256(), 0
    for root in sorted(str(r) for r in roots):
        for p in sorted(Path(root).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                h.update(str(p.relative_to(root)).encode() + b'\0' + shared.file_sha256(p).encode() + b'\n'); n += 1
    return h.hexdigest(), n


def installed_fingerprint():
    """Interpreter + installed numpy/MLX identity WITHOUT importing them (find_spec / metadata only)."""
    out = {'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix, 'platform': platform.platform(), 'machine': platform.machine()}
    for dist in ('numpy', 'mlx', 'mlx-metal'):
        try:
            out[dist + '_version'] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist + '_version'] = None
    spec = importlib.util.find_spec('mlx')
    if spec is not None and spec.submodule_search_locations:
        out['mlx_tree_sha256'], out['mlx_file_count'] = tree_sha256(list(spec.submodule_search_locations))
    else:
        out['mlx_tree_sha256'], out['mlx_file_count'] = None, 0
    return out


# ---- hashing / proof --------------------------------------------------------------------------------------------
def hash_unit_file(path):
    """One streaming pass: (whole-file sha256, {part: payload sha256}); payload ranges must be contiguous with no trailing bytes."""
    fh = hashlib.sha256(); parts = {}
    with Path(path).open('rb') as f:
        head = f.read(8); n = int.from_bytes(head, 'little'); hb = f.read(n); fh.update(head); fh.update(hb)
        header = {k: v for k, v in json.loads(hb).items() if k != '__metadata__'}
        pos = 0
        for name, m in sorted(header.items(), key=lambda kv: kv[1]['data_offsets'][0]):
            a, b = m['data_offsets']
            if a != pos:
                raise ValueError('Unit payload ranges are not contiguous: ' + str(path))
            h, left = hashlib.sha256(), b - a
            while left:
                chunk = f.read(min(qc.CHUNK, left))
                if not chunk:
                    raise ValueError('Short unit file: ' + str(path))
                h.update(chunk); fh.update(chunk); left -= len(chunk)
            parts[name] = h.hexdigest(); pos = b
        if f.read(1):
            raise ValueError('Trailing bytes in unit file: ' + str(path))
    return fh.hexdigest(), parts


def load_pins(plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, expected=cc.Expected()):
    """Plan, historical cache certificate and the 750-unit proof, each bound by an external sha256 pin and to each other."""
    if expected_plan_sha256 != PLAN_SHA256 and expected == cc.Expected():
        raise ScopeRefused('Production scope is bound to plan %s' % PLAN_SHA256)
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('plan_sha256') != expected_plan_sha256:
        raise ValueError('Plan hash differs from the pinned plan hash')
    hist = cc.load_historical(cert_path, cert_sha256, expected_plan_sha256, expected)
    if not proof_sha256 or qc.sha_file(proof_path) != proof_sha256:
        raise ValueError('Historical 750-unit proof hash differs from its pin')
    proof = json.loads(Path(proof_path).read_text())
    if proof.get('plan_sha256') != expected_plan_sha256 or [p['index'] for p in proof['units']] != list(range(expected.units)):
        raise ValueError('Proof is not for this plan or is not the full contiguous unit list')
    for u, p in zip(plan['units'], proof['units']):
        want = ['biases', 'scales', 'weight'] if u['action'] == 'quantize' else ['tensor']
        if p['unit_hash'] != qc.unit_hash(plan, u) or sorted(p['output_tensor_payload_sha256']) != want:
            raise ValueError('Proof entry does not match its plan unit: %d' % u['index'])
    return plan, hist, {p['index']: p for p in proof['units']}


def check_unit_file(plan, unit, proof_unit, path):
    file_sha, parts = hash_unit_file(path)
    if parts != proof_unit['output_tensor_payload_sha256']:
        raise ValueError('Unit %d payload differs from the pinned historical proof' % unit['index'])
    return file_sha


def authenticate_retained(plan, proofs, units_dir):
    """Every retained unit in `units_dir` (READ ONLY) must have a passing receipt bound to this plan/unit and match the proof exactly:
    source bytes, whole-file sha, payload sha. Returns {index: (receipt line, file path)}."""
    units_dir = Path(units_dir); lines = {}
    for line in (units_dir / 'receipts.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['index'] in lines:
            raise ValueError('Duplicate receipt for unit %d' % r['index'])
        lines[r['index']] = (line, r)
    out = {}
    for u in plan['units']:
        if u['action'] != 'retain':
            continue
        if u['index'] not in lines:
            raise ValueError('Retained unit %d has no receipt' % u['index'])
        line, r = lines[u['index']]; p = proofs[u['index']]; f = units_dir / r['output_file']
        if (r.get('plan_sha256') != plan['plan_sha256'] or r.get('unit_hash') != qc.unit_hash(plan, u) or not qc._passed(r)
                or r['output_file'] != f"unit_{u['index']:04d}.safetensors" or r['source_bytes_sha256'] != p['source_bytes_sha256']
                or r['output_file_sha256'] != p['output_file_sha256']):
            raise ValueError('Retained unit %d receipt does not match the plan/proof' % u['index'])
        if check_unit_file(plan, u, p, f) != p['output_file_sha256']:
            raise ValueError('Retained unit %d file sha differs from the proof' % u['index'])
        out[u['index']] = (line, f)
    return out


def adopt_retained(plan, proofs, cpu_units_dir, work_root):
    """Copy ONLY the authenticated retained unit files + their receipt lines into a NEW units dir; the CPU root is only read."""
    units = Path(work_root) / 'units'
    auth = authenticate_retained(plan, proofs, cpu_units_dir)
    units.mkdir(parents=True, exist_ok=False)
    for i, (line, src) in sorted(auth.items()):
        with src.open('rb') as a, (units / src.name).open('xb') as b:
            shutil.copyfileobj(a, b, qc.CHUNK); b.flush(); os.fsync(b.fileno())
    with (units / 'receipts.jsonl').open('x') as f:
        f.write(''.join(line + '\n' for _, (line, _) in sorted(auth.items()))); f.flush(); os.fsync(f.fileno())
    (units / 'plan_binding.json').write_text(json.dumps({'plan_sha256': plan['plan_sha256']}))
    authenticate_retained(plan, proofs, units)          # the copies themselves are re-verified against the proof
    manifest = {'plan_sha256': plan['plan_sha256'], 'scope_sha256': SCOPE_SHA256, 'adopted_retained_units': len(auth),
                'adopted_indices_sha256': qc.sha_bytes(qc.canonical(sorted(auth)).encode()), 'quantized_cpu_outputs_adopted': 0}
    with (Path(work_root) / 'adopt_manifest.json').open('x') as f:
        f.write(json.dumps(manifest, sort_keys=True, indent=1))
    return manifest


# ---- gates ------------------------------------------------------------------------------------------------------
def disk_estimate_bytes(plan, pending):
    out = sum(int(u['nbytes'] * qc.QUANT_OUT_FRACTION) for u in pending)
    cache = sum(int(u['nbytes'] * (qc.QUANT_OUT_FRACTION if u['action'] == 'quantize' else 1)) for u in plan['units'])
    return out + cache


def _existing_ancestor(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return p


def disk_gate(path, need_bytes, reserve_gib=DISK_RESERVE_GIB, usage=shutil.disk_usage):
    free = usage(_existing_ancestor(path)).free
    if free < need_bytes + reserve_gib * 1024**3:
        raise cc.ResourceStop('Free disk %.1f GiB below need %.1f GiB + %.0f GiB reserve' % (free / 1024**3, need_bytes / 1024**3, reserve_gib),
                              ['disk_reserve'])
    return free


def refuse_if_attempted(work_root):
    work_root = Path(work_root)
    for name in ('v2_ledger.json', 'v2_active.lock'):
        if (work_root / name).exists():
            raise ScopeRefused('%s exists: the single authorized GPU execution was already started; no rerun' % name)
    if list(work_root.glob('v2_failure_*.json')):
        raise ScopeRefused('A v2 failure receipt exists; STOP, no second attempt')


# ---- immutable execution scope (CPU preparation) -----------------------------------------------------------------
def _file_binding(path):
    p = canon_path(path)
    _req(p.is_file(), 'Not a regular file: %s' % path)
    return {'path': str(p), 'sha256': shared.file_sha256(p)}


def prepare_scope(*, plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, quant_map_path, source_dir, cpu_root,
                  work_root, scope_output, repo_root, expected=cc.Expected(), installed=installed_fingerprint, entries=None, require_mlx=True,
                  disk_usage=shutil.disk_usage):
    """CPU only, never imports MLX: stream-hash every source shard against the plan's pinned hashes, authenticate the 406 retained CPU
    outputs, fingerprint the whole local code closure and installed MLX, and publish an immutable private scope (never overwritten)."""
    r = Path(repo_root); roots = canonical_roots(r, canon_path(source_dir))
    for key, given in (('work_root', work_root), ('cpu_root', cpu_root)):
        _req(str(canon_path(given)) == roots[key], '%s must be the canonical %s' % (key, roots[key]))
    out = canon_path(scope_output)
    _req(out.parent == r / SCOPE_DIR_REL and not out.exists(), 'Scope output must be a NEW file directly under %s' % (r / SCOPE_DIR_REL))
    rep = plan_report(plan_path=plan_path, expected_plan_sha256=expected_plan_sha256, cert_path=cert_path, cert_sha256=cert_sha256, proof_path=proof_path,
                      proof_sha256=proof_sha256, cpu_root=cpu_root, work_root=work_root, quant_map_path=quant_map_path, source_dir=source_dir,
                      expected_source_sha256=None, hash_sources=True, expected=expected, disk_usage=disk_usage)
    plan, hist, proofs = load_pins(plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, expected)
    inst = installed()
    _req(not require_mlx or (inst.get('mlx_version') and inst.get('mlx_tree_sha256')), 'Installed MLX is not discoverable by this interpreter')
    inventory = [{'index': u['index'], 'action': u['action'], 'key': u['normalized_key'], 'file': u['source_file'], 'offset': u['offset'], 'nbytes': u['nbytes'],
                  'shape': u['shape'], 'module': u['module'], 'module_type': u['module_type'], 'unit_hash': qc.unit_hash(plan, u),
                  'proof_payload_sha256': proofs[u['index']]['output_tensor_payload_sha256']} for u in plan['units']]
    files = {'plan': _file_binding(plan_path), 'quant_map': _file_binding(quant_map_path), 'historical_certificate': _file_binding(cert_path),
             'historical_proof': _file_binding(proof_path)}
    shards = {k: dict(v) for k, v in plan['source_shards'].items()}
    inputs = {'roots': roots, 'files': files, 'source_shards': shards, 'inventory_sha256': shared.identity(inventory), 'plan_sha256': plan['plan_sha256'],
              'expected': asdict(expected), 'installed': inst}
    scope = {'schema': SCOPE_SCHEMA, 'scope_kind': SCOPE_KIND, 'experiment': 'VERO_TO_MRI_ACCURACY_RECOVERY_V1', 'action': ACTION, 'implemented_by': IMPLEMENTED_BY,
             'static_scope_sha256': SCOPE_SHA256, 'limits': dict(SCOPE), **inputs, 'counts': {k: rep[k] for k in (
                 'plan_units', 'retained_authenticated_from_cpu_root', 'quantized_pending_gpu')},
             'source_sha256': local_closure(entries or default_entries()), 'input_sha256': shared.identity(inputs)}
    shared.publish_once(out, scope)
    return {'scope_path': str(out), 'scope_file_sha256': shared.file_sha256(out), 'scope_sha256': shared.scope_digest(scope),
            'source_files_in_closure': len(scope['source_sha256']), 'input_sha256': scope['input_sha256'], 'source_shards_verified': len(shards)}


# ---- independent admission ---------------------------------------------------------------------------------------
def _flag_free(value):
    if isinstance(value, dict):
        return all(not (k in APPROVAL_KEYS and v is True) and _flag_free(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_flag_free(v) for v in value)
    return True


def _load_pinned(path, sha256, label):
    _req(path and shared.sha(sha256), '%s file and its exact sha256 are required' % label)
    p = canon_path(path)
    _req(p.is_file() and shared.file_sha256(p) == sha256, '%s file is missing or its sha256 differs from the supplied pin' % label)
    return json.loads(p.read_text())


class Admission:
    """Issued only by admit(). Holds the validated scope; re-verifies the reviewed source closure / inputs; consumes ONCE, last."""

    def __init__(self, scope, human, installed, consumed_dir):
        self.scope, self.human, self.installed, self.consumed_dir = scope, human, installed, Path(consumed_dir)
        self.digest = shared.scope_digest(scope)

    def check(self, phase, *, full=False):
        _req(shared.scope_digest(self.scope) == self.digest, 'Authorized scope mutated')
        for path, expected in self.scope['source_sha256'].items():
            _req(Path(path).is_file() and shared.file_sha256(path) == expected, 'Reviewed source changed before %s: %s' % (phase, path))
        if full:
            for name, b in self.scope['files'].items():
                _req(Path(b['path']).is_file() and shared.file_sha256(b['path']) == b['sha256'], 'Input file changed before %s: %s' % (phase, name))
            _req(self.installed() == self.scope['installed'], 'Installed interpreter/MLX fingerprint changed before ' + phase)
        return True

    def consume(self):
        """After every other gate: the scope and the human instruction become unusable, whatever happens next. Refusals earlier consume nothing."""
        self.consumed_dir.mkdir(parents=True, exist_ok=True)
        instruction = self.human['human_instruction_sha256']
        if (self.consumed_dir / ('authorization_consumed_%s.json' % instruction)).exists():
            raise ScopeRefused('Human authorization instruction was already consumed; a fresh instruction is required')
        try:
            shared.publish_once(self.consumed_dir / ('scope_consumed_%s.json' % self.digest), {'scope_sha256': self.digest, 'state': 'STARTED'})
        except FileExistsError:
            raise ScopeRefused('This exact scope was already consumed; a new scope with fresh human + independent review is required') from None
        try:
            shared.consume_once(self.consumed_dir)(self.human['human_instruction_sha256'])
        except ValueError as exc:
            raise ScopeRefused(str(exc)) from None

    def outcome(self, state, detail):
        try:
            shared.publish_once(self.consumed_dir / ('scope_outcome_%s.json' % self.digest), {'scope_sha256': self.digest, 'state': state, 'detail': str(detail)[:300]})
        except OSError:
            pass


def admit(*, scope_path, scope_sha256, human_path, human_sha256, review_path, review_sha256, now, repo_root, expected=cc.Expected(),
          installed=installed_fingerprint):
    """Separate scope / human / independent-review files with exact pins. Same digest-binding contract as the project's other admissions."""
    scope = _load_pinned(scope_path, scope_sha256, 'Prepared execution scope')
    human = _load_pinned(human_path, human_sha256, 'Human authorization')
    review = _load_pinned(review_path, review_sha256, 'Independent review')
    _req(isinstance(scope, dict) and scope.get('schema') == SCOPE_SCHEMA and scope.get('scope_kind') == SCOPE_KIND and scope.get('action') == ACTION and
         'mode' not in scope and _flag_free(scope), 'Not a prepared historical GPU cache v2 execution scope (or it carries an approval flag)')
    _req(scope.get('static_scope_sha256') == SCOPE_SHA256 and scope.get('limits') == SCOPE, 'Scope limits differ from the reviewed 344 + 344 = 688 budget')
    _req(scope.get('implemented_by') == IMPLEMENTED_BY, 'Scope must name its implementing agent kind')
    _req(expected != cc.Expected() or scope.get('plan_sha256') == PLAN_SHA256, 'Production scope is not bound to the frozen plan')
    _req(scope.get('expected') == asdict(expected), 'Scope inventory expectation differs (production: 750/344/406/1438/2 shards)')
    roots = scope.get('roots') or {}
    want = canonical_roots(repo_root, roots.get('source_dir'))
    _req(roots == want and roots['source_dir'] and str(canon_path(roots['source_dir'])) == roots['source_dir'] and Path(roots['source_dir']).is_dir(),
         'Scope roots differ from the canonical roots derived from the git common dir')
    _req(scope.get('input_sha256') == shared.identity({k: scope[k] for k in ('roots', 'files', 'source_shards', 'inventory_sha256', 'plan_sha256', 'expected', 'installed')}),
         'Scope input digest is inconsistent with its inputs')
    closure = scope.get('source_sha256')
    _req(isinstance(closure, dict) and all(shared.sha(v) for v in closure.values()) and all(any(p.endswith('/' + n) for p in closure) for n in REQUIRED_CLOSURE),
         'Complete local source closure required')
    adm = Admission(scope, human, installed, roots['consumed_dir'])
    adm.check('admission', full=True)
    binding = adm.digest
    _req(isinstance(human, dict) and human.get('authorized') is True and human.get('action') == ACTION and human.get('scope_sha256') == binding and
         shared.sha(human.get('human_instruction_sha256')) and human.get('approved_by'),
         'Fresh exact human authorization (instruction hash, action, scope digest) required')
    h = human
    try:
        issued, expires = (datetime.fromisoformat(h[k].replace('Z', '+00:00')) for k in ('issued_at_utc', 'expires_at_utc'))
    except (KeyError, ValueError, AttributeError):
        raise ScopeRefused('Human authorization lacks issue/expiry times') from None
    _req(issued <= now < expires and expires - issued <= timedelta(hours=MAX_AUTH_HOURS), 'Human authorization is not current or lasts longer than %d h' % MAX_AUTH_HOURS)
    _req(isinstance(review, dict) and review.get('verdict') in ('PASS', 'PASS_WITH_LIMITATIONS') and review.get('scope_sha256') == binding and
         review.get('reviewer_id') and review.get('reviewer_kind') in REVIEWER_KINDS and review.get('reviewer_kind') != scope['implemented_by'] and
         review.get('source_sha256') == closure and review.get('input_sha256') == scope['input_sha256'],
         'Independent review of this exact scope/source closure/inputs by a different agent kind (PASS or PASS_WITH_LIMITATIONS) required')
    return adm


def check_authorization(human_path, human_sha256, *, now, plan_sha256=PLAN_SHA256, script_file=None, scope_path=None, scope_sha256=None,
                        review_path=None, review_sha256=None, repo_root=None, expected=cc.Expected(), installed=installed_fingerprint):
    """Admission entry point: without the prepared scope AND the authenticated independent review nothing is authorized."""
    return admit(scope_path=scope_path, scope_sha256=scope_sha256, human_path=human_path, human_sha256=human_sha256, review_path=review_path,
                 review_sha256=review_sha256, now=now, repo_root=repo_root or repo_root_from_git(), expected=expected, installed=installed)


# ---- GPU backend ------------------------------------------------------------------------------------------------
class GpuConvertBackend(NativeConvertBackend):
    """Original NativeConvertBackend (same quantize / module-path / save / reload recipe) plus: GPU-only, quantized units only,
    per-part payload SHA vs the pinned proof BEFORE saving, distinct hard counts (344 / 344 / 344, 688 underlying), RSS/disk/window checks."""

    def __init__(self, proofs, *, mx=None, np=None, nn=None, clock=time.monotonic, t0=None, window_seconds=WINDOW_SECONDS,
                 rss_gib=cc._ps_rss_gib, rss_log=None, disk_need=lambda unit: 0, work_root=None, disk_usage=shutil.disk_usage, phase_check=None,
                 max_attempts=MAX_GPU_UNIT_ATTEMPTS, max_quantize=MAX_GPU_QUANTIZE_OPS, max_module=MAX_GPU_MODULE_PATH_CHECKS,
                 max_underlying=MAX_UNDERLYING_QUANTIZATIONS):
        if mx is None:
            NativeConvertBackend.__init__(self)       # imports MLX, sets the 8 GiB memory / 64 MiB cache limits (original code)
        else:
            self.mx, self.np, self.nn = mx, np, nn
            mx.set_memory_limit(MLX_MEMORY_LIMIT_BYTES); mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
        self.proofs, self.clock, self.window_seconds = proofs, clock, window_seconds
        self.t0 = clock() if t0 is None else t0
        self.rss_gib, self.rss_log, self.disk_need, self.work_root, self.disk_usage = rss_gib, rss_log, disk_need, work_root, disk_usage
        self.phase_check = phase_check
        self.max_attempts, self.max_quantize, self.max_module = (min(max_attempts, MAX_GPU_UNIT_ATTEMPTS), min(max_quantize, MAX_GPU_QUANTIZE_OPS),
                                                                 min(max_module, MAX_GPU_MODULE_PATH_CHECKS))
        self.max_underlying = min(max_underlying, MAX_UNDERLYING_QUANTIZATIONS)
        self.attempts = self.quantize_ops = self.module_path_checks = 0
        self.rss_peak_gib = 0.0; self.current = None; self.stopped = None
        self._check()

    @property
    def underlying_quantizations(self):
        return self.quantize_ops + self.module_path_checks

    def _check(self):
        if self.stopped:
            raise self.stopped
        mx = self.mx
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            raise GpuGuardError('MLX default device is not an available GPU; refusing (CPU would produce different quantized bytes)')

    def _rss(self, op):
        g = self.rss_gib()
        if g is None or g > cc.RSS_ENVELOPE_GIB:
            raise cc.ResourceStop('RSS %s GiB unknown or above %.1f GiB after %s' % (g, cc.RSS_ENVELOPE_GIB, op), ['process_rss_unknown_or_above_envelope'])
        self.rss_peak_gib = max(self.rss_peak_gib, g)
        if self.rss_log:
            with Path(self.rss_log).open('a') as f:
                f.write(json.dumps({'unit': self.current, 'op': op, 'rss_gib': g}) + '\n'); f.flush(); os.fsync(f.fileno())

    def _stop(self, exc):
        self.stopped = exc
        raise exc

    def _underlying_cap(self):
        if self.underlying_quantizations >= self.max_underlying:
            raise cc.AttemptCapExceeded('Underlying quantization cap %d (direct + module-path) reached' % self.max_underlying)

    def begin_unit(self):
        self._check()
        if self.phase_check:
            self.phase_check('begin_unit')
        if self.clock() - self.t0 > self.window_seconds:
            self._stop(WindowExceeded('Operation-start window of %.0f s elapsed with units still pending' % self.window_seconds))
        NativeConvertBackend.begin_unit(self)

    def load_unit(self, shard, unit):
        self._check()
        if unit['action'] != 'quantize' or unit['transform']:
            raise cc.AttemptCapExceeded('Only plain quantized units may touch the GPU; unit %d is not' % unit['index'])
        if self.attempts >= self.max_attempts:
            raise cc.AttemptCapExceeded('GPU unit attempt cap %d reached' % self.max_attempts)
        if self.work_root:
            disk_gate(self.work_root, self.disk_need(unit), usage=self.disk_usage)
        self.attempts += 1; self.current = unit['index']
        out = NativeConvertBackend.load_unit(self, shard, unit)
        self._check(); self._rss('load_unit')
        return out

    def quantize(self, w):
        self._check()
        if self.quantize_ops >= self.max_quantize:
            raise cc.AttemptCapExceeded('GPU quantize cap %d reached' % self.max_quantize)
        self._underlying_cap()
        self.quantize_ops += 1
        outs = NativeConvertBackend.quantize(self, w)
        want = self.proofs[self.current]['output_tensor_payload_sha256']
        got = dict(zip(('weight', 'scales', 'biases'), (self.bits_sha256(o) for o in outs)))
        self._rss('quantize')
        if got != want:
            self._stop(HistoricalPayloadMismatch('Unit %d GPU payload differs from the pinned historical proof (parts: %s); nothing saved'
                                                 % (self.current, ','.join(sorted(k for k in want if got.get(k) != want[k])))))
        return outs

    def quantize_module(self, w, module_type):
        self._check()
        if self.module_path_checks >= self.max_module:
            raise cc.AttemptCapExceeded('GPU module-path check cap %d reached' % self.max_module)
        self._underlying_cap()
        self.module_path_checks += 1
        out = NativeConvertBackend.quantize_module(self, w, module_type)
        self._check(); self._rss('quantize_module')
        return out

    def save(self, path, arrays):
        self._check(); NativeConvertBackend.save(self, path, arrays); self._rss('save')

    def reload(self, path):
        self._check(); out = NativeConvertBackend.reload(self, path); self._check(); self._rss('reload'); return out

    def clear(self):
        self._check(); NativeConvertBackend.clear(self)

    def release(self):
        """Final cache release that works even after a STOP (does not re-run the GPU guards)."""
        self.mx.clear_cache()


def make_gpu_backend(proofs, **kw):
    """The only constructor that imports MLX; called from execute() after every gate and the global lock."""
    return GpuConvertBackend(proofs, **kw)


# ---- pipeline ---------------------------------------------------------------------------------------------------
def check_roots(work_root, cpu_root=None):
    w = Path(work_root).resolve()
    if w.name != WORK_ROOT_NAME or Path(work_root).is_symlink():
        raise ValueError('Work root must be a non-symlink directory named ' + WORK_ROOT_NAME)
    if cpu_root is not None:
        c = Path(cpu_root).resolve()
        if c.name != CPU_ROOT_NAME or c == w or w.is_relative_to(c) or c.is_relative_to(w):
            raise ValueError('CPU root must be the separate failed root named ' + CPU_ROOT_NAME)
    return w


def write_failure(work_root, plan, exc, backend, extra):
    for n in range(1, 10_000):
        path = Path(work_root) / ('v2_failure_%03d.json' % n)
        try:
            f = path.open('x')
        except FileExistsError:
            continue
        with f:
            f.write(json.dumps({'plan_sha256': plan['plan_sha256'], 'scope_sha256': SCOPE_SHA256, 'error_type': type(exc).__name__,
                                'error': str(exc)[:500], 'reasons': getattr(exc, 'reasons', None), 'automatic_retry': False,
                                'rerun_allowed': False, 'gpu_unit_attempts': getattr(backend, 'attempts', 0),
                                'gpu_direct_quantize_ops': getattr(backend, 'quantize_ops', 0),
                                'gpu_module_path_quantizations': getattr(backend, 'module_path_checks', 0),
                                'gpu_underlying_quantizations': getattr(backend, 'underlying_quantizations', 0),
                                'current_unit': getattr(backend, 'current', None), **extra}, sort_keys=True))
            f.flush(); os.fsync(f.fileno())
        return path


def execute(*, scope_path, scope_sha256, human_path, human_sha256, review_path, review_sha256, repo_root=None, expected=cc.Expected(),
            backend_factory=make_gpu_backend, sampler=None, rss_gib=cc._ps_rss_gib, clock=time.monotonic,
            now=lambda: datetime.now(timezone.utc), installed=installed_fingerprint, disk_usage=shutil.disk_usage):
    from mri_rl.quant_cache import finalize_cache
    from mri_rl.resource_supervisor import ResourcePolicy
    adm = admit(scope_path=scope_path, scope_sha256=scope_sha256, human_path=human_path, human_sha256=human_sha256, review_path=review_path,
                review_sha256=review_sha256, now=now(), repo_root=repo_root or repo_root_from_git(), expected=expected, installed=installed)
    scope = adm.scope; roots, files = scope['roots'], scope['files']
    work = check_roots(roots['work_root']); units_dir, cache_dir, cert_out = work / 'units', work / 'cache', work / 'certificate.json'
    lock_path = Path(roots['global_lock'])
    _req(lock_path.exists() and stat.S_ISREG(os.lstat(lock_path).st_mode) and canon_path(lock_path) == lock_path,
         'The canonical remaining20 global lock must be the existing regular, non-symlink file')
    lock_fd = os.open(lock_path, os.O_RDWR)          # existing file only: never created, truncated, written or unlinked
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ScopeRefused('Canonical remaining20 global lock is held by another job; refused, not forced, deleted or modified') from None
        refuse_if_attempted(work)
        adm.check('after_global_lock', full=True)
        _, hist, proofs = load_pins(files['plan']['path'], scope['plan_sha256'], files['historical_certificate']['path'], files['historical_certificate']['sha256'],
                                    files['historical_proof']['path'], files['historical_proof']['sha256'], expected)
        plan, qmap = cc.prepare(files['plan']['path'], scope['plan_sha256'], roots['source_dir'], files['quant_map']['path'],
                                {k: v['sha256'] for k, v in scope['source_shards'].items()}, expected)       # re-streams every source shard
        if cache_dir.exists() or cert_out.exists():
            raise ValueError('Cache dir and certificate must not already exist')
        if not (work / 'adopt_manifest.json').is_file():
            raise ValueError('Retained units must be adopted first (adopt-retained)')
        adopted = authenticate_retained(plan, proofs, units_dir)                       # revalidate old receipts + retained file SHAs
        pending = [u for u in plan['units'] if u['index'] not in adopted]
        if any(u['action'] != 'quantize' for u in pending) or len(pending) != expected.quantized or \
                {p.name for p in units_dir.glob('unit_*.safetensors')} != {f.name for _, f in adopted.values()}:
            raise ValueError('Work root must hold exactly the authenticated retained units; every other unit is pending GPU quantization')
        s = cc.guarded_sampler(sampler, log_path=work / 'v2_resource.jsonl'); policy = ResourcePolicy()
        qc.gate(s, policy)
        disk_gate(work, disk_estimate_bytes(plan, pending), usage=disk_usage)
        adm.check('before_consumption', full=True)
        adm.consume()                                   # scope + human instruction consumed; everything below is final, no retry
        lock = work / 'v2_active.lock'; ledger = work / 'v2_ledger.json'
        with ledger.open('x') as f:
            f.write(json.dumps({'scope_sha256': adm.digest, 'human_instruction_sha256': adm.human['human_instruction_sha256'],
                                'started_utc': now().isoformat(), 'pending_quantized_units': len(pending)}, sort_keys=True)); f.flush(); os.fsync(f.fileno())
        with lock.open('x') as f:
            f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno())
        backend = None; t0 = clock()
        try:
            adm.check('before_backend', full=True)                   # last gate before the first MLX import
            backend = backend_factory(proofs, clock=clock, t0=t0, rss_gib=rss_gib, rss_log=work / 'v2_rss.jsonl', work_root=work, disk_usage=disk_usage,
                                      disk_need=lambda u: int(u['nbytes'] * qc.QUANT_OUT_FRACTION), max_attempts=len(pending), max_quantize=len(pending),
                                      max_module=len(pending), max_underlying=2 * len(pending), phase_check=adm.check)
            res = qc.convert(plan, roots['source_dir'], units_dir, backend, s, policy)
            if not res['complete']:
                raise qc.BlockedUnit('Conversion did not complete')
            for u in pending:                                                          # saved files re-checked against the proof
                check_unit_file(plan, u, proofs[u['index']], units_dir / f"unit_{u['index']:04d}.safetensors")
            qc.assemble_cache(plan, units_dir, cache_dir); s()
            finalize_cache(plan, units_dir, cache_dir, roots['source_dir'], qmap)
            cert = cc.verify_historical(plan, units_dir, cache_dir, roots['source_dir'], qmap, hist, expected); s()
        except Exception as exc:
            done = qc.load_done(plan, units_dir) if (units_dir / 'receipts.jsonl').exists() else {}
            write_failure(work, plan, exc, backend, {'stage': 'execute', 'last_resource_sample': s.last, 'units_with_passing_receipts': len(done),
                                                     'partial': True, 'elapsed_seconds': clock() - t0})
            adm.outcome('FAILED', '%s: %s' % (type(exc).__name__, exc))
            raise
        finally:
            if backend is not None:
                backend.release()
            lock.unlink(missing_ok=True)
        cert['gpu_reconstruction'] = {
            'device': 'gpu', 'scope_sha256': adm.digest, 'retained_units_adopted': len(adopted), 'retained_units_gpu_operations': 0,
            'gpu_unit_attempts': backend.attempts, 'gpu_quantize_ops': backend.quantize_ops, 'gpu_direct_quantize_ops': backend.quantize_ops,
            'gpu_module_path_checks': backend.module_path_checks, 'gpu_underlying_quantizations': backend.underlying_quantizations,
            'max_unit_attempts': MAX_GPU_UNIT_ATTEMPTS, 'max_quantize_ops': MAX_GPU_QUANTIZE_OPS, 'max_module_path_checks': MAX_GPU_MODULE_PATH_CHECKS,
            'max_underlying_quantizations': MAX_UNDERLYING_QUANTIZATIONS, 'model_loaded': False, 'forward_or_backward': False, 'training': False,
            'peak_process_rss_gib': max(backend.rss_peak_gib, s.peak_rss_gib), 'elapsed_seconds': clock() - t0,
            'every_unit_payload_equals_pinned_historical_proof': True}
        cert['scope'] += '; GPU re-quantization of the original base cache, not model training or inference'
        with cert_out.open('x') as f:
            f.write(json.dumps(cert, sort_keys=True, indent=1)); f.flush(); os.fsync(f.fileno())
        adm.outcome('SUCCEEDED', cert['gpu_reconstruction']['gpu_underlying_quantizations'])
        return {'certificate': cert}
    finally:
        os.close(lock_fd)                               # releases the flock; the canonical lock file itself is preserved


def plan_report(*, plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, cpu_root, work_root, quant_map_path,
                source_dir=None, expected_source_sha256=None, hash_sources=False, expected=cc.Expected(), disk_usage=shutil.disk_usage):
    """Read-only, CPU-only (never imports MLX, writes nothing): scope, authentication of the 406 retained CPU outputs, disk estimate."""
    plan, hist, proofs = load_pins(plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, expected)
    work = check_roots(work_root, cpu_root)
    qmap = json.loads(Path(quant_map_path).read_text())
    quant = [u for u in plan['units'] if u['action'] == 'quantize']
    if (len(plan['units']), len(quant)) != (expected.units, expected.quantized) or set(qmap) != {u['module'] for u in quant} or \
            any(v != qc.QUANT for v in qmap.values()):
        raise ValueError('Plan/quant-map inventory differs from the expected 750/344 affine-4bit-g64 scope')
    cpu_units = Path(cpu_root) / 'units'
    adopted = authenticate_retained(plan, proofs, cpu_units)
    src = 'NOT_RUN'
    if source_dir:
        qc.validate_plan(plan, source_dir); src = 'plan_validated'
        if hash_sources:
            qc.verify_source_shards(plan, source_dir)       # streams every shard against the plan's pinned sha256
            pinned = {k: v['sha256'] for k, v in plan['source_shards'].items()}
            if expected_source_sha256 is not None and pinned != dict(expected_source_sha256):
                raise ValueError('Plan source shard hashes differ from the supplied expected hashes')
            src = 'plan_validated_and_shards_hashed'
    need = disk_estimate_bytes(plan, quant)
    return {'command': 'plan', 'executed': False, 'scope_sha256': SCOPE_SHA256, 'plan_sha256': plan['plan_sha256'], 'plan_units': len(plan['units']),
            'retained_authenticated_from_cpu_root': len(adopted), 'quantized_pending_gpu': len(quant), 'quantized_cpu_outputs_adopted': 0,
            'gpu_unit_attempts_max': MAX_GPU_UNIT_ATTEMPTS, 'gpu_direct_quantize_ops_max': MAX_GPU_QUANTIZE_OPS,
            'gpu_module_path_quantizations_max': MAX_GPU_MODULE_PATH_CHECKS, 'underlying_quantizations_max': MAX_UNDERLYING_QUANTIZATIONS,
            'historical_shards': {k: v['bytes'] for k, v in hist['shards'].items()},
            'max_unit_forecast_gib': max(qc.forecast_gib(u) for u in quant), 'disk_need_gib': round(need / 1024**3, 2),
            'disk_free_gib': round(disk_usage(_existing_ancestor(work)).free / 1024**3, 2), 'disk_reserve_gib': DISK_RESERVE_GIB,
            'work_root_exists': work.exists(), 'source_checks': src, 'mlx_imported': any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules),
            'module_sha256': qc.sha_file(__file__)}


def main(argv=None, expected=cc.Expected(), repo_root=None, installed=installed_fingerprint):
    import argparse
    p = argparse.ArgumentParser(description='Finite GPU re-quantization of the original base cache (execute needs scope + human + independent review).')
    p.add_argument('command', choices=('plan', 'adopt-retained', 'prepare-scope', 'execute'))
    p.add_argument('--scope'); p.add_argument('--scope-sha256'); p.add_argument('--human-authorization'); p.add_argument('--human-authorization-sha256')
    p.add_argument('--independent-review'); p.add_argument('--independent-review-sha256')
    for n in ('plan', 'expected-plan-sha256', 'expected-cache-certificate', 'expected-cache-certificate-sha256', 'historical-proof',
              'historical-proof-sha256', 'quant-map', 'work-root', 'cpu-root', 'source-dir', 'scope-output'):
        p.add_argument('--' + n)
    p.add_argument('--hash-sources', action='store_true')
    a = p.parse_args(argv)
    root = Path(repo_root) if repo_root else repo_root_from_git()
    if a.command == 'execute':
        need = (a.scope, a.scope_sha256, a.human_authorization, a.human_authorization_sha256, a.independent_review, a.independent_review_sha256)
        if not all(need):
            p.error('execute needs --scope, --human-authorization and --independent-review, each with its exact --*-sha256')
        res = execute(scope_path=a.scope, scope_sha256=a.scope_sha256, human_path=a.human_authorization, human_sha256=a.human_authorization_sha256,
                      review_path=a.independent_review, review_sha256=a.independent_review_sha256, repo_root=root, expected=expected)
        print(json.dumps(res['certificate']['gpu_reconstruction'], sort_keys=True)); return
    for n in ('plan', 'expected_plan_sha256', 'expected_cache_certificate', 'expected_cache_certificate_sha256', 'historical_proof', 'historical_proof_sha256',
              'quant_map', 'work_root', 'cpu_root'):
        if not getattr(a, n):
            p.error('--%s is required' % n.replace('_', '-'))
    pins = dict(plan_path=a.plan, expected_plan_sha256=a.expected_plan_sha256, cert_path=a.expected_cache_certificate,
                cert_sha256=a.expected_cache_certificate_sha256, proof_path=a.historical_proof, proof_sha256=a.historical_proof_sha256)
    roots = canonical_roots(root)
    if str(canon_path(a.work_root)) != roots['work_root'] or str(canon_path(a.cpu_root)) != roots['cpu_root']:
        raise ScopeRefused('Work/CPU roots must be the canonical roots under %s' % root)
    if a.command == 'plan':
        print(json.dumps(plan_report(**pins, cpu_root=a.cpu_root, work_root=a.work_root, quant_map_path=a.quant_map, source_dir=a.source_dir,
                                     hash_sources=a.hash_sources, expected=expected), sort_keys=True)); return
    if a.command == 'adopt-retained':
        plan, hist, proofs = load_pins(**pins, expected=expected)
        print(json.dumps(adopt_retained(plan, proofs, Path(a.cpu_root) / 'units', a.work_root), sort_keys=True)); return
    if not (a.source_dir and a.scope_output):
        p.error('prepare-scope needs --source-dir and --scope-output')
    print(json.dumps(prepare_scope(**pins, quant_map_path=a.quant_map, source_dir=a.source_dir, cpu_root=a.cpu_root, work_root=a.work_root,
                                   scope_output=a.scope_output, repo_root=root, expected=expected, installed=installed), sort_keys=True))


if __name__ == '__main__':
    main()
