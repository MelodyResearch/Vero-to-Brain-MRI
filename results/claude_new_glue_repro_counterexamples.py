import sys, pathlib, tempfile, runpy, json, types
import numpy as np
repo = pathlib.Path.cwd(); sys.path[:0] = [str(repo / 'src'), str(repo)]
out = {}
# R1: calibration — a crash between mkdtemp('.pending-') and the reservation write leaves an unrecoverable directory
j = runpy.run_path(str(repo / 'tests/test_tier_c_calibration_journal.py'))
run, JB = j['run'], j['JournalBackend']
with tempfile.TemporaryDirectory() as t:
    base = pathlib.Path(t) / 'c'
    class Crash(JB):
        crash_at = 3
    try: run(base, factory=Crash)
    except RuntimeError: pass
    for p in base.glob('.pending-*'):
        (p / 'attempt_reservation.json').unlink()          # simulate death before the reservation was durable (dir exists, empty of reservation)
        for q in p.iterdir(): q.unlink()                   # nothing native had started: empty directory
    try:
        run(base, resume=True); out['empty_pending_dir'] = 'RECOVERED'
    except ValueError as e:
        out['empty_pending_dir'] = 'STUCK: ' + str(e)
# R2: SFT — frozen plan target token IDs are never compared with what the native step trains on (count only)
h = runpy.run_path(str(repo / 'tests/test_tier_c_sft.py')); sft = h['sft']; ROWS = h['ROWS']
cls = sft.native_backend_class(); b = cls.__new__(cls); seen = {}
import hashlib
b.mx = types.SimpleNamespace(array=lambda v, dtype=None: np.array(v), float32=np.float32, eval=lambda *x: None, clear_cache=lambda: None, sum=np.sum)
def vg(model, fn):
    def call(model_, prompt_ids, target_ids, pixels, kwargs):
        seen['target_ids'] = target_ids.tolist(); return np.array(1.0), {'g': np.array([1.0], dtype=np.float32)}
    return call
b.nn = types.SimpleNamespace(value_and_grad=vg)
b.model = types.SimpleNamespace(eval=lambda: None, config=types.SimpleNamespace(model_type='fixture'), trainable_parameters=lambda: {})
b.contract = types.SimpleNamespace(system_instruction='s', enable_thinking=False, assistant_prefix='END')
b.processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(encode=lambda *a, **k: [999]))      # DIFFERENT id, SAME count (1) as a hypothetical plan id [7]
b.template = lambda *a, **k: '<|image_pad|>END'; b.prepare_inputs = lambda *a, **k: {'input_ids': np.array([[1, 2]], dtype=np.int64)}
b.special_tokens = lambda *a: False; b.before_unit = lambda n: None
b.tree_map = lambda fn, *t: {k: fn(*(x[k] for x in t)) for k in t[0]}; b.tree_flatten = lambda d: list(d.items())
b.optimizer = types.SimpleNamespace(update=lambda m, g: None, state={})
ex = ROWS[:4]; b.expected_images = {r['id']: hashlib.sha256(pathlib.Path(r['images'][0]).read_bytes()).hexdigest() for r in ex}
ev = b.sft_step(ex, pathlib.Path('/tmp'))
planned = [7]                                              # what a frozen plan would hold for a 1-token target
out['sft_target_ids'] = {'planned_ids': planned, 'trained_ids': seen['target_ids'], 'native_step_accepted': True,
                         'counts_equal_so_runner_would_accept': len(planned) == ev['target_tokens'][0]}
assert not any(x in sys.modules for x in ('mlx.core', 'mlx_vlm', 'torch'))
print(json.dumps(out, indent=1))
