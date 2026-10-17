"""CPU-only pilot launcher checks on synthetic rows; imports no MLX/model runtime."""
import json
import sys
import pytest
from mri_rl import continuous_rl_pilot as pilot


def test_import_loads_no_native_runtime():
    assert not any(m in sys.modules for m in ('mlx.core', 'mlx_vlm', 'torch'))


def test_schedule_is_explicit_ordered_and_rejects_missing_or_duplicate(tmp_path):
    train = tmp_path / 'train.jsonl'
    train.write_text('\n'.join(json.dumps({'id': f'r{i}'}) for i in range(3)) + '\n')
    assert [r['id'] for r in pilot.load_schedule(train, ['r2', 'r0'])] == ['r2', 'r0']
    with pytest.raises(ValueError):
        pilot.load_schedule(train, ['r0', 'r0'])
    with pytest.raises(ValueError):
        pilot.load_schedule(train, ['r0', 'missing'])


def test_launch_configuration_binds_every_pilot_source(tmp_path):
    cfg = pilot.launch_configuration(output=tmp_path, seed=7, record_ids=['a', 'b'], group_size=2, mode='train')
    assert set(cfg['source_sha256']) == set(pilot.SOURCES)
    assert {'continuous_rl.py', 'mlx_continuous_gspo.py', 'continuous_rl_pilot.py'} <= set(cfg['source_sha256'])
    assert cfg['schedule_record_ids'] == ['a', 'b']
    changed = pilot.launch_configuration(output=tmp_path, seed=8, record_ids=['a', 'b'], group_size=2, mode='train')
    assert changed != cfg


def test_pilot_declared_budget_is_the_enforced_budget():
    import inspect
    assert pilot.MAX_UPDATES == 1
    src = inspect.getsource(pilot.main)
    assert 'max_optimizer_updates=MAX_UPDATES' in src and 'resume=args.resume' in src
    assert 'max_new_groups=min(len(schedule), 3)' in src


def test_native_constructor_uses_serial_materialization_not_bulk_eval():
    from pathlib import Path
    text = (Path(pilot.ROOT) / 'src/mri_rl/mlx_continuous_gspo.py').read_text()
    assert 'materialize_serial' in text and 'mx.eval(self.model.parameters())' not in text


def test_serial_materialization_visits_every_leaf_once_in_order():
    from mri_rl.likelihood_loading import materialize_serial
    seen, marks = [], []
    leaves = [(f'k{i}', i * 1.5) for i in range(130)]
    materialize_serial(leaves, evaluate=seen.append, collect=lambda: None, clear=lambda: None,
                       observe=marks.append, observe_every=64)
    assert seen == [v for _, v in leaves]
    assert marks == ['materialized:64', 'materialized:128', 'materialized:130']


def test_loader_helper_is_in_source_inventory():
    assert 'likelihood_loading.py' in pilot.SOURCES and 'likelihood_loading.py' in pilot.source_hashes()


def test_reload_refuses_boundary_that_differs_from_approved_digest():
    import inspect
    src = inspect.getsource(pilot.main)
    assert src.index("args.approved_boundary = boundary_receipt") < src.index('preflight(args')
    assert "== args.approved_boundary" in src
    assert src.index("== args.approved_boundary") < src.index('NativeGSPOBackend(adapter')


def _bare_backend(order, mutate):
    import hashlib
    from types import SimpleNamespace
    from mri_rl.mlx_continuous_gspo import NativeGSPOBackend
    b = NativeGSPOBackend.__new__(NativeGSPOBackend)
    b.model = SimpleNamespace(eval=lambda: None, config=None)
    b.processor = None
    b.contract = SimpleNamespace(system_instruction='s', assistant_prefix='A', enable_thinking=False)
    b.config = {'max_tokens': 16}
    b.mx = SimpleNamespace(clear_cache=lambda: None)
    b.template = lambda *a, **k: 'x<|image_pad|>A'
    def before(name):
        order.append('callback')
        if mutate == 'callback':
            image.write_bytes(b'CHANGED')
    b.before_unit = before
    def gen(*a, **k):
        order.append('generate')
        if mutate == 'generate':
            image.write_bytes(b'CHANGED')
        return SimpleNamespace(text='t', token_ids=[1])
    b.generate = gen
    return b


def test_greedy_generate_rechecks_image_after_callback_before_generate_and_after(tmp_path):
    global image
    import hashlib
    image = tmp_path / 'i.png'
    row = {'id': 'r', 'images': [str(image)], 'extra_info': {'question': 'q'}}
    for mutate, must_generate in ((None, True), ('callback', False), ('generate', True)):
        image.write_bytes(b'ORIGINAL')
        expected = hashlib.sha256(b'ORIGINAL').hexdigest()
        order = []
        b = _bare_backend(order, mutate)
        if mutate:
            with pytest.raises(ValueError):
                b.greedy_generate(row, expected)
        else:
            assert b.greedy_generate(row, expected)['image_sha256'] == expected
        assert ('generate' in order) == must_generate


def test_reload_guard_failure_still_closes_backend(tmp_path):
    import hashlib
    image = tmp_path / 'i.png'; image.write_bytes(b'ORIGINAL')
    expected = hashlib.sha256(b'ORIGINAL').hexdigest()
    image.write_bytes(b'CHANGED BY CONSTRUCTOR CALLBACK')
    class B:
        closed = False; generated = False
        def greedy_generate(self, row, sha): B.generated = True; return {}
        def close(self): B.closed = True
    with pytest.raises(ValueError):
        pilot.guarded_reload_generation(B(), {'images': [str(image)]}, expected)
    assert B.closed and not B.generated


def test_cache_option_is_optional_bound_into_scope_and_sourced():
    import inspect
    assert 'likelihood_cache.py' in pilot.SOURCES
    cfg = pilot.launch_configuration(output='/x', seed=7, record_ids=['a'], group_size=2, mode='train',
                                     cache_binding={'certificate_sha256': 'z'})
    assert cfg['derived_cache'] == {'certificate_sha256': 'z'}
    assert pilot.launch_configuration(output='/x', seed=7, record_ids=['a'], group_size=2, mode='train')['derived_cache'] is None
    src = inspect.getsource(pilot.main)
    assert 'cache=cache' in src and 'cache_binding_for(args)' in src and src.index('cache_binding_for') < src.index('preflight(args')


def test_cache_binding_requires_certificate_and_matches_original_non_weight_files(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from tests.test_likelihood_cache import certified
    c, cp, sha, arts, _ = certified(tmp_path, monkeypatch)
    model = Path(next(iter(arts))).parent           # synthetic "original model dir" with config/tokenizer/chat template
    args = SimpleNamespace(cache_dir=c, cache_certificate=cp, cache_certificate_sha256=sha, model=model)
    assert pilot.cache_binding_for(args)['certificate_sha256'] == sha
    assert pilot.cache_binding_for(SimpleNamespace(cache_dir=None)) is None
    with pytest.raises(ValueError): pilot.cache_binding_for(SimpleNamespace(cache_dir=c, cache_certificate=None, cache_certificate_sha256=None, model=model))
    with pytest.raises(ValueError): pilot.cache_binding_for(SimpleNamespace(cache_dir=c, cache_certificate=cp, cache_certificate_sha256='0' * 64, model=model))


def test_native_constructor_cache_branch_is_serial_parity_checked_and_restores_saved_map():
    import inspect
    from mri_rl import mlx_continuous_gspo as g
    text = inspect.getsource(g.NativeGSPOBackend.__init__)
    assert 'verify_loaded_parameters' in text and 'quantization_bits=None' in text and "load(str(cache['dir'])" in text
    assert text.index("before_unit('cache_parameters_verified')") < text.index('quantization_bits=None')
    assert "load(str(model_path),lazy=True)" in text      # uncached path unchanged
