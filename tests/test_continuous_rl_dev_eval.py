"""CPU stub checks for the serial dev-evaluation entrypoint; synthetic, no model."""
import json
import hashlib
import pytest
from mri_rl import continuous_rl_dev_eval as dev


def H(rows):
    return {r['record_id']: hashlib.sha256(open(r['image'], 'rb').read()).hexdigest() for r in rows}


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def make(tmp_path, n=3):
    manifest = tmp_path / 'm.jsonl'; rows = []
    for i in range(n):
        img = tmp_path / f'{i}.png'; img.write_bytes(b'IMG%d' % i)
        rows.append({'record_id': f'q{i}', 'category': 'c', 'image': str(img), 'prompt': [{'role': 'user', 'content': 'q'}]})
    manifest.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    model = tmp_path / 'model'; model.mkdir(); (model / 'config.json').write_text('{}')
    return manifest, rows, model


class Stub:
    def __init__(self, log, hits=False): self.log, self.hits, self.closed = log, hits, False
    def generate_record(self, record, sha, max_tokens):
        assert sha == hashlib.sha256(open(record['image'], 'rb').read()).hexdigest()
        self.log.append(record['record_id'])
        return {'response': 'r', 'token_ids': [1], 'image_placeholder_count': 1, 'generation_tokens': 1,
                'finish_reason': 'stop', 'hit_max_tokens': self.hits}
    def close(self): self.closed = True


def test_serial_eval_writes_every_row_and_resume_never_regenerates(tmp_path):
    manifest, rows, model = make(tmp_path)
    expected = dev.expected_config(model, manifest, None, 256)
    out = tmp_path / 'o.jsonl'; log = []
    first = dev.evaluate(rows[:2], lambda: Stub(log), out, expected, resume=False, approved_image_sha256=H(rows))
    assert first['generated'] == 2 and len(out.read_text().splitlines()) == 2
    log2 = []
    again = dev.evaluate(rows[:2], lambda: Stub(log2), out, expected, resume=True, approved_image_sha256=H(rows), approved_output_sha256=sha(out))
    assert again['generated'] == 0 and again['reused'] == 2 and log2 == []
    more = dev.evaluate(rows, lambda: Stub(log2), out, expected, resume=True, approved_image_sha256=H(rows), approved_output_sha256=sha(out))
    assert log2 == ['q2'] and more['reused'] == 2
    with pytest.raises(ValueError):
        dev.evaluate(rows, lambda: Stub([]), out, expected, resume=False, approved_image_sha256=H(rows))


def test_resume_rejects_changed_configuration(tmp_path):
    manifest, rows, model = make(tmp_path)
    out = tmp_path / 'o.jsonl'
    dev.evaluate(rows, lambda: Stub([]), out, dev.expected_config(model, manifest, None, 256), resume=False, approved_image_sha256=H(rows))
    with pytest.raises(ValueError):
        dev.evaluate(rows, lambda: Stub([]), out, dev.expected_config(model, manifest, None, 128), resume=True, approved_image_sha256=H(rows), approved_output_sha256=sha(out))


def test_subset_keeps_manifest_order_and_rejects_unknown(tmp_path):
    manifest, rows, _ = make(tmp_path)
    assert [r['record_id'] for r in dev.select_records(manifest, ['q2', 'q0'])] == ['q0', 'q2']
    with pytest.raises(ValueError):
        dev.select_records(manifest, ['q9'])


def test_backend_closed_when_generation_fails(tmp_path):
    manifest, rows, model = make(tmp_path)
    class Bad(Stub):
        def generate_record(self, *a): raise ValueError('guard')
    holder = []
    def factory(): holder.append(Bad([])); return holder[0]
    with pytest.raises(ValueError):
        dev.evaluate(rows, factory, tmp_path / 'o.jsonl', dev.expected_config(model, manifest, None, 256), resume=False, approved_image_sha256=H(rows))
    assert holder[0].closed


def test_native_generate_record_rechecks_image_after_callback(tmp_path):
    from types import SimpleNamespace
    from mri_rl.mlx_continuous_gspo import NativeGSPOBackend
    manifest, rows, _ = make(tmp_path)
    r = rows[0]; sha = hashlib.sha256(open(r['image'], 'rb').read()).hexdigest()
    for mutate in (False, True):
        calls = []
        b = NativeGSPOBackend.__new__(NativeGSPOBackend)
        b.model = SimpleNamespace(eval=lambda: None, config=None); b.processor = None
        b.mx = SimpleNamespace(clear_cache=lambda: None)
        b.template = lambda *a, **k: 'x<|image_pad|>'
        b.before_unit = lambda name: open(r['image'], 'wb').write(b'X') if mutate else None
        b.generate = lambda *a, **k: calls.append(1) or SimpleNamespace(text='t', token_ids=[1])
        if mutate:
            with pytest.raises(ValueError):
                b.generate_record(r, sha, 16)
            assert not calls
        else:
            assert b.generate_record(r, sha, 16)['response'] == 't'


def test_dev_source_inventory_hashes_loader_and_backend():
    assert {'mlx_continuous_gspo.py', 'likelihood_loading.py', 'frozen_eval.py'} <= set(dev.source_hashes())


def test_image_changed_after_review_is_rejected_before_backend_construction(tmp_path):
    manifest, rows, model = make(tmp_path)
    approved = H(rows)
    open(rows[1]['image'], 'wb').write(b'REPLACED AFTER REVIEW')
    built = []
    with pytest.raises(ValueError):
        dev.evaluate(rows, lambda: built.append(1) or Stub([]), tmp_path / 'o.jsonl',
                     dev.expected_config(model, manifest, None, 256), resume=False, approved_image_sha256=approved)
    assert not built and not (tmp_path / 'o.jsonl').exists()


def test_changed_starting_output_or_model_is_rejected_before_construction(tmp_path):
    manifest, rows, model = make(tmp_path)
    expected = dev.expected_config(model, manifest, None, 256); out = tmp_path / 'o.jsonl'
    dev.evaluate(rows[:1], lambda: Stub([]), out, expected, resume=False, approved_image_sha256=H(rows))
    reviewed = sha(out); out.write_text(out.read_text() + '\n')
    built = []
    with pytest.raises(ValueError):
        dev.evaluate(rows, lambda: built.append(1) or Stub([]), out, expected, resume=True,
                     approved_image_sha256=H(rows), approved_output_sha256=reviewed)
    out.write_text(out.read_text().rstrip('\n') + '\n'); good = sha(out)
    def bad(): raise ValueError('Model artifacts changed after review')
    with pytest.raises(ValueError):
        dev.evaluate(rows, lambda: built.append(1) or Stub([]), out, expected, resume=True,
                     approved_image_sha256=H(rows), approved_output_sha256=good, verify_model=bad)
    assert not built


def test_model_inventory_binds_every_file_and_rejects_wrong_dir(tmp_path, monkeypatch):
    import types
    model = tmp_path / 'm'; model.mkdir()
    for n, b in (('config.json', b'{}'), ('preprocessor_config.json', b'p'), ('tokenizer_config.json', b't'),
                 ('chat_template.json', b'c'), ('w.safetensors', b'WEIGHTS')):
        (model / n).write_bytes(b)
    (tmp_path / 'artifact_paths.json').write_text(json.dumps({
        'model_config': str(model / 'config.json'), 'processor_config': str(model / 'preprocessor_config.json'),
        'tokenizer_config': str(model / 'tokenizer_config.json'), 'chat_template': str(model / 'chat_template.json')}))
    monkeypatch.setattr(dev, 'check_artifacts', lambda c, p: [])  # contract content checks are covered by experiment_contract tests
    c = types.SimpleNamespace()
    inv = dev.model_inventory(model, c, tmp_path / 'contract.json')
    assert {f['path'] for f in inv['files']} == {'config.json', 'preprocessor_config.json', 'tokenizer_config.json',
                                                 'chat_template.json', 'w.safetensors'}
    (model / 'w.safetensors').write_bytes(b'OTHER WEIGHTS')
    assert dev.model_inventory(model, c, tmp_path / 'contract.json') != inv
    other = tmp_path / 'other'; other.mkdir(); (other / 'w.safetensors').write_bytes(b'x')
    with pytest.raises(ValueError):
        dev.model_inventory(other, c, tmp_path / 'contract.json')
