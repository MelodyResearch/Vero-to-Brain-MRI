"""CPU stub checks for the serial dev-evaluation entrypoint; synthetic, no model."""
import json
import hashlib
import pytest
from mri_rl import continuous_rl_dev_eval as dev


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
    first = dev.evaluate(rows[:2], lambda: Stub(log), out, expected, resume=False)
    assert first['generated'] == 2 and len(out.read_text().splitlines()) == 2
    log2 = []
    again = dev.evaluate(rows[:2], lambda: Stub(log2), out, expected, resume=True)
    assert again['generated'] == 0 and again['reused'] == 2 and log2 == []
    more = dev.evaluate(rows, lambda: Stub(log2), out, expected, resume=True)
    assert log2 == ['q2'] and more['reused'] == 2
    with pytest.raises(ValueError):
        dev.evaluate(rows, lambda: Stub([]), out, expected, resume=False)


def test_resume_rejects_changed_configuration(tmp_path):
    manifest, rows, model = make(tmp_path)
    out = tmp_path / 'o.jsonl'
    dev.evaluate(rows, lambda: Stub([]), out, dev.expected_config(model, manifest, None, 256), resume=False)
    with pytest.raises(ValueError):
        dev.evaluate(rows, lambda: Stub([]), out, dev.expected_config(model, manifest, None, 128), resume=True)


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
        dev.evaluate(rows, factory, tmp_path / 'o.jsonl', dev.expected_config(model, manifest, None, 256), resume=False)
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
