import json
import struct
from pathlib import Path
import pytest
from mri_rl import quant_convert as qc, quant_cache as cache
from tests.test_quant_convert import make, healthy, _st  # noqa: F401


def build(tmp_path):
    plan, src = make(tmp_path)
    out = tmp_path / 'o'; out.mkdir(); lines = []
    for u in plan['units']:
        f = f"unit_{u['index']:04d}.safetensors"
        t = ({'weight': ('U32', [2, 2], bytes([u['index'] + 1] * 16)), 'scales': ('BF16', [2, 1], bytes([7] * 4)), 'biases': ('BF16', [2, 1], bytes([9] * 4))}
             if u['action'] == 'quantize' else {'tensor': ('BF16', [2], bytes([5] * 4))})
        _st(out / f, t)
        lines.append(json.dumps({'index': u['index'], 'plan_sha256': plan['plan_sha256'], 'unit_hash': qc.unit_hash(plan, u),
            'output_file': f, 'output_file_sha256': qc.sha_file(out / f), 'reload_bitwise_equal': True,
            'module_path_bitwise_equal': True if u['action'] == 'quantize' else None,
            'retained_bitwise_equal': True if u['action'] == 'retain' else None, 'residual_within_budget': True, 'peak_within_budget': True}))
    (out / 'receipts.jsonl').write_text('\n'.join(lines) + '\n')
    qc.assemble_cache(plan, out, tmp_path / 'c', shard_limit_bytes=40)
    (src / 'config.json').write_text(json.dumps({'model_type': 'qwen3_vl', 'a': 1}))
    (src / 'tokenizer.json').write_text('{"t":1}'); (src / 'chat_template.json').write_text('{"c":1}')
    qmap = {u['module']: dict(qc.QUANT) for u in plan['units'] if u['action'] == 'quantize'}
    return plan, src, out, tmp_path / 'c', qmap


def test_finalize_and_certify_roundtrip(tmp_path):
    plan, src, out, c, qmap = build(tmp_path)
    cache.finalize_cache(plan, out, c, src, qmap)
    cert = cache.certify_cache(plan, out, c, src, qmap)
    assert cert['tensors_checked'] == 7 and cert['quantized_modules'] == 2 and cert['config_delta_keys'] == ['quantization', 'quantization_config']
    assert json.loads((c / 'config.json').read_text())['quantization']['m0'] == qc.QUANT
    assert (c / 'tokenizer.json').read_bytes() == (src / 'tokenizer.json').read_bytes()


def test_certify_rejects_payload_corruption_config_drift_and_map_mismatch(tmp_path):
    plan, src, out, c, qmap = build(tmp_path)
    cache.finalize_cache(plan, out, c, src, qmap)
    cfg = json.loads((c / 'config.json').read_text()); cfg['a'] = 2; (c / 'config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError): cache.certify_cache(plan, out, c, src, qmap)
    cfg['a'] = 1; (c / 'config.json').write_text(json.dumps(cfg))
    shard = sorted(c.glob('model-*.safetensors'))[0]; raw = bytearray(shard.read_bytes()); raw[-1] ^= 1; shard.write_bytes(bytes(raw))
    with pytest.raises(ValueError): cache.certify_cache(plan, out, c, src, qmap)
    with pytest.raises(ValueError): cache.finalize_cache(plan, out, c, src, {'m0': dict(qc.QUANT)})
