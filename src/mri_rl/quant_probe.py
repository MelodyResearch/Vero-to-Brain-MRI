"""Tiny native probe: is mx.quantize (affine, 4-bit, group 64) bitwise-identical to the installed
nn.quantize/QuantizedLinear path for two prospectively chosen smallest public base tensors?

Public Qwen weights only (no MRI, no images, no generation, no parameter update, no 8B load). Header and
sha256 validation happen in pure Python before any native import. One worker, <=2 tensors, tiny memory budget.
Each tensor receipt is fsynced immediately; receipts hold hashes and resource aggregates, never tensor values.
This does not establish full-model numerical parity.
"""
from __future__ import annotations
import hashlib
import json
import os
import struct
from pathlib import Path

ALLOWED = {'vision_tower.blocks.0.attn.proj.weight': [1152, 1152],
           'vision_tower.blocks.1.attn.proj.weight': [1152, 1152]}
MAX_FILE_BYTES = 8 * 1024 * 1024        # source is ~2.53 MiB BF16
TINY_PEAK_MIB = 256                      # per-tensor MLX peak budget (expected ~10 MiB)
QUANT = {'bits': 4, 'group_size': 64, 'mode': 'affine'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_inputs(manifest_path, base_dir):
    """Pure-Python allowlist/shape/dtype/hash validation of single-tensor safetensors fragments."""
    manifest = json.loads(Path(manifest_path).read_text())
    entries = manifest['tensors'] if isinstance(manifest, dict) else manifest
    if not 1 <= len(entries) <= 2 or len({e['key'] for e in entries}) != len(entries):
        raise ValueError('One or two distinct allowlisted tensors required')
    out = []
    for e in entries:
        key = e['key']
        if key not in ALLOWED or e.get('shape') != ALLOWED[key] or e.get('dtype') != 'BF16':
            raise ValueError('Tensor outside the prospective allowlist')
        path = (Path(base_dir) / e['file']).resolve()
        if not path.is_relative_to(Path(base_dir).resolve()) or not path.is_file() or path.is_symlink():
            raise ValueError('Fragment must be a regular file inside the input directory')
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError('Fragment exceeds the tiny-operation size budget')
        if _sha(path) != e['sha256']:
            raise ValueError('Fragment hash differs from provenance')
        with path.open('rb') as f:
            (hlen,) = struct.unpack('<Q', f.read(8))
            header = json.loads(f.read(hlen))
        tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        if set(tensors) != {key} or tensors[key]['dtype'] != 'BF16' or tensors[key]['shape'] != ALLOWED[key]:
            raise ValueError('Fragment header differs from the allowlisted tensor')
        out.append({**e, 'path': str(path), 'bytes': size})
    return out


def gate_check(sampler):
    """Tiny-op gate: stop only on conditions that matter for any native work; no 8B load floor here."""
    from mri_rl.paced_resources import decisions
    from mri_rl.resource_supervisor import ResourcePolicy
    s = sampler()
    action, reasons = decisions(s, loading=False)
    level, _ = ResourcePolicy().observe(s)
    stop = (action == 'STOP' or level == 'CRITICAL' or s.thermal_state == 'THROTTLED' or
            bool(s.other_heavy_processes) or s.other_heavy_processes is None)
    if stop:
        raise RuntimeError('Tiny probe stopped by critical resource condition: ' + ','.join(reasons or [level]))
    return s


def resource_view(s, backend):
    return {'available_gib': s.available_gib, 'pressure_free_percent': s.pressure_free_percent,
            'swap_used_gib': s.swap_used_gib, 'process_rss_gib': s.process_rss_gib,
            'power_source': s.power_source, 'battery_percent': s.battery_percent,
            'mlx_active_mib': backend.active_mib(), 'mlx_peak_mib': backend.peak_mib()}


def run(entries, out_dir, backend, sampler):
    """Process tensors serially; the receipt of each completed tensor is on disk before the next starts."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=False)
    receipts = out_dir / 'probe_receipts.jsonl'
    results = []
    for e in entries:
        before = resource_view(gate_check(sampler), backend)
        w = backend.load(e['path'], e['key'])
        ref = backend.quantize_ref(w)            # mx.quantize
        via = backend.quantize_module(w)         # nn.quantize on a Linear holding w (the real library path)
        names = ('weight', 'scales', 'biases')
        same = {n: backend.bitwise_equal(a, b) for n, a, b in zip(names, ref, via)}
        saved = out_dir / (e['key'].replace('.', '_') + '.safetensors')
        backend.save(saved, dict(zip(names, ref)))
        back = backend.reload(saved)
        reload_same = {n: backend.bitwise_equal(ref[i], back[n]) for i, n in enumerate(names)}
        after = resource_view(gate_check(sampler), backend)
        receipt = {'key': e['key'], 'source_sha256': e['sha256'], 'quantization': QUANT,
                   'shapes': {n: backend.shape(a) for n, a in zip(names, ref)},
                   'dtypes': {n: backend.dtype(a) for n, a in zip(names, ref)},
                   'module_path_bitwise_equal': same, 'reload_bitwise_equal': reload_same,
                   'dtype_shape_equal': backend.same_layout(ref, via),
                   'output_file_sha256': _sha(saved), 'resource_before': before, 'resource_after': after,
                   'peak_within_budget': after['mlx_peak_mib'] <= TINY_PEAK_MIB}
        with receipts.open('a') as f:
            f.write(json.dumps(receipt, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
        results.append(receipt)
        if not receipt['peak_within_budget']:
            raise RuntimeError('Tiny probe exceeded its memory budget; stopping')
    ok = all(all(r['module_path_bitwise_equal'].values()) and all(r['reload_bitwise_equal'].values()) and
             r['dtype_shape_equal'] for r in results)
    return {'tensors': len(results), 'all_bitwise_equal': ok}


class NativeBackend:
    """Imports MLX lazily; only constructed after validate_inputs and an initial gate pass."""
    def __init__(self):
        import mlx.core as mx
        import mlx.nn as nn
        self.mx, self.nn = mx, nn
        mx.set_memory_limit(512 * 1024**2); mx.set_cache_limit(32 * 1024**2)
        mx.reset_peak_memory()
    def active_mib(self): return self.mx.get_active_memory() / 1024**2
    def peak_mib(self): return self.mx.get_peak_memory() / 1024**2
    def load(self, path, key):
        arrays = self.mx.load(path)
        assert set(arrays) == {key}
        w = arrays[key]; self.mx.eval(w); return w
    def quantize_ref(self, w):
        out = self.mx.quantize(w, **QUANT); self.mx.eval(*out); return out
    def quantize_module(self, w):
        lin = self.nn.Linear(w.shape[1], w.shape[0], bias=False); lin.weight = w
        holder = self.nn.Module(); holder.proj = lin
        self.nn.quantize(holder, group_size=QUANT['group_size'], bits=QUANT['bits'], mode=QUANT['mode'])
        q = holder.proj; out = (q.weight, q.scales, q.biases); self.mx.eval(*out); return out
    def _bits(self, a):
        return a.view(self.mx.uint16) if a.dtype == self.mx.bfloat16 else a
    def bitwise_equal(self, a, b):
        return a.shape == b.shape and a.dtype == b.dtype and bool(self.mx.array_equal(self._bits(a), self._bits(b)).item())
    def same_layout(self, xs, ys): return all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(xs, ys))
    def shape(self, a): return list(a.shape)
    def dtype(self, a): return str(a.dtype)
    def save(self, path, arrays): self.mx.save_safetensors(str(path), arrays)
    def reload(self, path): return self.mx.load(str(path))


def main():
    import argparse
    from pathlib import Path as P
    from mri_rl.resource_supervisor import sample_macos
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=P, required=True)
    p.add_argument('--input-dir', type=P, required=True)
    p.add_argument('--output', type=P, required=True)
    a = p.parse_args()
    entries = validate_inputs(a.manifest, a.input_dir)
    gate_check(sample_macos)
    print(json.dumps(run(entries, a.output, NativeBackend(), sample_macos)))


if __name__ == '__main__':
    main()
