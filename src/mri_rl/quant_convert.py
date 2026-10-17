"""Streaming exact base-quantization converter (public Qwen weights only; no model construction).

Converts one source tensor at a time from a hash-bound PLAN into per-tensor output files, with the same
installed quantizer the loader uses (mx.quantize, affine). Retained tensors are copied bit-exactly. Every unit:
shared persistent resource policy -> load one tensor -> compute -> save -> reload bitwise check -> optional
nn.quantize module-path bitwise check -> fsynced receipt -> cache clear. Resume skips units whose receipt and
output hash still verify. No second model, no MRI, no generation. The derived cache has the SAME effective
weights as the original (never another 4-bit download); full 344+retained identity is a separate verification
that this module only records unit-by-unit evidence for.

PLAN (JSON, sha256 bound by the caller): {"quantization": {"bits":4,"group_size":64,"mode":"affine"},
 "units": [{"index":i,"source_key":..,"source_file":..,"action":"quantize"|"retain","module_type":"linear"|"embedding",
            "shape":[..],"dtype":"BF16","output_key":..}]}
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

QUANT = {'bits': 4, 'group_size': 64, 'mode': 'affine'}
DTYPES = {'BF16': 'bfloat16', 'F32': 'float32', 'F16': 'float16'}
MAX_UNIT_PEAK_MIB = 4096     # >= ~3x the largest source tensor (1187 MiB); expected quantize peak is lower


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_plan(plan, source_dir):
    require = lambda c, m: (_ for _ in ()).throw(ValueError(m)) if not c else None
    require(plan.get('quantization') == QUANT, 'Quantization must be affine 4-bit group 64')
    units = plan['units']
    require(units and [u['index'] for u in units] == list(range(len(units))), 'Units must be contiguous from 0')
    require(len({u['source_key'] for u in units}) == len(units), 'Duplicate source keys')
    require(len({u['output_key'] for u in units}) == len(units), 'Duplicate output keys')
    src = Path(source_dir).resolve()
    for u in units:
        require(u['action'] in ('quantize', 'retain'), 'Unknown unit action')
        require(u['dtype'] in DTYPES and (u['action'] == 'retain' or u['dtype'] == 'BF16'), 'Unsupported dtype')
        require(u['action'] == 'retain' or (u['module_type'] in ('linear', 'embedding') and
                u['shape'][-1] % QUANT['group_size'] == 0), 'Quantized unit shape/type invalid')
        f = (src / u['source_file']).resolve()
        require(f.is_relative_to(src) and f.is_file(), 'Source shard missing or outside source dir')
    return units


def gate(sampler, policy):
    from mri_rl.quant_probe import gate_check
    return gate_check(sampler, policy)


def convert(plan, source_dir, out_dir, backend, sampler, policy=None, *, verify_module_path=True, max_units=None):
    """Process pending units serially; returns counts. Safe to re-invoke (resumes from receipts)."""
    from mri_rl.resource_supervisor import ResourcePolicy
    policy = policy or ResourcePolicy()
    units = validate_plan(plan, source_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = out_dir / 'receipts.jsonl'
    done = {}
    if receipts_path.exists():
        for line in receipts_path.read_text().splitlines():
            r = json.loads(line)
            f = out_dir / r['output_file']
            if not f.is_file() or _sha(f) != r['output_file_sha256']:
                raise ValueError('Existing unit output no longer matches its receipt; preserve and investigate')
            done[r['index']] = r
    converted = 0
    for u in units:
        if u['index'] in done:
            continue
        if max_units is not None and converted >= max_units:
            break
        s = gate(sampler, policy)
        w = backend.load(str(Path(source_dir) / u['source_file']), u['source_key'])
        if backend.shape(w) != u['shape'] or backend.dtype(w) != DTYPES.get(u['dtype']):
            raise ValueError('Source tensor differs from plan')
        names = ('weight', 'scales', 'biases') if u['action'] == 'quantize' else ('tensor',)
        outs = backend.quantize(w) if u['action'] == 'quantize' else (w,)
        out_file = f"unit_{u['index']:04d}.safetensors"
        backend.save(out_dir / out_file, dict(zip(names, outs)))
        back = backend.reload(out_dir / out_file)
        reload_ok = all(backend.bitwise_equal(o, back[n]) for n, o in zip(names, outs))
        module_ok = None
        if u['action'] == 'quantize' and verify_module_path:
            via = backend.quantize_module(w, u['module_type'])
            module_ok = all(backend.bitwise_equal(a, b) for a, b in zip(outs, via))
        retain_ok = backend.bitwise_equal(w, back['tensor']) if u['action'] == 'retain' else None
        s2 = gate(sampler, policy)
        r = {'index': u['index'], 'source_key': u['source_key'], 'output_key': u['output_key'], 'action': u['action'],
             'output_file': out_file, 'output_file_sha256': _sha(out_dir / out_file),
             'shapes': {n: backend.shape(o) for n, o in zip(names, outs)},
             'dtypes': {n: backend.dtype(o) for n, o in zip(names, outs)},
             'reload_bitwise_equal': reload_ok, 'module_path_bitwise_equal': module_ok, 'retained_bitwise_equal': retain_ok,
             'mlx_peak_mib': backend.peak_mib(), 'swap_used_gib': [s.swap_used_gib, s2.swap_used_gib],
             'available_gib': [s.available_gib, s2.available_gib], 'pressure_free_percent': [s.pressure_free_percent, s2.pressure_free_percent]}
        with receipts_path.open('a') as f:
            f.write(json.dumps(r, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
        converted += 1
        del w, outs, back
        backend.clear()
        if not (reload_ok and module_ok is not False and retain_ok is not False):
            raise RuntimeError('Bitwise mismatch in unit %d; evidence preserved, stopping' % u['index'])
        if r['mlx_peak_mib'] > MAX_UNIT_PEAK_MIB:
            raise RuntimeError('Unit exceeded its memory budget; stopping')
    total = len(units); finished = sum(1 for u in units if u['index'] in done) + converted
    return {'units_total': total, 'units_done': finished, 'converted_this_call': converted, 'complete': finished == total}


class NativeConvertBackend:
    def __init__(self):
        import mlx.core as mx
        import mlx.nn as nn
        self.mx, self.nn = mx, nn
        mx.set_memory_limit(8 * 1024**3); mx.set_cache_limit(64 * 1024**2); mx.reset_peak_memory()
        self._shards = {}
    def load(self, path, key):
        shard = self._shards.get(path) or self._shards.setdefault(path, self.mx.load(path))  # lazy mmap handles
        w = shard[key]; self.mx.eval(w); return w
    def quantize(self, w):
        out = self.mx.quantize(w, **QUANT); self.mx.eval(*out); return out
    def quantize_module(self, w, module_type):
        holder = self.nn.Module()
        if module_type == 'embedding':
            holder.m = self.nn.Embedding(w.shape[0], w.shape[1]); holder.m.weight = w
        else:
            holder.m = self.nn.Linear(w.shape[1], w.shape[0], bias=False); holder.m.weight = w
        self.nn.quantize(holder, group_size=QUANT['group_size'], bits=QUANT['bits'], mode=QUANT['mode'])
        q = holder.m; out = (q.weight, q.scales, q.biases); self.mx.eval(*out); return out
    def _bits(self, a): return a.view(self.mx.uint16) if a.dtype == self.mx.bfloat16 else a
    def bitwise_equal(self, a, b):
        return a.shape == b.shape and a.dtype == b.dtype and bool(self.mx.array_equal(self._bits(a), self._bits(b)).item())
    def shape(self, a): return list(a.shape)
    def dtype(self, a): return str(a.dtype).replace('mlx.core.', '')
    def save(self, path, arrays): self.mx.save_safetensors(str(path), arrays)
    def reload(self, path): return self.mx.load(str(path))
    def peak_mib(self): return self.mx.get_peak_memory() / 1024**2
    def clear(self): self.mx.clear_cache()


def main():
    import argparse
    from mri_rl.resource_supervisor import ResourcePolicy, sample_macos
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--expected-plan-sha256', required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-units', type=int)
    a = p.parse_args()
    if _sha(a.plan) != a.expected_plan_sha256:
        raise SystemExit('Plan hash mismatch')
    plan = json.loads(a.plan.read_text()); policy = ResourcePolicy()
    validate_plan(plan, a.source_dir); gate(sample_macos, policy)
    print(json.dumps(convert(plan, a.source_dir, a.output, NativeConvertBackend(), sample_macos, policy,
                             max_units=a.max_units)))


if __name__ == '__main__':
    main()
