"""Streaming exact base-quantization converter (public Qwen weights only; no model construction).

Pipeline: build_plan (pure Python, from the header/quant-map metadata) -> convert (native, one tensor at a time)
-> assemble_cache (pure Python streaming byte copy, no MLX). Source bytes are read as ONE bounded fragment
from the trusted header offsets (never `mx.load` shard dictionaries, so no evaluated source array outlives its
unit). Quantized tensors use the installed mx.quantize (affine, 4-bit, group 64) and are bit-compared with the
installed nn.quantize module path; retained tensors are copied bit-exactly, except the installed vision
canonicalization (VisionModel.sanitize: patch_embed conv transpose(0,2,3,4,1)) which is called, not reimplemented.
The cache holds the SAME effective weights as the original; it is never a different 4-bit download. Evidence here
is per-unit; whole-cache identity to the loader path is certified separately.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

QUANT = {'bits': 4, 'group_size': 64, 'mode': 'affine'}
DTYPES = {'BF16': 'bfloat16'}
MAX_UNIT_PEAK_MIB = 4096        # >= ~3x the largest source tensor (1187 MiB)
MAX_RESIDUAL_ACTIVE_MIB = 256   # active MLX bytes after a unit's release + cache clear; larger means a leak
CHUNK = 64 * 1024 * 1024
VISION_CANON = 'installed_vision_sanitize'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class BlockedUnit(RuntimeError):
    """A prior unit failed its checks; its evidence is preserved and that unit stays blocked."""


def build_plan(meta_dir):
    """Exact plan from the public metadata; verifies the 344-module map is fully matched by source tensors."""
    meta = Path(meta_dir)
    index = json.loads((meta / 'source_header_index.json').read_text())
    qmap = json.loads((meta / 'base_quantization.json').read_text())
    shards = json.loads((meta / 'source_shard_sha256.json').read_text())
    prov = json.loads((meta / 'metadata_provenance.json').read_text())
    tensors = index['tensors']
    if len(tensors) != 750 or len(qmap) != 344 or any(v != QUANT for v in qmap.values()):
        raise ValueError('Metadata inventory differs from the expected 750 tensors / 344 affine-4bit-g64 modules')
    if any('position_ids' in t['normalized_key'] for t in tensors):
        raise ValueError('position_ids tensors are omitted by the installed rule; none are expected')
    by_key = {t['normalized_key']: t for t in tensors}
    if len(by_key) != len(tensors):
        raise ValueError('Duplicate normalized keys')
    missing = [m for m in qmap if m + '.weight' not in by_key]
    if missing:
        raise ValueError('Quantization map modules without a source weight: %d' % len(missing))
    shard_names = {s['file'] for s in shards['source_shards']}
    ordered = sorted(tensors, key=lambda t: (Path(t['source_shard']).name, t['absolute_data_offset']))
    units = []
    for i, t in enumerate(ordered):
        key = t['normalized_key']
        quant = key.endswith('.weight') and key[:-len('.weight')] in qmap
        module = key[:-len('.weight')] if quant else None
        shape = t['shape']
        canon = VISION_CANON if 'patch_embed.proj.weight' in key else None
        if canon:
            if shape != [1152, 3, 2, 16, 16]:
                raise ValueError('Unexpected patch_embed shape')
            shape = [1152, 2, 16, 16, 3]
        if Path(t['source_shard']).name not in shard_names or t['dtype'] != 'BF16':
            raise ValueError('Unit source differs from hashed shards / BF16')
        nbytes = t['data_offsets'][1] - t['data_offsets'][0]
        units.append({'index': i, 'source_key': t['original_key'], 'normalized_key': key,
                      'source_file': Path(t['source_shard']).name, 'offset': t['absolute_data_offset'],
                      'nbytes': nbytes, 'source_shape': t['shape'], 'dtype': 'BF16', 'shape': shape,
                      'action': 'quantize' if quant else 'retain', 'module': module,
                      'module_type': ('embedding' if module and module.endswith('embed_tokens') else 'linear') if quant else None,
                      'transform': canon})
    plan = {'schema': 'exact-quant-plan-v1', 'quantization': dict(QUANT), 'units': units,
            'source_shards': {s['file']: {'bytes': s['bytes'], 'sha256': s['sha256']} for s in shards['source_shards']},
            'lineage': {'metadata_files_sha256': prov['files_sha256'], 'original_adapter_config_sha256': prov.get('original_adapter_configuration_sha256'),
                        'shard_identity_file_sha256': sha_file(meta / 'source_shard_sha256.json')}}
    plan['plan_sha256'] = sha_bytes(canonical({k: v for k, v in plan.items() if k != 'plan_sha256'}).encode())
    return plan


def unit_hash(plan, unit):
    return sha_bytes(canonical({'plan': plan['plan_sha256'], 'unit': unit}).encode())


def validate_plan(plan, source_dir):
    body = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    if plan.get('plan_sha256') != sha_bytes(canonical(body).encode()):
        raise ValueError('Plan hash does not match its content')
    if plan['quantization'] != QUANT:
        raise ValueError('Quantization must be affine 4-bit group 64')
    units = plan['units']
    if not units or [u['index'] for u in units] != list(range(len(units))):
        raise ValueError('Units must be contiguous from 0')
    if len({u['normalized_key'] for u in units}) != len(units):
        raise ValueError('Duplicate keys')
    src = Path(source_dir).resolve()
    for u in units:
        f = (src / u['source_file']).resolve()
        if not f.is_relative_to(src) or not f.is_file() or u['source_file'] not in plan['source_shards']:
            raise ValueError('Source shard missing, outside the source dir, or not in the hashed inventory')
        if u['offset'] + u['nbytes'] > plan['source_shards'][u['source_file']]['bytes']:
            raise ValueError('Unit range exceeds its shard')
        if u['action'] == 'quantize' and (u['module_type'] not in ('linear', 'embedding') or u['shape'][-1] % 64):
            raise ValueError('Quantized unit shape/type invalid')
    return units


def verify_source_shards(plan, source_dir):
    for name, meta in plan['source_shards'].items():
        p = Path(source_dir) / name
        if p.stat().st_size != meta['bytes'] or sha_file(p) != meta['sha256']:
            raise ValueError('Source shard identity differs from the immutable inventory: ' + name)


def gate(sampler, policy):
    from mri_rl.quant_probe import gate_check
    return gate_check(sampler, policy)


def _passed(r):
    return bool(r['reload_bitwise_equal']) and r['module_path_bitwise_equal'] in (True, None) and \
        r['retained_bitwise_equal'] in (True, None) and bool(r['residual_within_budget'])


def load_done(plan, out_dir):
    """Receipts bound to this exact plan/unit; any failed, foreign or changed evidence blocks (never 'done')."""
    out_dir = Path(out_dir); path = out_dir / 'receipts.jsonl'; done = {}
    units = {u['index']: u for u in plan['units']}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            u = units.get(r['index'])
            if u is None or r.get('plan_sha256') != plan['plan_sha256'] or r.get('unit_hash') != unit_hash(plan, u):
                raise BlockedUnit('Receipt does not belong to this exact plan/unit; preserve and investigate')
            if not _passed(r):
                raise BlockedUnit('Unit %d has a failed receipt; blocked pending root-cause review' % r['index'])
            f = out_dir / r['output_file']
            if not f.is_file() or sha_file(f) != r['output_file_sha256']:
                raise BlockedUnit('Unit %d output no longer matches its receipt' % r['index'])
            done[r['index']] = r
    return done


def convert(plan, source_dir, out_dir, backend, sampler, policy=None, *, verify_module_path=True, max_units=None):
    """Process pending units serially; safe to re-invoke. Returns aggregate counts."""
    from mri_rl.resource_supervisor import ResourcePolicy
    policy = policy or ResourcePolicy()
    units = validate_plan(plan, source_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    binding = out_dir / 'plan_binding.json'
    if binding.exists():
        if json.loads(binding.read_text())['plan_sha256'] != plan['plan_sha256']:
            raise BlockedUnit('Output directory is bound to a different plan')
    else:
        binding.write_text(json.dumps({'plan_sha256': plan['plan_sha256']}))
    done = load_done(plan, out_dir)
    receipts_path = out_dir / 'receipts.jsonl'; converted = 0
    for u in units:
        if u['index'] in done:
            continue
        if max_units is not None and converted >= max_units:
            break
        s = gate(sampler, policy)
        backend.begin_unit()
        w, raw_sha = backend.load_unit(Path(source_dir) / u['source_file'], u)
        if u['transform'] == VISION_CANON:
            w = backend.canonicalize_vision(u['normalized_key'], w)
        if backend.shape(w) != u['shape'] or backend.dtype(w) != DTYPES[u['dtype']]:
            raise ValueError('Source tensor differs from plan')
        names = ('weight', 'scales', 'biases') if u['action'] == 'quantize' else ('tensor',)
        outs = backend.quantize(w) if u['action'] == 'quantize' else (w,)
        out_file = f"unit_{u['index']:04d}.safetensors"
        backend.save(out_dir / out_file, dict(zip(names, outs)))
        back = backend.reload(out_dir / out_file)
        reload_ok = all(backend.bitwise_equal(o, back[n]) for n, o in zip(names, outs))
        module_ok = retained_ok = None
        if u['action'] == 'quantize' and verify_module_path:
            via = backend.quantize_module(w, u['module_type'])
            module_ok = all(backend.bitwise_equal(a, b) for a, b in zip(outs, via))
            del via
        if u['action'] == 'retain':
            # without a transform the retained bits must equal the raw source bytes exactly
            retained_ok = backend.bitwise_equal(w, back['tensor']) and (
                u['transform'] is not None or backend.bits_sha256(w) == raw_sha)
        peak = backend.peak_mib()
        shapes = {n: backend.shape(o) for n, o in zip(names, outs)}
        dtypes = {n: backend.dtype(o) for n, o in zip(names, outs)}
        del w, outs, back
        backend.clear()
        residual = backend.active_mib()
        s2 = gate(sampler, policy)
        r = {'index': u['index'], 'plan_sha256': plan['plan_sha256'], 'unit_hash': unit_hash(plan, u),
             'source_key': u['source_key'], 'normalized_key': u['normalized_key'], 'action': u['action'],
             'transform': u['transform'], 'source_bytes_sha256': raw_sha, 'output_file': out_file,
             'output_file_sha256': sha_file(out_dir / out_file), 'shapes': shapes, 'dtypes': dtypes,
             'reload_bitwise_equal': reload_ok, 'module_path_bitwise_equal': module_ok, 'retained_bitwise_equal': retained_ok,
             'mlx_peak_mib': peak, 'mlx_active_after_release_mib': residual,
             'residual_within_budget': residual <= MAX_RESIDUAL_ACTIVE_MIB,
             'swap_used_gib': [s.swap_used_gib, s2.swap_used_gib], 'available_gib': [s.available_gib, s2.available_gib],
             'pressure_free_percent': [s.pressure_free_percent, s2.pressure_free_percent]}
        r['passed'] = _passed(r)
        with receipts_path.open('a') as f:
            f.write(json.dumps(r, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
        converted += 1
        if not r['passed']:
            raise BlockedUnit('Check failed in unit %d (evidence preserved; unit blocked)' % u['index'])
        if peak > MAX_UNIT_PEAK_MIB:
            raise RuntimeError('Unit exceeded its memory budget; stopping')
    finished = len(done) + converted
    return {'units_total': len(units), 'units_done': finished, 'converted_this_call': converted,
            'complete': finished == len(units)}


def _read_st_header(path):
    with Path(path).open('rb') as f:
        n = int.from_bytes(f.read(8), 'little'); header = json.loads(f.read(n))
    return {k: v for k, v in header.items() if k != '__metadata__'}, 8 + n


def assemble_cache(plan, out_dir, cache_dir, shard_limit_bytes=4 << 30):
    """Pure-Python streaming assembly of per-unit outputs into safetensors shards + index (no MLX, no big read)."""
    out_dir, cache_dir = Path(out_dir), Path(cache_dir)
    done = load_done(plan, out_dir)
    if len(done) != len(plan['units']):
        raise ValueError('Conversion is incomplete; refusing to assemble a partial cache')
    cache_dir.mkdir(parents=True, exist_ok=False)
    entries = []  # (name, dtype, shape, unit file, begin, end)
    for u in plan['units']:
        header, base = _read_st_header(out_dir / done[u['index']]['output_file'])
        for part, meta in header.items():
            name = (u['module'] + '.' + part) if u['action'] == 'quantize' else u['normalized_key']
            entries.append((name, meta['dtype'], meta['shape'], out_dir / done[u['index']]['output_file'],
                            base + meta['data_offsets'][0], base + meta['data_offsets'][1]))
    names = [e[0] for e in entries]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate tensor names in the assembled cache')
    groups, cur, size = [], [], 0
    for e in entries:
        n = e[5] - e[4]
        if cur and size + n > shard_limit_bytes:
            groups.append(cur); cur, size = [], 0
        cur.append(e); size += n
    groups.append(cur)
    weight_map, total = {}, 0
    for gi, group in enumerate(groups, 1):
        fname = f'model-{gi:05d}-of-{len(groups):05d}.safetensors'
        header, pos = {}, 0
        for name, dtype, shape, _, b, e in group:
            header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [pos, pos + (e - b)]}; pos += e - b
            weight_map[name] = fname
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' ' * (-len(raw) % 8)
        with (cache_dir / fname).open('wb') as out:
            out.write(len(raw).to_bytes(8, 'little')); out.write(raw)
            for name, dtype, shape, src, b, e in group:
                with src.open('rb') as f:
                    f.seek(b); left = e - b
                    while left:
                        chunk = f.read(min(CHUNK, left)); out.write(chunk); left -= len(chunk)
            out.flush(); os.fsync(out.fileno())
        total += pos
    (cache_dir / 'model.safetensors.index.json').write_text(json.dumps(
        {'metadata': {'total_size': total}, 'weight_map': weight_map}, sort_keys=True))
    manifest = {'plan_sha256': plan['plan_sha256'], 'tensor_count': len(names), 'shard_sha256':
                {p.name: sha_file(p) for p in sorted(cache_dir.glob('*.safetensors'))},
                'note': 'Derived exact cache; config/tokenizer integration and loader identity certification are separate.'}
    (cache_dir / 'cache_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=1))
    return {'shards': len(groups), 'tensors': len(names), 'total_bytes': total}


class NativeConvertBackend:
    def __init__(self):
        import numpy as np
        import mlx.core as mx
        import mlx.nn as nn
        self.np, self.mx, self.nn = np, mx, nn
        mx.set_memory_limit(8 * 1024**3); mx.set_cache_limit(64 * 1024**2)
    def begin_unit(self): self.mx.reset_peak_memory()
    def load_unit(self, shard, unit):
        """One bounded fragment: chunked readinto a preallocated buffer; hash the raw bytes; no shard-level arrays."""
        np, mx = self.np, self.mx
        buf = np.empty(unit['nbytes'] // 2, dtype=np.uint16); view = memoryview(buf).cast('B'); h = hashlib.sha256()
        with Path(shard).open('rb') as f:
            f.seek(unit['offset']); got = 0
            while got < unit['nbytes']:
                n = f.readinto(view[got:got + min(CHUNK, unit['nbytes'] - got)])
                if not n: raise ValueError('Short read from source shard')
                h.update(view[got:got + n]); got += n
        arr = mx.array(buf.reshape(unit['source_shape'])).view(mx.bfloat16)
        del buf, view
        mx.eval(arr)
        return arr, h.hexdigest()
    def canonicalize_vision(self, key, w):
        from mlx_vlm.models.qwen3_vl.vision import VisionModel
        out = VisionModel.sanitize(None, {key: w})  # installed rule, called as a plain function
        if set(out) != {key}: raise ValueError('Installed sanitize changed the key set')
        r = out[key]; self.mx.eval(r); return r
    def quantize(self, w):
        out = self.mx.quantize(w, **QUANT); self.mx.eval(*out); return out
    def quantize_module(self, w, module_type):
        holder = self.nn.Module()
        if module_type == 'embedding':
            holder.m = self.nn.Embedding(w.shape[0], w.shape[1])
        else:
            holder.m = self.nn.Linear(w.shape[1], w.shape[0], bias=False)
        holder.m.weight = w
        self.nn.quantize(holder, group_size=QUANT['group_size'], bits=QUANT['bits'], mode=QUANT['mode'])
        q = holder.m; out = (q.weight, q.scales, q.biases); self.mx.eval(*out); return out
    def _bits(self, a): return a.view(self.mx.uint16) if a.dtype == self.mx.bfloat16 else a
    def bitwise_equal(self, a, b):
        return a.shape == b.shape and a.dtype == b.dtype and bool(self.mx.array_equal(self._bits(a), self._bits(b)).item())
    def bits_sha256(self, a):
        return hashlib.sha256(self.np.ascontiguousarray(self.np.array(self._bits(a))).tobytes()).hexdigest()
    def shape(self, a): return list(a.shape)
    def dtype(self, a): return str(a.dtype).replace('mlx.core.', '')
    def save(self, path, arrays): self.mx.save_safetensors(str(path), arrays)
    def reload(self, path): return self.mx.load(str(path))
    def peak_mib(self): return self.mx.get_peak_memory() / 1024**2
    def active_mib(self): return self.mx.get_active_memory() / 1024**2
    def clear(self): self.mx.clear_cache()


def main():
    import argparse
    from mri_rl.resource_supervisor import ResourcePolicy, sample_macos
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('plan', 'convert', 'assemble'))
    p.add_argument('--metadata-dir', type=Path)
    p.add_argument('--plan', type=Path)
    p.add_argument('--expected-plan-sha256')
    p.add_argument('--source-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--max-units', type=int)
    a = p.parse_args()
    if a.command == 'plan':
        plan = build_plan(a.metadata_dir); a.output.write_text(json.dumps(plan, sort_keys=True))
        print(json.dumps({'plan_sha256': plan['plan_sha256'], 'units': len(plan['units']),
                          'quantized': sum(u['action'] == 'quantize' for u in plan['units'])})); return
    plan = json.loads(a.plan.read_text())
    if plan['plan_sha256'] != a.expected_plan_sha256:
        raise SystemExit('Plan hash mismatch')
    if a.command == 'assemble':
        print(json.dumps(assemble_cache(plan, a.output, a.cache_dir))); return
    validate_plan(plan, a.source_dir); verify_source_shards(plan, a.source_dir)
    policy = ResourcePolicy(); gate(sample_macos, policy)
    print(json.dumps(convert(plan, a.source_dir, a.output, NativeConvertBackend(), sample_macos, policy, max_units=a.max_units)))


if __name__ == '__main__':
    main()
