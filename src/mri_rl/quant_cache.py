"""Derived-cache integration and parameter-lineage certificate (pure Python; no MLX, no model, no big reads).

Takes the assembled shards from quant_convert.assemble_cache and (1) copies every non-weight source file byte-exactly,
(2) writes config.json = the ORIGINAL config plus ONLY the quantization keys the installed loader reads
(`quantization` / `quantization_config`: the 344-module map, same affine/4-bit/g64), keeping the original config as
config_source.json, and (3) certifies payload, index, lineage and config delta. Architecture, preprocessing, tokenizer,
chat template and generation settings are byte-identical copies. It certifies bytes and lineage; it does not load or
run the model (that needs its own measured materialization profile).
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from mri_rl.quant_convert import QUANT, _read_st_header, load_done, sha_file, unit_hash

WEIGHT_SUFFIXES = ('.safetensors',)
WEIGHT_INDEX = 'model.safetensors.index.json'


def quant_config(qmap):
    """Per-module map in the form the installed loader reads, plus the global default."""
    if any(v != QUANT for v in qmap.values()):
        raise ValueError('Quantization map must be uniform affine 4-bit group 64')
    return {'group_size': QUANT['group_size'], 'bits': QUANT['bits'], 'mode': QUANT['mode'],
            **{m: dict(QUANT) for m in sorted(qmap)}}


def finalize_cache(plan, units_dir, cache_dir, source_dir, qmap):
    cache_dir, source_dir = Path(cache_dir), Path(source_dir)
    index = json.loads((cache_dir / WEIGHT_INDEX).read_text())['weight_map']
    expected = set()
    for u in plan['units']:
        if u['action'] == 'quantize':
            expected |= {u['module'] + '.' + p for p in ('weight', 'scales', 'biases')}
        else:
            expected.add(u['normalized_key'])
    if set(index) != expected:
        raise ValueError('Cache index differs from the plan-derived tensor inventory')
    if {u['module'] for u in plan['units'] if u['action'] == 'quantize'} != set(qmap):
        raise ValueError('Quantized modules differ from the saved 344-module map')
    copied = {}
    for p in sorted(source_dir.iterdir()):
        if p.is_file() and p.suffix not in WEIGHT_SUFFIXES and p.name != WEIGHT_INDEX:
            shutil.copyfile(p, cache_dir / (('config_source.json') if p.name == 'config.json' else p.name))
            copied[p.name] = sha_file(p)
    cfg = json.loads((source_dir / 'config.json').read_text())
    if 'quantization' in cfg or 'quantization_config' in cfg:
        raise ValueError('Source config already carries quantization keys')
    qc = quant_config(qmap)
    new = {**cfg, 'quantization': qc, 'quantization_config': qc}
    (cache_dir / 'config.json').write_text(json.dumps(new, indent=2, sort_keys=False))
    return copied


def certify_cache(plan, units_dir, cache_dir, source_dir, qmap):
    """Return a safe-aggregate certificate; raise on any inconsistency."""
    cache_dir, source_dir, units_dir = Path(cache_dir), Path(source_dir), Path(units_dir)
    done = load_done(plan, units_dir)
    if len(done) != len(plan['units']):
        raise ValueError('Not all units have passing receipts')
    index = json.loads((cache_dir / WEIGHT_INDEX).read_text())['weight_map']
    shard_headers, file_sizes = {}, {}
    for shard in sorted(set(index.values())):
        header, base = _read_st_header(cache_dir / shard)
        size = (cache_dir / shard).stat().st_size
        end = 0
        for meta in sorted(header.values(), key=lambda m: m['data_offsets'][0]):
            if meta['data_offsets'][0] != end:
                raise ValueError('Shard tensor ranges are not contiguous: ' + shard)
            end = meta['data_offsets'][1]
        if base + end != size:
            raise ValueError('Shard size differs from its header: ' + shard)
        shard_headers[shard] = (header, base); file_sizes[shard] = size
    checked = 0
    for u in plan['units']:
        rec = done[u['index']]
        uh, ubase = _read_st_header(units_dir / rec['output_file'])
        parts = [('weight', 'weight'), ('scales', 'scales'), ('biases', 'biases')] if u['action'] == 'quantize' else [('tensor', None)]
        for upart, cpart in parts:
            name = (u['module'] + '.' + cpart) if cpart else u['normalized_key']
            ch, cbase = shard_headers[index[name]]
            um, cm = uh[upart], ch[name]
            if um['dtype'] != cm['dtype'] or um['shape'] != cm['shape']:
                raise ValueError('Cache tensor layout differs from its unit: ' + name)
            n = um['data_offsets'][1] - um['data_offsets'][0]
            if n != cm['data_offsets'][1] - cm['data_offsets'][0]:
                raise ValueError('Cache tensor size differs: ' + name)
            with (units_dir / rec['output_file']).open('rb') as a, (cache_dir / index[name]).open('rb') as b:
                a.seek(ubase + um['data_offsets'][0]); b.seek(cbase + cm['data_offsets'][0]); left = n
                while left:
                    k = min(64 << 20, left)
                    if a.read(k) != b.read(k):
                        raise ValueError('Cache payload bytes differ from the verified unit: ' + name)
                    left -= k
            checked += 1
    cfg_src = json.loads((source_dir / 'config.json').read_text())
    cfg_new = json.loads((cache_dir / 'config.json').read_text())
    delta = {k for k in set(cfg_src) | set(cfg_new) if cfg_src.get(k) != cfg_new.get(k)}
    if (delta != {'quantization', 'quantization_config'} or cfg_new['quantization'] != quant_config(qmap)
            or cfg_new['quantization_config'] != quant_config(qmap)):
        raise ValueError('Config differs from the source beyond the exact quantization keys')
    if sha_file(cache_dir / 'config_source.json') != sha_file(source_dir / 'config.json'):
        raise ValueError('config_source.json is not the original config')
    others = {}
    for p in sorted(source_dir.iterdir()):
        if p.is_file() and p.suffix not in WEIGHT_SUFFIXES and p.name not in (WEIGHT_INDEX, 'config.json'):
            if sha_file(cache_dir / p.name) != sha_file(p):
                raise ValueError('Non-weight source file differs in the cache: ' + p.name)
            others[p.name] = sha_file(p)
    return {'plan_sha256': plan['plan_sha256'], 'units': len(plan['units']), 'tensors_checked': checked,
            'quantized_modules': len(qmap), 'retained_tensors': sum(u['action'] == 'retain' for u in plan['units']),
            'shards': {s: {'bytes': file_sizes[s], 'sha256': sha_file(cache_dir / s)} for s in sorted(file_sizes)},
            'config_delta_keys': sorted(delta), 'config_source_sha256': sha_file(cache_dir / 'config_source.json'),
            'config_cached_sha256': sha_file(cache_dir / 'config.json'), 'non_weight_files_identical': others,
            'payload_bytes_equal_to_verified_units': True,
            'scope': 'bytes+lineage only; no model load, no inference; effective-weight loader parity needs the measured cached-load profile'}
