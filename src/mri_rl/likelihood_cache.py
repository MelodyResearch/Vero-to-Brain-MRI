"""Exact derived-cache binding and loaded-parameter identity for the read-only likelihood runner.

Execution-only: the frozen prompts, pixels, candidate labels, scoring loop and precision are untouched. The cache
(quant_convert/quant_cache) holds the SAME effective 4-bit weights as the original base; this module (1) binds the
certified cache by hash BEFORE any native import and (2) after the installed loader has built the model from it,
compares every one of the 1438 loaded base parameters byte-for-byte with the certified cache payload.
"""
import hashlib
import json
import struct
from pathlib import Path

EXPECTED_TENSORS = 1438
WEIGHT_INDEX = 'model.safetensors.index.json'


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _header(path):
    with Path(path).open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return {k: v for k, v in json.loads(f.read(n)).items() if k != '__metadata__'}, 8 + n


def verify_cache_identity(cache_dir, certificate_path, expected_certificate_sha256, model_artifacts):
    """Pure-Python binding. model_artifacts: {path: sha256} from the frozen contract (original model files)."""
    cache, cert_path = Path(cache_dir), Path(certificate_path)
    if _sha(cert_path) != expected_certificate_sha256:
        raise ValueError('Cache certificate differs from the approved digest')
    cert = json.loads(cert_path.read_text())
    if cert.get('tensors_checked') != EXPECTED_TENSORS or cert.get('payload_bytes_equal_to_verified_units') is not True:
        raise ValueError('Cache certificate does not certify the full 1438-tensor payload')
    index = json.loads((cache / WEIGHT_INDEX).read_text())['weight_map']
    if len(index) != EXPECTED_TENSORS or set(index.values()) != set(cert['shards']):
        raise ValueError('Cache index differs from the certificate')
    for shard, meta in cert['shards'].items():
        p = cache / shard
        if p.stat().st_size != meta['bytes'] or _sha(p) != meta['sha256']:
            raise ValueError('Cache shard differs from the certificate: ' + shard)
    if _sha(cache / 'config.json') != cert['config_cached_sha256'] or _sha(cache / 'config_source.json') != cert['config_source_sha256']:
        raise ValueError('Cache config differs from the certificate')
    originals = {Path(p).name: d for p, d in model_artifacts.items()}
    if originals.get('config.json') != cert['config_source_sha256']:
        raise ValueError('Cache source config is not the frozen original model config')
    src, new = json.loads((cache / 'config_source.json').read_text()), json.loads((cache / 'config.json').read_text())
    delta = {k for k in set(src) | set(new) if src.get(k) != new.get(k)}
    if delta != {'quantization', 'quantization_config'} or new['quantization'] != new['quantization_config']:
        raise ValueError('Cache config differs from the original beyond the identical quantization keys')
    for name, digest in cert['non_weight_files_identical'].items():
        if _sha(cache / name) != digest or (name in originals and originals[name] != digest):
            raise ValueError('Cache non-weight file differs from the frozen original: ' + name)
    return {'certificate_sha256': expected_certificate_sha256, 'plan_sha256': cert['plan_sha256'],
            'shards': cert['shards'], 'config_cached_sha256': cert['config_cached_sha256'],
            'config_source_sha256': cert['config_source_sha256'], 'tensors': EXPECTED_TENSORS,
            'quantized_modules': cert['quantized_modules'], 'cache_dir': str(cache.resolve())}


def verify_loaded_parameters(named_parameters, cache_dir, to_bytes):
    """Compare every loaded base parameter with its certified cache bytes.

    to_bytes(value) -> (raw_bytes, dtype_label, shape) with dtype_label in {'BF16','U32'}.
    Returns the number compared; raises on any missing, extra, or differing parameter."""
    cache = Path(cache_dir)
    index = json.loads((cache / WEIGHT_INDEX).read_text())['weight_map']
    headers = {}
    seen = set()
    for name, value in named_parameters:
        if name not in index:
            raise ValueError('Loaded parameter absent from the certified cache: ' + name)
        shard = index[name]
        if shard not in headers:
            headers[shard] = _header(cache / shard)
        header, base = headers[shard]
        meta = header[name]
        raw, dtype, shape = to_bytes(value)
        if dtype != meta['dtype'] or list(shape) != meta['shape']:
            raise ValueError('Loaded parameter layout differs from the cache: ' + name)
        begin, end = meta['data_offsets']
        with (cache / shard).open('rb') as f:
            f.seek(base + begin)
            if len(raw) != end - begin or f.read(end - begin) != raw:
                raise ValueError('Loaded parameter bytes differ from the certified cache: ' + name)
        seen.add(name)
    if seen != set(index):
        raise ValueError('Loaded model lacks %d certified cache tensors' % len(set(index) - seen))
    return len(seen)
