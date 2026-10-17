"""Adapter-agnostic SERIAL greedy development evaluation on the native backend (NOT_RUN until scoped approval).

Same frozen dev manifest, model, prompt rendering (frozen_eval) and generation settings for the starting
and the updated adapter. Each result is fsynced; resume reuses only rows whose full configuration
(model/adapter/manifest/max_tokens/temperature) matches, which also admits identical-contract baseline
rows produced by frozen_eval. Raw text/token ids stay in the private output; stdout is aggregates only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from mri_rl.continuous_rl import require, digest
from mri_rl.experiment_contract import (_check_review, check_artifacts, load_contract, resolve_artifact_paths,
                                        sha256_file)
from mri_rl.frozen_eval import adapter_metadata, prepare_resume
from mri_rl.paced_resources import PacedGate

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('likelihood_cache.py', 'continuous_rl_pilot.py', 'continuous_rl_dev_eval.py', 'mlx_continuous_gspo.py', 'frozen_eval.py', 'likelihood_loading.py',
           'paced_resources.py', 'resource_supervisor.py', 'experiment_contract.py', 'continuous_rl.py')


def source_hashes():
    return {n: sha256_file(ROOT / 'src' / 'mri_rl' / n) for n in SOURCES}


def select_records(manifest, subset_ids):
    records = [json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()]
    require(records and len({r['record_id'] for r in records}) == len(records), 'Valid unique manifest required')
    if subset_ids is None:
        return records
    wanted = [i for i in subset_ids if i]
    require(len(set(wanted)) == len(wanted) and set(wanted) <= {r['record_id'] for r in records},
            'Subset must be unique manifest record ids')
    return [r for r in records if r['record_id'] in set(wanted)]  # manifest order, never score-ordered


def model_inventory(model, contract, contract_path):
    """Validate the supplied model dir against the contract and hash EVERY file (weights, tokenizer,
    processor, template). No native imports. Returns a canonical, reviewable inventory."""
    model = Path(model).resolve()
    require(model.is_dir(), 'Model directory missing')
    paths = resolve_artifact_paths(Path(contract_path))
    for name, filename in (('model_config', 'config.json'), ('processor_config', 'preprocessor_config.json'),
                           ('tokenizer_config', 'tokenizer_config.json'), ('chat_template', 'chat_template.json')):
        require(paths[name].resolve() == (model / filename).resolve(), 'Supplied model dir differs from contract artifacts')
    failures = check_artifacts(contract, paths)
    require(not failures, 'Contract artifact check failed: ' + '; '.join(failures))
    files = sorted(p for p in model.rglob('*') if p.is_file())
    require(any(p.suffix == '.safetensors' for p in files), 'No model weights found')
    return {'model_dir': str(model),
            'files': [{'path': str(p.relative_to(model)), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in files]}


def expected_config(model, manifest, adapter, max_tokens):
    model = Path(model)
    template = model / 'chat_template.json'
    if not template.exists():
        template = model / 'chat_template.jinja'
    return {'model_path': str(model.resolve()), 'model_config_sha256': sha256_file(model / 'config.json'),
            'model_template_sha256': sha256_file(template) if template.exists() else None,
            'template_override_sha256': None, 'manifest_sha256': sha256_file(Path(manifest)),
            'max_tokens': max_tokens, 'temperature': 0, **adapter_metadata(adapter)}


def evaluate(records, backend_factory, output, expected, *, resume, approved_image_sha256,
             approved_output_sha256=None, verify_model=None):
    """Generate for every pending record with one backend; returns aggregate counts only."""
    output = Path(output)
    # Everything below compares against values captured BEFORE review; nothing is re-blessed from disk.
    require(set(approved_image_sha256) >= {r['record_id'] for r in records if r.get('image')},
            'Reviewed image hashes must cover every image record')
    require((digest(output) if output.exists() else None) == approved_output_sha256,
            'Existing output differs from the reviewed starting output')
    output.parent.mkdir(parents=True, exist_ok=True)
    done = prepare_resume(output, records, expected) if resume else set()
    if not resume:
        require(not output.exists(), 'Output exists; use resume to preserve completed work')
    pending = [r for r in records if r['record_id'] not in done]
    shas = {r['record_id']: (approved_image_sha256[r['record_id']] if r.get('image') else None) for r in pending}
    for r in pending:
        require(not r.get('image') or digest(r['image']) == shas[r['record_id']],
                'Image bytes differ from the reviewed hash; refusing before model construction')
    if verify_model is not None and pending:
        verify_model()
    counts = {'manifest_records': len(records), 'reused': len(done), 'generated': 0, 'hit_max_tokens': 0}
    if not pending:
        return counts
    backend = backend_factory()
    try:
        with output.open('a' if resume else 'x') as out:
            for r in pending:
                started = time.monotonic()
                g = backend.generate_record(r, shas[r['record_id']], expected['max_tokens'])
                row = {'record_id': r['record_id'], 'category': r['category'], 'image_sha256': shas[r['record_id']],
                       **expected, 'prompt': r['prompt'], 'engine': 'continuous_rl_dev_eval_native_serial_v1',
                       'elapsed_seconds': round(time.monotonic() - started, 3), **g}
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                out.flush(); os.fsync(out.fileno())
                counts['generated'] += 1; counts['hit_max_tokens'] += int(g['hit_max_tokens'])
    finally:
        backend.close()
    return counts


def main():
    p = argparse.ArgumentParser()
    for n in ('model', 'contract', 'adapter', 'manifest', 'output', 'review-verdict'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--expected-review-sha256', required=True)
    p.add_argument('--subset-ids-file', type=Path)
    p.add_argument('--label', required=True, help='e.g. B28_start or B28_gspo1; bound into the approval scope')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--cache-certificate', type=Path)
    p.add_argument('--cache-certificate-sha256')
    a = p.parse_args()
    for n in ('model', 'contract', 'adapter', 'manifest', 'output', 'review_verdict'):
        setattr(a, n, getattr(a, n).resolve())
    require(a.output.is_relative_to(ROOT / 'local_data'), 'Outputs must stay under ignored local_data')
    for n in ('cache_dir', 'cache_certificate'):
        if getattr(a, n) is not None:
            setattr(a, n, getattr(a, n).resolve())
    from mri_rl.continuous_rl_pilot import cache_binding_for
    a.cache_binding = cache_binding_for(a)             # pure Python, before any native import; None when no cache requested
    cache = {'dir': str(a.cache_dir)} if a.cache_binding else None
    contract = load_contract(a.contract)
    max_tokens = contract.generation['max_tokens']
    subset = a.subset_ids_file.read_text().split() if a.subset_ids_file else None
    records = select_records(a.manifest, subset)
    expected = expected_config(a.model, a.manifest, a.adapter, max_tokens)
    inventory = model_inventory(a.model, contract, a.contract)
    image_hashes = {r['record_id']: digest(r['image']) for r in records if r.get('image')}
    start_output = digest(a.output) if a.output.exists() else None
    require(a.resume or start_output is None, 'Output exists; use --resume to preserve completed work')
    scope = {'execution_tier': 'B', 'kind': 'continuous_rl_dev_eval_v1', 'label': a.label,
             'contract_id': contract.identity(), 'manifest_sha256': expected['manifest_sha256'],
             'subset_record_ids': [r['record_id'] for r in records], 'max_tokens': max_tokens, 'temperature': 0,
             'adapter_checkpoint_sha256': expected['adapter_checkpoint_sha256'],
             'adapter_config_sha256': expected['adapter_config_sha256'],
             'output': str(a.output), 'weight_updates': 0, 'seed': a.seed, 'resume': a.resume,
             'derived_cache': a.cache_binding, 'starting_output_sha256': start_output, 'image_sha256': image_hashes, 'model_inventory': inventory, 'source_sha256': source_hashes()}
    _check_review(review_verdict_path=a.review_verdict, expected_review_sha256=a.expected_review_sha256, scope=scope)
    gate = PacedGate(a.output.with_name(a.output.name + '_resource_events.jsonl'))
    def before_unit(name):
        gate.before_unit(name, loading=is_loading_unit(name, cache))
    from mri_rl.mlx_continuous_gspo import NativeGSPOBackend, is_loading_unit
    config = {'group_size': 2, 'max_tokens': max_tokens}
    counts = evaluate(records, lambda: NativeGSPOBackend(a.adapter, config, model_path=a.model,
                      contract=contract, seed=a.seed, before_unit=before_unit, cache=cache), a.output, expected, resume=a.resume,
                      approved_image_sha256=image_hashes, approved_output_sha256=start_output,
                      verify_model=lambda: require(model_inventory(a.model, contract, a.contract) == inventory,
                                                   'Model artifacts changed after review'))
    print(json.dumps({'label': a.label, **counts, 'output_sha256': digest(a.output) if a.output.exists() else None}))


if __name__ == '__main__':
    main()
