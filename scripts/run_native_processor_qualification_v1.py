#!/usr/bin/env python3
"""CPU-only: qualify the pinned native Qwen3-VL processor on the 56 TRAIN rows of the DATA56 capsule. Counts-only stdout.

No weights, no MLX, no torch, no network, single thread. Streams one image at a time. The capsule is consumed by this LOCAL
program only (whole-file SHA pinned); details go to --private-out, aggregates to --safe-out. Failures print only the
exception TYPE and write the full message privately (no traceback with paths).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np                                                  # noqa: E402
from mri_rl import native_processor_qualification_v1 as q           # noqa: E402

CAPSULE_SHA256 = '1c2791d45e38759212766fc3d3553e1e1523dbc1b87167bf595171b515c986b9'


def forbidden_for(entry, record):
    names = {'patient_key': entry.get('patient_key_PRIVATE'), 'site': entry.get('site_PRIVATE'), 'source_record_id': entry.get('original_source_record_id_PRIVATE'),
             'opaque_id': entry.get('opaque_id'), 'file_name': Path(record['images'][0]).name, 'file_stem': Path(record['images'][0]).stem, 'row_id': record.get('id')}
    return {k: v for k, v in names.items() if v}


def run(a):
    raw = a.capsule.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CAPSULE_SHA256:
        return {'status': 'BLOCKED_CAPSULE_SHA_MISMATCH'}, 4
    capsule = json.loads(raw)
    entries = capsule['rows']
    rows = [e['record_PRIVATE'] for e in entries]
    if len(rows) != 56 or len({r['id'] for r in rows}) != 56:
        return {'status': 'BLOCKED_ROW_INVENTORY', 'rows': len(rows)}, 5
    joined = q.join_capsule(rows, {'rows': {e['record_PRIVATE']['id']: e for e in entries}})
    if not joined['ok']:
        return {'status': 'BLOCKED_CAPSULE_JOIN_GAP', 'rows_missing_keys': len(joined['missing_keys']), 'reason': joined.get('reason')}, 3
    contract_raw = a.contract.read_bytes()
    contract = json.loads(contract_raw)
    capsule_rows, binding = {}, []
    for e, r in zip(entries, rows):
        typed = [p for m in r['prompt'] for p in m['content'] if isinstance(p, dict) and p.get('type') == 'image']
        texts = [p.get('text') for m in r['prompt'] for p in m['content'] if isinstance(p, dict) and p.get('type') == 'text']
        typed_path = next((p.get(k) for p in typed for k in ('image', 'path', 'url') if isinstance(p.get(k), str)), None)
        binding.append({'one_typed_image': len(typed) == 1, 'typed_equals_top': typed_path == r['images'][0] and len(r['images']) == 1,
                        'typed_text_equals_extra_info_question': texts == [r['extra_info']['question']],
                        'file_sha_equals_expected_pin': q.sha256_file(r['images'][0]) == e['expected_png_sha256'],
                        'split_is_train': str(e['split']).lower() == 'train',
                        'expected_label_equals_reward_model': e['expected_label_PRIVATE'] == r['reward_model']['ground_truth']})
        capsule_rows[r['id']] = {'forbidden': forbidden_for(e, r), 'pair': e['pair_key']}
    processor = q.load_pinned_processor(a.model)
    result = q.qualify_rows(processor, rows, capsule_rows, system_instruction=contract['system_instruction'], assistant_prefix=contract['assistant_prefix'],
                            enable_thinking=contract['enable_thinking'], model_config=json.loads((a.model / 'config.json').read_text()))
    for r, b in zip(rows, binding):
        rec = next(x for x in result['per_row'] if x['row_id'] == r['id'])
        rgb = np.asarray(q.load_rgb(r['images'][0]))
        rec['decoded_rgb_sha256_CURRENT'] = q.sha256_array(rgb)
        rec['binding'] = b
        rec['ok'] = bool(rec['ok'] and all(b.values()))
    result['ok'] = bool(result['ok'] and all(x['ok'] for x in result['per_row']))
    agg = q.aggregate(result['per_row'], expected_rows=56)
    per = result['per_row']
    safe = {'schema': 'native-processor-qualification-cpu-v1', 'status': q.STATUS, 'ok': result['ok'], **agg, 'prompt_text_variants': result['prompt_text_variants'],
            'swap_pairs': result['swap_pairs'], 'swap_pixel_changes': result['swap_pixel_changes'], 'swap_text_unchanged': result['swap_text_unchanged'],
            'image_pad_tokens_distinct': sorted({x['image_pad_tokens'] for x in per}), 'visual_tokens_independent_distinct': sorted({x['visual_tokens_independent'] for x in per}),
            'prompt_length_min_max': [min(x['prompt_length'] for x in per), max(x['prompt_length'] for x in per)], 'grid_distinct': sorted([list(g) for g in {tuple(x['image_grid_thw']) for x in per}]),
            'source_size_wh_distinct': sorted([list(g) for g in {tuple(x['source_size_wh']) for x in per}]),
            'pixel_std_min': min(x['pixel_std'] for x in per), 'unique_pixel_sha256': len({x['pixel_sha256'] for x in per}), 'unique_decoded_rgb_sha256': len({x['decoded_rgb_sha256_CURRENT'] for x in per}),
            'leak_hits_total': sum(len(x['leak_hits']) for x in per), 'binding_checks_all_true': {k: sum(x['binding'][k] for x in per) for k in per[0]['binding']},
            'capsule_whole_file_sha256': CAPSULE_SHA256, 'contract_sha256': hashlib.sha256(contract_raw).hexdigest(), 'pinned_sources': q.pinned_sources(),
            'decoded_rgb_hash_note': 'CURRENT decode recorded once by this program; NOT an independent historical pixel re-verification (inherited pixel-source audit + PNG byte pin is the basis).',
            'backend_f32_model_inputs': 'NOT_RUN', 'models_loaded': 0, 'mlx_imported': 'mlx' in sys.modules, 'torch_imported': 'torch' in sys.modules,
            'threads': os.environ.get('OMP_NUM_THREADS'), 'evidence_tier': 'OBSERVED_CPU', 'cross_model_review': 'NOT_VERIFIED'}
    with a.private_out.open('x') as f:
        json.dump(result, f, sort_keys=True, allow_nan=False)
    with a.safe_out.open('x') as f:
        json.dump(safe, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    return {k: safe[k] for k in ('ok', 'rows', 'rows_ok', 'prompt_text_variants', 'swap_pairs', 'swap_pixel_changes', 'swap_text_unchanged', 'leak_hits_total',
                                 'image_pad_tokens_distinct', 'unique_pixel_sha256', 'binding_checks_all_true')}, (0 if safe['ok'] else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capsule', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--private-out', type=Path, required=True)
    p.add_argument('--safe-out', type=Path, required=True)
    a = p.parse_args()
    try:
        out, code = run(a)
    except Exception as error:      # no traceback to stdout: it can carry private paths
        err = a.private_out.with_suffix('.error_PRIVATE.txt')
        err.write_text(traceback.format_exc())
        out, code = {'status': 'FAILED', 'error_type': type(error).__name__}, 2
    print(json.dumps(out, sort_keys=True))
    return code


if __name__ == '__main__':
    sys.exit(main())
