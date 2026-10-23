#!/usr/bin/env python3
"""CPU-only: measure the expanded prompt length of every frozen heterogeneous-pilot midline model input. Counts-only stdout.

Uses the existing torch-free native_processor_qualification_v1 helper (pinned Qwen3VLProcessor from file, pinned mlx_vlm
prompt_utils formatter, numpy instead of mx). No weights, no MLX, no model, no network, no AutoProcessor. Rows come from the
pilot's own CPU `prepare` (exact frozen suffix, 2 TRAIN + 2 SWAP + 2 BORDER + 2 EVAL); the two frozen V2 concise sequence TRAIN rows
are re-measured as a method check and must give the executed V2 length 233. The native path re-verifies every length at run time.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np                                                      # noqa: E402
from mri_rl import heterogeneous_mri_pilot_entry_v1 as entry            # noqa: E402
from mri_rl import heterogeneous_mri_pilot_v1 as hp                     # noqa: E402
from mri_rl import native_processor_qualification_v1 as q               # noqa: E402
from mri_rl.shared_rollout_gspo_core_v1 import publish_once             # noqa: E402


def length(processor, formatter, config, contract, row, image_sha256):
    image = Path(row['images'][0])
    entry.require(q.sha256_file(image) == image_sha256, 'Image differs from its bound SHA')
    prompt = q.render(formatter, processor, config, contract['system_instruction'], row['extra_info']['question'], enable_thinking=contract['enable_thinking'])
    entry.require(prompt.endswith(contract['assistant_prefix']), 'Rendered prompt prefix differs from the contract')
    out = q.run_processor(processor, prompt, q.load_rgb(image))
    entry.require(bool(np.all(out['attention_mask'] == 1)), 'Unpadded single prompt expected')
    return int(out['input_ids'].shape[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    prepared = entry.prepare(a.spec)
    entry.require(prepared['status'] != 'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN', 'A measured scope already exists; measurement is once-only')
    out = entry._inside(a.output, entry.PILOT_DIR)
    entry.require(not out.exists(), 'Measurement output already exists (once-only)')
    v2 = entry.load(entry.V2_SPEC); runtime = entry.load(v2['runtime_spec']); contract = entry.load(runtime['contract'])
    model = Path(runtime['model']); config = json.loads((model / 'config.json').read_text())
    processor, formatter = q.load_pinned_processor(model), q.load_formatter()
    rows = prepared['measurement_rows_PRIVATE']
    by_row = {rid: length(processor, formatter, config, contract, r, r['image_sha256']) for rid, r in sorted(rows.items())}
    pin = prepared['pilot_input_PRIVATE']
    seq = [length(processor, formatter, config, contract, c['row'], pin['sequence_image_sha256'][c['row']['id']]) for c in pin['sequence_unit_PRIVATE']['contexts']]
    record = {'schema': entry.MEASUREMENT_SCHEMA, 'rows_identity': entry.rows_identity(rows), 'midline_suffix_sha256': hp.MIDLINE_SUFFIX_SHA256,
              'by_row': by_row, 'sequence_concise_check': {'expected': entry.V2_CONCISE_SEQUENCE_LENGTH, 'actual': seq},
              'source_qa_sha256': entry.PINNED['source_qa'], 'midline_manifest_sha256': entry.PINNED['midline_manifest'],
              'pinned_sources': q.pinned_sources(), 'contract_sha256': q.sha256_file(runtime['contract']), 'model_calls': 0,
              'mlx_imported': 'mlx' in sys.modules, 'torch_imported': 'torch' in sys.modules, 'method': 'native_processor_qualification_v1 CPU helper'}
    entry.require(not record['mlx_imported'], 'MLX was imported')
    publish_once(out, record)
    print(json.dumps({'rows': len(by_row), 'lengths_sorted': sorted(by_row.values()), 'sequence_concise_check': record['sequence_concise_check'],
                      'mlx_imported': record['mlx_imported'], 'torch_imported': record['torch_imported'], 'model_calls': 0}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
