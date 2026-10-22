#!/usr/bin/env python3
"""Offline CPU binding of the five preselected public-source items; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mri_rl.vqarad_midline_task_v1 import build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior evidence: output must be new')
    d = args.private_source_dir
    raw = (d / 'vqarad_original_PUBLIC_SOURCE_PRIVATE.json').read_bytes()
    metadata = json.loads((d / 'osf_list_PUBLIC_SOURCE_PRIVATE.txt').read_text())
    entry = next(x for x in metadata['data'] if x['attributes']['name'] == 'VQA_RAD Dataset Public.json')
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry['attributes']['extra']['hashes']['sha256']:
        raise ValueError('Original QA JSON differs from official OSF SHA')
    files = []
    for name in ['vqarad_image_list', 'vqarad_image_list_2', 'vqarad_image_list_3', 'vqarad_image_list_4']:
        files.extend(json.loads((d / (name + '_PUBLIC_SOURCE_PRIVATE.txt')).read_text())['data'])
    if len(files) != 315 or len({x['attributes']['name'] for x in files}) != 315:
        raise ValueError('Expected complete four-page official image index')
    hashes = {x['attributes']['name']: x['attributes']['extra']['hashes']['sha256'] for x in files}
    plan = [{'qid': '52', 'split': 'train'}, {'qid': '69', 'split': 'train'},
            {'qid': '1037', 'split': 'eval'}, {'qid': '2236', 'split': 'eval'},
            {'qid': '1050', 'split': 'reserve'}]
    result = build_manifest(json.loads(raw), plan, d / 'vqarad_five_PRIVATE', hashes)
    result['source_json_sha256'] = digest
    result['source_json_url'] = 'https://osf.io/download/6qdas/'
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'CPU_BINDING_PASS_ONLY', 'counts': result['counts'],
                      'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'gpu_calls': 0, 'optimizer_updates': 0}))


if __name__ == '__main__':
    main()
