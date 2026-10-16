"""Frozen C2 training-cohort PNG controls, exactly preserving the C1 features."""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
C1_SOURCE = ROOT / 'scripts/c1_image_shortcut_baseline.py'
spec = importlib.util.spec_from_file_location('c1shortcut', C1_SOURCE)
c1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c1)
EXPECTED_COUNTS = {'mri_tumor_presence': 32, 'mri_enhancing_presence': 12}
COHORT_SHA = '5786681d14498f4ac2545d858e293a5f12e5ad0cd31234e8c2b6a79ef030b06b'
C1_SHA = 'f88cc384e5d50a152826ee852418fc6b9a3ceb87fc5f9ce4f3204f9e2d03f283'
MISSION = {
    'main_question_link': 'Measure simple image shortcuts before admitting C tasks to Vero-to-MRI recipe transfer.',
    'claim_and_falsifier': 'Simple grayscale content may predict frozen training-cohort labels; report exact accuracy against fixed-no, including negative results. Neither outcome establishes MRI understanding.',
    'change_isolation': 'Apply unchanged C1 16x16 bilinear grayscale and 16-bin histogram 1NN to frozen C2 32/12 cohorts; no tuning, exclusions, new patients or changes to B.',
    'evidence_provenance': 'Bind immutable C2 render receipt, every PNG hash, C1 source and this source; private record/patient/nearest-neighbor receipts.',
    'shortcut_baseline': 'Separate per-category leave-one-patient-out neighbors; fixed no, existing level/site association controls; do not pool overlapping patients across task cohorts.',
    'resources_decision': 'One finite CPU pass over 44 small PNGs, one image at a time and one thread, no MRI volume arrays or model imports. Results only guide later source/site/visibility checks; clinical admission remains unverified.'}


def validate_rows(rows):
    if Counter(r['category'] for r in rows) != EXPECTED_COUNTS:
        raise ValueError('Frozen category counts differ')
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate record IDs')
    for category, n in EXPECTED_COUNTS.items():
        group = [r for r in rows if r['category'] == category]
        if len({r['patient'] for r in group}) != n:
            raise ValueError('Each category requires distinct patients')
        if Counter(r['label'] for r in group) != {'yes': n//2, 'no': n//2}:
            raise ValueError('Frozen class balance differs')


def extract_features(path, expected_sha):
    if c1.sha(path) != expected_sha:
        raise ValueError('Image hash changed')
    with Image.open(path) as image:
        if image.format != 'PNG' or image.width*image.height > 1_000_000:
            raise ValueError('Only bounded PNG frames accepted')
        gray = image.convert('L')
        thumbnail = np.asarray(gray.resize((16,16), Image.Resampling.BILINEAR), dtype=np.float64).ravel()/255
        h = np.histogram(np.asarray(gray), bins=16, range=(0,256))[0]
        return thumbnail, h/h.sum()


def nearest_neighbors(rows, features):
    neighbors = []
    for i, row in enumerate(rows):
        candidates = [j for j,r in enumerate(rows) if r['patient'] != row['patient']]
        if not candidates:
            raise ValueError('No other-patient neighbor')
        neighbors.append(min(candidates, key=lambda j: (float(np.sum((features[i]-features[j])**2)), rows[j]['record_id'])))
    # The original C1 implementation is authoritative; this adds neighbor evidence.
    if [rows[j]['label'] for j in neighbors] != c1.lopo_predictions(rows, features):
        raise ValueError('C1 prediction mismatch')
    return neighbors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    if not output.is_relative_to(ROOT/'local_data') or output.exists():
        raise ValueError('New private output directory required')
    if c1.sha(args.cohort) != COHORT_SHA or c1.sha(C1_SOURCE) != C1_SHA:
        raise ValueError('Frozen input/source hash mismatch')
    caps = {key: os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
    if any(v != '1' for v in caps.values()):
        raise ValueError('Explicit single-thread caps required')
    rows = json.loads(args.cohort.read_text())['rows_PRIVATE']
    validate_rows(rows)
    provenance = {'cohort_sha256': COHORT_SHA, 'c1_source_sha256': C1_SHA,
                  'source_sha256': c1.sha(__file__), 'numpy_version': np.__version__,
                  'pillow_version': pillow_version, 'threadcaps': caps}
    output.mkdir()
    (output/'scope_PRIVATE.json').write_text(json.dumps({'mission_alignment': MISSION, **provenance}, indent=2)+'\n')
    result = {'status': 'TESTED_CPU', 'evidence': 'OBSERVED', 'tag': 'EXPLORATORY_DIAGNOSTIC',
              'scope_sha256': c1.sha(output/'scope_PRIVATE.json'), **provenance,
              'interpretation': 'Training-cohort shortcut diagnostic; not held-out generalization or clinical admission.',
              'rows': len(rows), 'patients': len({r['patient'] for r in rows}),
              'categories': {}}
    for category in EXPECTED_COUNTS:
        group = [r for r in rows if r['category'] == category]
        pairs = [extract_features(Path(r['image_PRIVATE']), r['image_sha256']) for r in group]
        methods = {}
        for k, name in enumerate(('thumbnail16','histogram16')):
            features = np.array([p[k] for p in pairs])
            neighbors = nearest_neighbors(group, features)
            predictions = []
            for i,j in enumerate(neighbors):
                row, neighbor = group[i], group[j]
                predictions.append({'record_id': row['record_id'], 'patient': row['patient'],
                    'image_sha256': row['image_sha256'], 'gold': row['label'],
                    'prediction': neighbor['label'], 'neighbor_record_id': neighbor['record_id'],
                    'neighbor_patient': neighbor['patient'], 'neighbor_image_sha256': neighbor['image_sha256'],
                    'squared_distance': float(np.sum((features[i]-features[j])**2))})
            methods[name] = {'correct': sum(p['gold']==p['prediction'] for p in predictions), 'n': len(group),
                'confusion': {f'{gold}->{guess}': sum(p['gold']==gold and p['prediction']==guess for p in predictions)
                              for gold in ('no','yes') for guess in ('no','yes')},
                'same_patient_neighbor_count': sum(p['patient']==p['neighbor_patient'] for p in predictions),
                'predictions_PRIVATE': predictions}
        result['categories'][category] = methods
    receipt = output/'receipt_PRIVATE.json'
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    public = {**result, 'receipt_sha256': c1.sha(receipt)}
    for methods in public['categories'].values():
        for method in methods.values():
            method.pop('predictions_PRIVATE')
    print(json.dumps(public, indent=2))


if __name__ == '__main__':
    main()
