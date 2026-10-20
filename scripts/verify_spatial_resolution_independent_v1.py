"""Independent local TRAIN mask/ontology audit; no network or model imports.

Recalculates the fifteen priority endpoints without importing their builder.
Only aggregate results are written publicly; source keys remain local.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--metadata-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'))
    import numpy as np
    import nibabel as nib
    from scipy import ndimage
    w = Path(__file__).resolve().parents[1]
    report_path = w/'results/SPATIAL_ONTOLOGY_RESOLUTION_V1.json'
    report = load(report_path)
    for name, expected in report['code_sha256'].items():
        path = {'helper': w/'src/mri_rl/spatial_ontology_resolution_v1.py',
                'builder': w/'scripts/build_spatial_ontology_resolution_v1.py',
                'tests': w/'tests/test_spatial_ontology_resolution_v1.py'}[name]
        assert sha(path) == expected
    d = a.candidate.resolve()
    assert d.is_relative_to(w/'local_data') and not a.output.exists()
    for name, expected in report['private_artifact_sha256'].items():
        assert sha(d/name) == expected
    binding = load(d/'binding_PRIVATE.json')
    for name, path in binding['input_paths_PRIVATE'].items():
        assert sha(path) == binding['input_sha256'][name]
    paragraph = ' '.join(''.join(n.itertext()) for n in ET.parse(binding['input_paths_PRIVATE']['paper']).getroot().iter('p'))
    assert '3 – contrast-enhancing tumor' in paragraph and 'openneuro.ds007045.v1.0.1' in paragraph
    trees = []
    for name in ('tree_v101', 'tree_v201'):
        tree = load(binding['input_paths_PRIVATE'][name]); assert tree['truncated'] is False
        trees.append({v['path']: v for v in tree['tree']})
    old, current = trees
    metadata = load(d/'mask_source_lineage_PRIVATE.json')
    assert len(metadata) == 28
    lineage = {}
    for item in metadata:
        ref = item['source_relative_path_PRIVATE']
        pointer_path = a.metadata_root/ref
        pointer = os.readlink(pointer_path)
        payload = pointer.encode()
        blob = hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
        assert current[ref]['mode'] == '120000' and current[ref]['sha'] == blob
        same = old.get(ref, {}).get('mode') == '120000' and old.get(ref, {}).get('sha') == blob
        assert item['lineage']['historical_same_annotation_pointer'] == same
        annex = re.search(r'SHA256E-s(\d+)--([0-9a-f]{64})\.nii\.gz', pointer)
        assert annex and annex[2] == item['mask_sha256'] and int(annex[1]) == Path(item['local_mask_path_PRIVATE']).stat().st_size
        lineage[item['patient_PRIVATE']] = same
    original = {r['id']: r for r in rows(binding['input_paths_PRIVATE']['priority'])}
    amended = rows(d/'spatial_priority_amended_PRIVATE.verl.jsonl')
    assert len(original) == len(amended) == 15
    grouped = defaultdict(list)
    for row in amended:
        oldrow = original[row['id']]
        for field in ('ability', 'prompt', 'reward_model', 'extra_info', 'images'):
            assert row[field] == oldrow[field]
        assert row['training_authorized'] is False
        detail = row['spatial_ontology_resolution_v1']
        assert detail['training_authorized'] is False and detail['scientific_admission'] == 'HOLD'
        assert detail['expert_adjudication'] == 'NOT_COMPLETED'
        grouped[row['extra_info']['patient_key']].append(row)
    bypatient = {m['patient_PRIVATE']: m for m in metadata}
    counts = Counter()
    for key, selected in grouped.items():
        source = bypatient[key]; mask_path = Path(source['local_mask_path_PRIVATE'])
        assert sha(mask_path) == source['mask_sha256']
        image = nib.load(mask_path)
        assert image.shape == (182, 218, 182) and nib.aff2axcodes(image.affine) == ('L', 'A', 'S')
        data = np.asanyarray(image.dataobj)
        assert np.isfinite(data).all() and np.isin(data, (0, 1, 2, 3)).all()
        volumes, components = ndimage.label(data > 0, structure=np.ones((3, 3, 3), np.uint8))
        counts['volumes_multiple_26_components'] += int(components > 1)
        del volumes
        for row in selected:
            z = int(re.search(r'_z(\d+)_', row['id'])[1])
            mask = np.rot90(data[:, :, z]) > 0
            y, x = np.nonzero(mask); n = len(x); h, width = mask.shape
            assert n > 0
            cx = Fraction(2*int(x.sum())+n, 2*n)
            cy = Fraction(2*int(y.sum())+n, 2*n)
            col, rr = int(3*cx//width), int(3*cy//h)
            target = ('left', 'center', 'right')[col]+'-'+('top', 'middle', 'bottom')[rr]
            e = row['spatial_ontology_resolution_v1']['endpoint_PRIVATE']
            assert e['grid'] == target == row['reward_model']['ground_truth']
            assert np.allclose([2*float(cx), 2*float(cy)], e['union_centroid_display_edge_xy'], rtol=0, atol=1e-10)
            assert e['area_native_pixels'] == n
            labels, components2 = ndimage.label(mask, structure=np.ones((3, 3), np.uint8))
            assert components2 == e['components_8'] == 1
            assert mask[int(np.floor(float(cy))), int(np.floor(float(cx)))]
            assert e['boundary_distance_display_pixels'] >= 10
            assert e['visible_focus_center'] == 'NOT_ADJUDICATED'
            counts['rows'] += 1
            counts['paper_release_same_pointer_rows'] += lineage[key]
        counts['mask_decodes'] += 1
        del data
    forms = rows(d/'expert_MRI_only_adjudication_queue_PRIVATE.jsonl')
    assert len(forms) == 84 and sha(d/'expert_MRI_only_adjudication_queue_PRIVATE.jsonl') == binding['input_sha256']['expert']
    assert all(r['expert_adjudication'] == 'NOT_COMPLETED' and
               all(v == [] if k == 'ambiguity_and_abstention_reasons' else v is None
                   for k, v in r['expert_fields_to_complete'].items()) for r in forms)
    assert counts == Counter(rows=15, paper_release_same_pointer_rows=13, mask_decodes=13, volumes_multiple_26_components=7)
    result = {'schema': 'spatial-ontology-independent-review-v1', 'verdict': 'PASS_WITH_LIMITATIONS',
              'reviewer_id': 'Codex /root', 'source_report_sha256': sha(report_path), 'verifier_source_sha256': sha(__file__),
              'candidate_manifest_sha256': sha(d/'spatial_priority_amended_PRIVATE.verl.jsonl'),
              'independent_priority_mask_arithmetic': dict(counts),
              'reviewer_mask_decodes_including_preserved_schema_failed_attempt': 26,
              'reviewer_unique_masks': 13,
              'preserved_verifier_failure': 'Initial blank-field assertion expected None for the existing empty ambiguity list; source/data unchanged, schema check corrected.',
              'all28_public_release_pointer_bindings_checked': True, 'unchanged_paper_release_masks': sum(lineage.values()),
              'new_release_masks_with_subtype_version_semantics_unverified': len(lineage)-sum(lineage.values()),
              'all15_original_targets_prompts_preserved': True, 'all84_expert_forms_identical_and_blank': True,
              'source_numeric3_paper_definition_checked': True, 'network_calls': 0, 'MRI_decodes': 0,
              'models_loaded': 0, 'optimizer_updates': 0, 'final100_reads': 0, 'scientifically_admitted_rows': 0,
              'training_authorized': False,
              'limitations': ['Source-mask centers/components do not establish visible lesion centers or clinical multifocal disease.',
                              'Two new-release annotations still lack exact numeric3 version evidence.',
                              'Expert visual answerability and question/endpoint alignment remain required.']}
    with a.output.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'verdict': result['verdict'], 'sha256': sha(a.output), 'independent_priority_mask_arithmetic': dict(counts)}))


if __name__ == '__main__':
    main()
