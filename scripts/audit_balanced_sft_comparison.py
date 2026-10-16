#!/usr/bin/env python3
"""CPU-only private audit and matched scoring; never loads a model or test labels.

Run from repo with PYTHONPATH=src python scripts/audit_balanced_sft_comparison.py
--root local_data/ds007045_bounded --output-dir local_data/... [--balanced-outputs ...].
All patient-linked outputs are restricted to ignored local_data.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from mri_rl.ds007045_internal_dev_eval import score_outputs, _keyed_jsonl
from mri_rl.mlx_sft_warmstart import CATEGORIES, _schedule_train_rows
from mri_rl.vero_mri_reward import _answer_and_format, _single_box_anywhere_answer, compute_score_from_data_source


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def majority(values):
    counts = Counter(values)
    return min(counts, key=lambda label: (-counts[label], label))


def level_baselines(train, dev):
    """Fit only the 28 internal training patients; fixed lexical tie break."""
    result = {}
    for task in CATEGORIES:
        training = [r for r in train if r['ability'] == task]
        testing = [r for r in dev if r['ability'] == task]
        groups = defaultdict(list)
        for r in training:
            groups[re.search(r'_z(\d+)_', r['id']).group(1)].append(r['reward_model']['ground_truth'])
        rules = {level: majority(labels) for level, labels in groups.items()}
        prior = majority([r['reward_model']['ground_truth'] for r in training])
        predictions = [rules[re.search(r'_z(\d+)_', r['id']).group(1)] for r in testing]
        result[task] = {'train_label_counts': dict(Counter(r['reward_model']['ground_truth'] for r in training)),
                        'train_level_counts': {z: dict(Counter(v)) for z, v in groups.items()},
                        'level_rule': rules, 'global_majority_label': prior, 'dev_n': len(testing),
                        'level_correct': sum(p == r['reward_model']['ground_truth'] for p, r in zip(predictions, testing)),
                        'majority_correct': sum(prior == r['reward_model']['ground_truth'] for r in testing)}
    return result


def audit_schedule(root):
    train_path = root / 'vero_training_v1/train.verl.jsonl'
    dev_path = root / 'vero_training_v1/internal_dev.verl.jsonl'
    train, dev = rows(train_path), rows(dev_path)
    cases = json.loads((root / 'visual_prompts/private_manifest.json').read_text())['private_cases']
    sets = {'train': {r['extra_info']['patient_key'] for r in train},
            'internal_dev': {r['extra_info']['patient_key'] for r in dev},
            'historical_test': {c['anonymous_token'] for c in cases if c['split'] == 'test'}}
    require([len(sets[k]) for k in sets] == [28, 4, 8], 'Wrong patient counts')
    require(not any(a & b for a, b in itertools.combinations(sets.values(), 2)), 'Patient overlap')
    require(sets['train'] | sets['internal_dev'] == {c['anonymous_token'] for c in cases if c['split'] == 'train'}, 'Source training partition mismatch')
    require(len(train) == 382 and len(dev) == 54, 'Wrong record counts')
    require(len({r['id'] for r in train + dev}) == 436, 'Repeated record IDs')
    for split, records in [('train', train), ('internal_dev', dev)]:
        require(all(r['extra_info']['internal_split'] == split and r['extra_info']['patient_lineage'] == 'BIDS_SUBJECT_VERIFIED' for r in records), 'Split/lineage mismatch')
    legacy = _schedule_train_rows(train_path, 28, 7, 'legacy_lexical_v1')
    balanced = _schedule_train_rows(train_path, 28, 7, 'balanced_sequence_v2')
    # Independent reconstruction, before comparing with the production scheduler.
    offset = int(hashlib.sha256(b'7:sequence-label-offset').hexdigest(), 16) % 2
    reconstructed = {}
    for category in CATEGORIES:
        by_patient = defaultdict(list)
        for row in train:
            if row['ability'] == category:
                by_patient[row['extra_info']['patient_key']].append(row)
        ordered = sorted(by_patient, key=lambda p: hashlib.sha256(f'7:{category}:{p}'.encode()).hexdigest())
        selected = []
        for index, patient in enumerate(ordered):
            options = by_patient[patient]
            if category == 'mri_sequence':
                label = ('FLAIR', 'contrast-enhanced T1')[(index + offset) % 2]
                options = [r for r in options if r['reward_model']['ground_truth'] == label]
            selected.append(sorted(options, key=lambda r: r['id'])[0])
        reconstructed[category] = selected
    historical = json.loads((root / 'sft_warmstart_28step_v1/receipt.json').read_text())
    historical_examples = historical['examples']
    require(len(historical_examples) == 112, 'Historical example count mismatch')
    inventory = []
    for step, (old_batch, new_batch) in enumerate(zip(legacy, balanced)):
        for category_index, (old, new) in enumerate(zip(old_batch, new_batch)):
            category = new['ability']
            require(new == reconstructed[category][step], 'Independent schedule differs')
            require(old['extra_info']['patient_key'] == new['extra_info']['patient_key'], 'Patient ordering changed')
            evidence = historical_examples[step * 4 + category_index]
            require(evidence['step'] == step and evidence['category'] == category and evidence['patient_key'] == old['extra_info']['patient_key'], 'Historical receipt mismatch')
            if category != 'mri_sequence':
                require(old == new, 'Nonsequence example changed')
            inventory.append({'step': step + 1, 'category': category, 'patient_key': new['extra_info']['patient_key'],
                              'record_id': new['id'], 'label': new['reward_model']['ground_truth'],
                              'image_sha256': digest(new['images'][0]), 'legacy_record_id': old['id'],
                              'legacy_label': old['reward_model']['ground_truth'], 'legacy_image_sha256': digest(old['images'][0])})
    sequence_counts = dict(Counter(r['label'] for r in inventory if r['category'] == 'mri_sequence'))
    require(sequence_counts == {'FLAIR': 14, 'contrast-enhanced T1': 14}, 'Sequence balance failure')
    return {'status': 'TESTED_CPU', 'patient_counts': {k: len(v) for k, v in sets.items()},
            'patient_keys_PRIVATE': {k: sorted(v) for k, v in sets.items()}, 'train_records_sha256': digest(train_path),
            'dev_records_sha256': digest(dev_path), 'examples': inventory, 'sequence_counts': sequence_counts,
            'nonsequence_equal': 84, 'changed_sequence_records': sum(r['record_id'] != r['legacy_record_id'] for r in inventory),
            'historical_record_reconstruction_limit': 'Historical receipt saves category/patient/step but not record/image hashes; exact legacy IDs reconstructed from pinned manifests and lexical selection.',
            'baseline_fit_patients': 28, 'baselines': level_baselines(train, dev), 'historical_test_labels_used': False}


def detailed_score(manifest, gold_path, output_path):
    validated = score_outputs(manifest, gold_path, output_path)
    gold, outputs = _keyed_jsonl(gold_path), _keyed_jsonl(output_path)
    tasks, patients, outcomes = {}, defaultdict(list), {}
    for key, row in outputs.items():
        target = gold[key]
        task = target['category']
        grade = compute_score_from_data_source('OpenNeuro_ds007045', row['response'], target['ground_truth'], target['extra_info'])
        answer, formatting = _answer_and_format(row['response'])
        labels = ({'flair', 'contrast-enhanced t1'} if task == 'mri_sequence' else
                  {'yes', 'no'} if task in {'mri_tumor_presence', 'mri_enhancing_presence'} else
                  {f'{x}-{y}' for x in ('left', 'center', 'right') for y in ('top', 'middle', 'bottom')})
        prediction = answer.lower() if answer.lower() in labels else 'INVALID'
        item = tasks.setdefault(task, {'n': 0, 'correct': 0, 'strict_correct': 0, 'format_valid': 0, 'invalid_answers': 0, 'confusion': {}, 'prediction_counts': Counter(), 'semantic_single_box_counts': Counter(), 'characters': []})
        item['n'] += 1
        item['correct'] += int(grade['accuracy'] > 0)
        item['strict_correct'] += int(grade['accuracy'] > 0 and formatting == 1)
        item['format_valid'] += int(formatting == 1)
        item['invalid_answers'] += int(prediction == 'INVALID')
        item['prediction_counts'][prediction] += 1
        semantic = _single_box_anywhere_answer(row['response'], task).lower() or 'INVALID'
        item['semantic_single_box_counts'][semantic] += 1
        item['confusion'].setdefault(target['ground_truth'].lower(), Counter())[prediction] += 1
        item['characters'].append(len(row['response']))
        correct = int(grade['accuracy'] > 0)
        outcomes[key] = {'patient_key': target['patient_key'], 'task': task, 'correct': correct, 'prediction': prediction, 'ground_truth': target['ground_truth']}
        patients[target['patient_key']].append(correct)
    for task, item in tasks.items():
        item['accuracy'] = item['correct'] / item['n']
        item['balanced_accuracy'] = mean(counts.get(label, 0) / sum(counts.values()) for label, counts in item['confusion'].items())
        item['max_class_prediction_fraction'] = max(item['prediction_counts'].values()) / item['n']
        lengths = item.pop('characters')
        item['response_characters'] = {'min': min(lengths), 'mean': mean(lengths), 'max': max(lengths)}
    return {**validated, 'by_category': tasks, 'equal_task_macro_accuracy': mean(t['accuracy'] for t in tasks.values()),
            'worst_task_accuracy': min(t['accuracy'] for t in tasks.values()),
            'patient_outcomes_PRIVATE': {p: {'correct': sum(v), 'n': len(v), 'accuracy': mean(v)} for p, v in patients.items()},
            'response_tokens': 'NOT_MEASURED: historical output receipt does not save generated token counts',
            'truncation': 'NOT_MEASURED: historical output receipt does not save finish reason; response syntax is not a valid substitute',
            'outcomes_PRIVATE': outcomes, 'outputs_sha256': digest(output_path)}


def compare(manifest, gold_path, named_outputs):
    summaries = {name: detailed_score(manifest, gold_path, path) for name, path in named_outputs.items()}
    contract_fields = ('model_config_sha256', 'model_template_sha256', 'max_tokens', 'temperature', 'adapter_config_sha256')
    configurations = {tuple(rows(path)[0].get(k) for k in contract_fields) for path in named_outputs.values()}
    require(len(configurations) == 1, 'Cross-model evaluation settings mismatch')
    paired = {}
    for a, b in itertools.combinations(summaries, 2):
        aa, bb = summaries[a]['outcomes_PRIVATE'], summaries[b]['outcomes_PRIVATE']
        require(set(aa) == set(bb), 'Paired record mismatch')
        by_patient = defaultdict(lambda: {'n': 0, 'a_correct': 0, 'b_correct': 0, 'gained': 0, 'lost': 0})
        for key in aa:
            x, y = aa[key], bb[key]
            p = by_patient[x['patient_key']]
            p['n'] += 1
            p['a_correct'] += x['correct']
            p['b_correct'] += y['correct']
            p['gained'] += x['correct'] == 0 and y['correct'] == 1
            p['lost'] += x['correct'] == 1 and y['correct'] == 0
        for p in by_patient.values():
            p['accuracy_difference_b_minus_a'] = (p['b_correct'] - p['a_correct']) / p['n']
        paired[f'{a}__vs__{b}'] = {'patient_outcomes_PRIVATE': dict(by_patient),
                                  'patient_mean_difference_b_minus_a': mean(p['accuracy_difference_b_minus_a'] for p in by_patient.values()),
                                  'uncertainty': 'Four development patients: descriptive paired results only, no confirmatory inference.'}
    return {'status': 'TESTED_CPU', 'models': summaries, 'paired': paired, 'comparison_settings_equal': True, 'heldout_test_used': False}


def prospective_metadata(root, metadata):
    """Read only committed metadata and annex pointers, never MRI/mask payloads."""
    commit = '283007b96977b5b3285bd22cd67b7d97114b1eed'
    def git(*args):
        return subprocess.check_output(['git', '-C', str(metadata), *args], text=True)
    require(git('rev-parse', 'HEAD').strip() == commit, 'Metadata revision mismatch')
    paths = set(git('ls-tree', '-r', '--name-only', commit).splitlines())
    participants = git('show', f'{commit}:participants.tsv').splitlines()
    subjects = [line.split('\t')[0] for line in participants[1:] if line.strip()]
    require(len(subjects) == len(set(subjects)) == 363, 'Unexpected metadata population')
    existing = {r['private_subject'] for r in json.loads((root / 'visual_prompts/private_manifest.json').read_text())['private_cases']}
    candidates = []
    for subject in subjects:
        if subject in existing:
            continue
        payload_paths = [f'derivatives/processing_MNI152_skull-stripped_N4/processing_MNI152_skull-stripped_N4/{subject}/{subject}_{modality}_MNI152_brain_n4.nii.gz' for modality in ('flair', 't1ce')]
        payload_paths.append(f'derivatives/segmentation/{subject}/{subject}_MNI152_tumor-mask.nii.gz')
        require(all(p in paths for p in payload_paths + [f'{subject}/anat/{subject}_FLAIR.json', f'{subject}/anat/{subject}_run-02_T1w.json']), 'Missing eligibility metadata')
        payloads = []
        for path in payload_paths:
            pointer = git('show', f'{commit}:{path}')
            match = re.search(r'SHA256E-s(\d+)--([a-f0-9]{64})', pointer)
            require(match is not None, 'Expected annex pointer, refusing image read')
            payloads.append({'path': path, 'bytes': int(match.group(1)), 'sha256': match.group(2)})
        site = re.match(r'sub-([A-Z]+)', subject).group(1)
        candidates.append({'subject_PRIVATE': subject, 'site': site, 'payloads': payloads,
                           'selection_hash': hashlib.sha256(f'ds007045-prospective-site2-v1:{commit}:{subject}'.encode()).hexdigest()})
    require(len(candidates) == 323, 'Unexpected reserve size')
    selected = []
    for site in sorted({c['site'] for c in candidates}):
        selected.extend(sorted((c for c in candidates if c['site'] == site), key=lambda c: (c['selection_hash'], c['subject_PRIVATE']))[:2])
    require(len(selected) == 14, 'Expected 2 patients from each of 7 reserve sites')
    return {'status': 'METADATA_SELECTION_PREPARED_NOT_TEST_RELEASED', 'source_commit': commit,
            'selection_rule': 'Smallest SHA256(ds007045-prospective-site2-v1:commit:subject) two per site; lexical subject tie-break.',
            'eligible_reserve': len(candidates), 'reserve_site_counts': dict(Counter(c['site'] for c in candidates)),
            'reserve_payload_bytes': sum(p['bytes'] for c in candidates for p in c['payloads']),
            'selected_count': len(selected), 'selected_payload_bytes': sum(p['bytes'] for c in selected for p in c['payloads']),
            'selected_PRIVATE': selected, 'new_payload_downloaded_or_inspected': False,
            'release_gate': 'BLOCKED until exact comparison checkpoint hashes, renderer, task eligibility, scoring and patient uncertainty protocol are frozen and reviewed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--balanced-outputs', type=Path)
    parser.add_argument('--metadata-repo', type=Path)
    args = parser.parse_args()
    private = Path(__file__).resolve().parents[1] / 'local_data'
    require(args.output_dir.resolve().is_relative_to(private.resolve()), 'Patient-linked audits must stay in local_data')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_schedule(args.root)
    (args.output_dir / 'schedule_split_baselines_PRIVATE.json').write_text(json.dumps(audit, indent=2) + '\n')
    evaluation = args.root / 'internal_dev_eval_v1'
    named = {'pre_sft': evaluation / 'sft0_matched_native_outputs_PRIVATE.jsonl', 'historical_sft28': evaluation / 'sft28_matched_native_outputs_PRIVATE.jsonl'}
    if args.balanced_outputs:
        named['balanced_sft28'] = args.balanced_outputs
    scored = compare(evaluation / 'internal_dev_manifest_sft_train_matched.jsonl', evaluation / 'internal_dev_gold_sft_train_matched_PRIVATE.jsonl', named)
    (args.output_dir / 'matched_comparison_PRIVATE.json').write_text(json.dumps(scored, indent=2) + '\n')
    if args.metadata_repo:
        prospective = prospective_metadata(args.root, args.metadata_repo)
        (args.output_dir / 'prospective_metadata_selection_PRIVATE.json').write_text(json.dumps(prospective, indent=2) + '\n')
    print(json.dumps({'patient_counts': audit['patient_counts'], 'sequence_counts': audit['sequence_counts'], 'nonsequence_equal': audit['nonsequence_equal'], 'changed_sequence_records': audit['changed_sequence_records'], 'baselines': audit['baselines'], 'models': {k: {a: v[a] for a in ('accuracy_correct', 'format_valid', 'equal_task_macro_accuracy', 'worst_task_accuracy', 'patient_mean_accuracy')} for k, v in scored['models'].items()}}, indent=2))


if __name__ == '__main__':
    main()
