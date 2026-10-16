"""Synthetic scorer regression tests; no patient or image payloads."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('balanced_audit', Path(__file__).parents[1] / 'scripts/audit_balanced_sft_comparison.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(tmp_path):
    image = tmp_path / 'image.bin'
    image.write_bytes(b'synthetic')
    manifest = tmp_path / 'manifest.jsonl'
    gold = tmp_path / 'gold.jsonl'
    entries = [('p1_a', 'p1', 'FLAIR'), ('p1_b', 'p1', 'contrast-enhanced T1'), ('p2_a', 'p2', 'FLAIR')]
    prompts = [{'record_id': key, 'category': 'mri_sequence', 'image': str(image), 'prompt': [{'role': 'user', 'content': 'Synthetic?'}]} for key, _, _ in entries]
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in prompts))
    gold.write_text(''.join(json.dumps({'record_id': key, 'category': 'mri_sequence', 'patient_key': patient, 'internal_split': 'internal_dev', 'ground_truth': truth, 'extra_info': {'reward_type': 'mri_sequence'}}) + '\n' for key, patient, truth in entries))
    predictions = [{**p, 'response': r'<think>x</think><answer>\boxed{FLAIR}</answer>', 'manifest_sha256': audit.digest(manifest), 'image_sha256': audit.digest(image), 'image_placeholder_count': 1, 'template_override_sha256': None, 'adapter_verified_parameter_tensors': 1, 'model_path': '/synthetic', 'model_config_sha256': 'c', 'model_template_sha256': 't', 'adapter_checkpoint_sha256': 'a', 'adapter_config_sha256': 'lora', 'max_tokens': 256, 'temperature': 0} for p in prompts]
    first, second = tmp_path / 'first.jsonl', tmp_path / 'second.jsonl'
    first.write_text(''.join(json.dumps(r) + '\n' for r in predictions))
    predictions[1]['response'] = r'<think>x</think><answer>\boxed{contrast-enhanced T1}</answer>'
    for r in predictions:
        r['adapter_checkpoint_sha256'] = 'b'
    second.write_text(''.join(json.dumps(r) + '\n' for r in predictions))
    return manifest, gold, first, second


def test_paired_cluster_metric_and_balanced_accuracy(tmp_path):
    manifest, gold, first, second = fixture(tmp_path)
    result = audit.compare(manifest, gold, {'a': first, 'b': second})
    assert result['models']['a']['accuracy_correct'] == 2
    assert result['models']['a']['by_category']['mri_sequence']['balanced_accuracy'] == .5
    assert result['models']['a']['patient_mean_accuracy'] == .75
    assert result['paired']['a__vs__b']['patient_mean_difference_b_minus_a'] == .25
    assert 'NOT_MEASURED' in result['models']['a']['truncation']


def test_rejects_cross_model_token_budget_change(tmp_path):
    manifest, gold, first, second = fixture(tmp_path)
    changed = audit.rows(second)
    for row in changed:
        row['max_tokens'] = 128
    second.write_text(''.join(json.dumps(r) + '\n' for r in changed))
    with pytest.raises(ValueError, match='Cross-model'):
        audit.compare(manifest, gold, {'a': first, 'b': second})


def test_rejects_duplicate_predictions(tmp_path):
    manifest, gold, first, _ = fixture(tmp_path)
    with first.open('a') as f:
        f.write(first.read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='duplicate'):
        audit.detailed_score(manifest, gold, first)


def test_level_rule_fits_training_only_and_tie_is_fixed():
    def row(split, label):
        return {'id': f'{split}_z045_paired_presence', 'ability': 'mri_tumor_presence', 'reward_model': {'ground_truth': label}}
    # Supply all categories so the scorer's fixed four-task contract is satisfied.
    train, dev = [], []
    for task in audit.CATEGORIES:
        for label in ['no', 'yes']:
            r = row('train', label); r['ability'] = task; train.append(r)
        r = row('dev', 'yes'); r['ability'] = task; dev.append(r)
    result = audit.level_baselines(train, dev)
    assert result['mri_tumor_presence']['level_rule'] == {'045': 'no'}
    assert result['mri_tumor_presence']['level_correct'] == 0
