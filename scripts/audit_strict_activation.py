"""Replay private generated strict rewards; emit only aggregate diagnostics."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from mri_rl.vero_mri_reward import compute_score_from_data_source, _answer_and_format


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit(run: Path, records: Path):
    receipt = json.loads((run / 'receipt.json').read_text())
    raw = json.loads((run / 'raw_rollouts.json').read_text())
    rows = {r['id']: r for r in map(json.loads, records.open())}
    assert sha(records) == receipt['training_records_sha256']
    assert len(raw) == len(receipt['attempts']) == receipt['groups_attempted']
    groups, answers, patients = [], Counter(), set()
    for group, saved in zip(raw, receipt['attempts']):
        row = rows[group['record_id']]
        assert row['extra_info']['internal_split'] == 'train'
        assert row['extra_info']['patient_lineage'] == 'BIDS_SUBJECT_VERIFIED'
        assert sha(Path(row['images'][0])) == group['image_sha256'] == saved['image_sha256']
        assert hashlib.sha256(group['rendered_prompt'].encode()).hexdigest() == saved['rendered_prompt_sha256']
        patients.add(row['extra_info']['patient_key'])
        grades = []
        for generated, logged in zip(group['rollouts'], saved['rewards']):
            grade = compute_score_from_data_source(row['data_source'], generated['text'],
                        row['reward_model']['ground_truth'], row['extra_info'], format_score=0.2)
            assert all(grade[k] == logged[k] for k in ('accuracy','format','score'))
            grades.append(grade)
            answers[_answer_and_format(generated['text'])[0]] += 1
        assert len(grades) == receipt['group_size'] == len(saved['rewards'])
        task = [g['accuracy'] for g in grades]
        active = all(g['format'] == 1 for g in grades) and max(task) > min(task)
        assert saved['active'] == active
        groups.append({'format_valid':sum(g['format'] == 1 for g in grades),
                       'task_correct':sum(task),'n':len(grades),
                       'task_variance':sum((x-sum(task)/len(task))**2 for x in task)/len(task),
                       'active':active,'response_tokens':sum(saved['response_tokens'])})
    steps = receipt['optimizer_steps']
    if steps:
        assert steps == 1 and receipt['policy_updated']
        assert sha(run/'before_adapters.safetensors') == receipt['before_adapter_sha256']
        assert sha(run/'after_adapters.safetensors') == receipt['after_adapter_sha256']
        assert receipt['before_adapter_sha256'] != receipt['after_adapter_sha256']
        reload = json.loads((run/'reload_inference_receipt.json').read_text())
        assert sha(run/'reload_inference_receipt.json') == receipt['reload_inference_receipt_sha256']
        assert reload['exact_adapter_tensor_match'] and reload['fresh_model_loaded']
        assert reload['loaded_adapter_sha256'] == receipt['after_adapter_sha256']
    else:
        assert not receipt['policy_updated'] and not any(g['active'] for g in groups)
    return {'status':receipt['status'],'train_patients':len(patients),'groups':groups,
            'answer_counts':dict(answers),'optimizer_steps':steps,
            'gradient_l2':receipt.get('gradient_l2'),
            'checkpoint_reloaded':receipt.get('checkpoint_reloaded',False),
            'source_receipt_sha256':sha(run/'receipt.json'),
            'raw_rollouts_sha256':sha(run/'raw_rollouts.json'),
            'science_claim':False}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.run,args.records),indent=2))
