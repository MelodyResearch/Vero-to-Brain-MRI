"""Experiment 3 on ALREADY SAVED V1 DEV24 outputs (Parent/A/B): CPU only, zero model calls.

1. Checks from saved generation fingerprints whether each same-patient same-level FLAIR/T1 pair had identical
   prompt/input_ids/mask/grid metadata (only pixels differ). If so, the two real evaluations are an exact paired image
   substitution taken FROM THE MAIN EVALUATION (not a separate intervention).
2. Classifies each pair per model (discriminates / constant FLAIR / constant T1 / reversed / invalid) and compares A,B
   to Parent: discrimination gains vs constant-answer (class-prior) shifts.
Full per-pair detail goes to a PRIVATE file; stdout carries aggregate counts only (no ids, text or paths).
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mri_rl import accuracy_pilot_v2_core as core
from mri_rl import accuracy_pilot_v2_entry as entry
from mri_rl.shared_rollout_gspo_core_v1 import publish_once, file_sha256


def strip(rows):
    return [{'id': r['id'], 'truth': r['truth'], 'text': r['text'], 'tokens': len(r['actual_PRIVATE']['response']['token_ids']),
             'wall_seconds': r.get('wall_seconds', 0.), 'actual_PRIVATE': r['actual_PRIVATE']} for r in rows]


def analyze(dev, rows):
    pairs = core.dev_pairs(dev['records_PRIVATE'])
    for label in rows:
        core.require([r['id'] for r in rows[label]] == [r['id'] for r in dev['records_PRIVATE']], 'Same frozen DEV24 order required')
    substitution = {label: core.substitution_equivalence(pairs, rows[label]) for label in rows}
    profiles = {label: core.pair_profile(pairs, rows[label]) for label in rows}
    versus = {label: core.compare_to_reference(pairs, rows['Parent'], rows[label]) for label in ('A', 'B')}
    versus['B_vs_A'] = core.compare_to_reference(pairs, rows['A'], rows['B'])
    metrics = {label: core.greedy_summary(rows[label]) for label in rows}
    return {'schema': 'accuracy-pilot-v2-v1-saved-pair-analysis-v1', 'pairs': len(pairs), 'substitution': substitution, 'profiles': profiles,
            'versus': versus, 'metrics': metrics, 'model_calls': 0, 'label': core.SUBSTITUTION_LABEL, 'independent_intervention': False,
            'cohort': 'FROZEN_INTERNAL_DEV24_PREVIOUSLY_EXPLORED'}


def safe_summary(result):
    v = result['versus']
    return {'pairs': result['pairs'], 'all_pairs_exact_substitution': {k: s['all_exact'] for k, s in result['substitution'].items()},
            'pair_counts': {k: p['counts'] for k, p in result['profiles'].items()},
            'predicted_T1': {k: p['predicted_T1'] for k, p in result['profiles'].items()},
            'vs_parent': {k: {x: v[k][x] for x in ('discrimination_gains', 'discrimination_losses', 'constant_answer_shift_to_T1', 'constant_answer_shift_to_FLAIR',
                                                   'case_wrong_to_correct', 'case_correct_to_wrong', 'sign_test_two_sided_p_DESCRIPTIVE', 'reading')} for k in v},
            'model_calls': 0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery-spec', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    rspec = json.loads(args.recovery_spec.read_text()); dev = json.loads(Path(rspec['dev24']).read_text())
    rows = {label: strip(entry.v1_rows(rspec['output'], label)[0]) for label in ('Parent', 'A', 'B')}
    result = analyze(dev, rows)
    publish_once(args.output, result)
    print(json.dumps({**safe_summary(result), 'private_file_sha256': file_sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
