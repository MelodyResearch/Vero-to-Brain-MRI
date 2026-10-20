"""Private saved-only matched-evaluation arithmetic; no model imports.

The external caller must authenticate actual exits, closed logs, output hashes,
numerical scopes and their independent review before using these functions.
Only aggregate results may be published. Development estimates are exploratory.
"""
from __future__ import annotations
import math
import random
from collections import defaultdict
from statistics import mean
from mri_rl.paired_z045_evaluation_v1 import (
    LABELS, candidate_prediction, cost_per_update, decide, require,
    strict_prediction, swap_summary,
)


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def task_metrics(rows, prediction):
    """Binary confusion includes an explicit invalid column; no invalid salvage."""
    require(rows, 'Nonempty task required')
    matrix = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        y, p = row['truth_index'], row[prediction]
        require(y in (0, 1) and p in (-1, 0, 1), 'Binary truth/prediction required')
        matrix[y][2 if p == -1 else p] += 1
    require(all(sum(r) for r in matrix), 'Both true classes required')
    recalls = [matrix[i][i] / sum(matrix[i]) for i in (0, 1)]
    return {'n': len(rows), 'confusion_truth_by_prediction_0_1_INVALID': matrix,
            'accuracy': sum(matrix[i][i] for i in (0, 1)) / len(rows),
            'balanced_accuracy': mean(recalls), 'recall': recalls,
            'invalid': sum(r[2] for r in matrix)}


def bootstrap(rows, prediction, repetitions=2000, seed=7):
    clusters = defaultdict(list)
    for r in rows:
        clusters[r['cluster_PRIVATE']].append(r)
    keys = sorted(clusters)
    require(keys and repetitions == 2000 and seed == 7, 'Frozen bootstrap required')
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        sample = [r for k in rng.choices(keys, k=len(keys)) for r in clusters[k]]
        # Existing patient/cycle units contain both labels. Do not silently
        # drop a one-class resample if a malformed future plan is substituted.
        values.append(task_metrics(sample, prediction)['balanced_accuracy'])
    values.sort()
    return {'clusters': len(keys), 'replicates': repetitions, 'seed': seed,
            'balanced_accuracy_percentile_95': [values[49], values[1949]],
            'interpretation': 'Exploratory repeated-dev cluster bootstrap; not new-patient generalization.'}


def _expected(plan):
    require(plan.get('formal100_access') is False and plan['decoding'] == {
        'temperature': 0., 'max_tokens': 256,
        'prompt': 'UNCHANGED_ORIGINAL_VERO_SYSTEM_AND_TASK_QUESTION'}, 'Frozen development contract required')
    all_rows = plan['records_PRIVATE']
    real = {r['id']: r for r in all_rows if r['condition'] == 'real_image'}
    aliases = {r['id']: r for r in all_rows if r['condition'] != 'real_image'}
    require(len(real) == 40 and len(aliases) == 8 and len(all_rows) == 48,
            'Exactly40 unique real rows and8 aliases required')
    require(sum(r['row_PRIVATE']['ability'] == 'mri_sequence' for r in real.values()) == 24,
            'Frozen24 sequence/16 Dallas required')
    return real, aliases


def analyze_checkpoint(plan, output):
    expected, aliases = _expected(plan)
    require(output['checkpoint_label'] in ('parent', 'terminal') and
            output['model_parameters_Adam_unchanged'] is True and output['RNG_restored'] is True,
            'Actual state integrity required')
    rows = output['rows']
    require(len(rows) == 40 and len({r['id'] for r in rows}) == 40 and
            {r['id'] for r in rows} == set(expected), 'Exact unique output coverage required')
    require(set(output['aliases']) == set(aliases), 'Exact alias coverage required')
    private = {}
    for raw in rows:
        ref = expected[raw['id']]
        source = ref['row_PRIVATE']
        task = source['ability']
        labels = list(LABELS[task])
        truth = source.get('extra_info', {}).get('answer', source['reward_model']['ground_truth'])
        require(raw['task'] == task and raw['truth'] == truth and truth in labels and
                raw['image_sha256'] == ref['image_sha256'], 'Truth/task/image binding mismatch')
        candidate = raw['candidates']
        require(candidate['labels'] == labels and len(candidate['per_label']) == 2 and
                candidate['prefix_oracle']['tolerance'] == .002 and
                finite(candidate['prefix_oracle']['max_abs_error']) and
                0 <= candidate['prefix_oracle']['max_abs_error'] <= .002,
                'Canonical order/oracle gate required')
        for label, score in zip(labels, candidate['per_label']):
            lp = score['label_logps_PRIVATE']
            require(score['label'] == label and type(score['token_count']) is int and
                    score['token_count'] == len(lp) and 0 < len(lp) <= 5 and
                    all(finite(v) for v in lp), 'Exact finite label token partition required')
            require(all(finite(score[k]) for k in ('mean', 'sum', 'first')) and
                    abs(score['sum'] - sum(lp)) <= 1e-6 and
                    abs(score['mean'] - mean(lp)) <= 1e-6 and
                    abs(score['first'] - lp[0]) <= 1e-6, 'Candidate arithmetic differs')
        g = raw['greedy']
        cost = raw['costs']
        require(type(g['generation_tokens']) is int and 0 < g['generation_tokens'] <= 256 and
                cost['generated_tokens'] == g['generation_tokens'] and cost['greedy_calls'] == 1 and
                cost['candidate_score_API_calls'] == 3 and
                type(cost['teacherforced_input_positions']) is int and
                0 < cost['teacherforced_input_positions'] <= 1275, 'Per-record finite cost violated')
        extra = source['extra_info']
        record = {'truth_index': labels.index(truth),
                  'greedy_prediction': strict_prediction(g['text'], task),
                  'cluster_PRIVATE': extra['patient_key'] if task == 'mri_sequence' else extra['statistical_cluster_key'],
                  'task': task, 'prompt_sha256': raw['prompt_sha256']}
        for mode in ('mean', 'sum', 'first'):
            scores = [s[mode] for s in candidate['per_label']]
            record[mode + '_prediction'] = candidate_prediction(scores)
            record[mode + '_margin'] = scores[record['truth_index']] - scores[1 - record['truth_index']]
        private[raw['id']] = record
    summary = {}
    for task in LABELS:
        subset = [r for r in private.values() if r['task'] == task]
        summary[task] = {'strict_greedy': task_metrics(subset, 'greedy_prediction'),
                         'strict_cluster_uncertainty': bootstrap(subset, 'greedy_prediction'),
                         'candidate': {m: {**task_metrics(subset, m + '_prediction'),
                                           'mean_correct_minus_wrong_margin': mean(r[m + '_margin'] for r in subset)}
                                       for m in ('mean', 'sum', 'first')}}
    real_predictions, swapped_predictions, truth_indices = [], [], []
    candidate_real, candidate_swap, separations = [], [], []
    for key, alias in aliases.items():
        observed = output['aliases'][key]
        require(all(observed.get(k) == alias[k] for k in (
            'reuse_real_id_PRIVATE', 'original_real_id_PRIVATE', 'image_sha256')), 'Alias identity changed')
        original, reused = private[alias['original_real_id_PRIVATE']], private[alias['reuse_real_id_PRIVATE']]
        require(original['task'] == reused['task'] == 'mri_sequence' and
                original['truth_index'] == 1 - reused['truth_index'] and
                original['prompt_sha256'] == reused['prompt_sha256'] and
                expected[alias['reuse_real_id_PRIVATE']]['image_sha256'] == alias['image_sha256'],
                'Swap is not identical opposite-modality real input')
        real_predictions.append(original['greedy_prediction'])
        swapped_predictions.append(reused['greedy_prediction'])
        candidate_real.append(original['mean_prediction'])
        candidate_swap.append(reused['mean_prediction'])
        truth_indices.append(original['truth_index'])
        # Both margins oriented to the original truth: m(real)-m(swapped).
        separations.append(original['mean_margin'] + reused['mean_margin'])
    summary['image_swap'] = {'greedy': swap_summary(real_predictions, swapped_predictions, truth_indices),
                             'candidate_mean': swap_summary(candidate_real, candidate_swap, truth_indices),
                             'mean_margin_separation': mean(separations), 'new_inference_calls': 0,
                             'independent_pairs': 4, 'aliases': 8}
    seq, dallas = (summary[t]['strict_greedy'] for t in LABELS)
    summary['equal_task_macro_accuracy'] = mean([seq['accuracy'], dallas['accuracy']])
    summary['worst_task_accuracy'] = min(seq['accuracy'], dallas['accuracy'])
    totals = {'greedy_calls': 40, 'candidate_score_API_calls': 120,
              'generated_tokens': sum(r['costs']['generated_tokens'] for r in rows),
              'teacherforced_input_positions': sum(r['costs']['teacherforced_input_positions'] for r in rows)}
    require(all(output['actual_calls'].get(k) == v for k, v in totals.items()), 'Actual call counters disagree')
    require(finite(output['wall_seconds']) and output['wall_seconds'] >= 0, 'Measured wall required')
    summary['cost'] = {**totals, 'wall_seconds': output['wall_seconds'],
        'wall_seconds_scope': 'Per-checkpoint inference row loop; excludes model loading, parameter/Adam inventory audits and release. End-to-end elapsed time requires the external process witness.'}
    vector = {'sequence_balanced_accuracy': seq['balanced_accuracy'], 'FLAIR_recall': seq['recall'][0],
              'T1_recall': seq['recall'][1],
              'sequence_candidate_balanced_accuracy': summary['mri_sequence']['candidate']['mean']['balanced_accuracy'],
              'Dallas_accuracy': dallas['accuracy'], 'Dallas_same_recall': dallas['recall'][1],
              'Dallas_different_recall': dallas['recall'][0]}
    return summary, vector, private


def compare_saved(plan, parent, terminal, training_cost):
    require(parent['checkpoint_label'] == 'parent' and terminal['checkpoint_label'] == 'terminal', 'Parent/terminal order required')
    require(parent['numerical_contract_sha256'] == terminal['numerical_contract_sha256'], 'Different numerical contracts cannot form weight comparison')
    p, pv, pr = analyze_checkpoint(plan, parent)
    c, cv, cr = analyze_checkpoint(plan, terminal)
    require(all(pr[k]['prompt_sha256'] == cr[k]['prompt_sha256'] for k in pr), 'Parent/terminal prompts changed')
    transitions = {}
    for task in LABELS:
        keys = [k for k in pr if pr[k]['task'] == task]
        transitions[task] = {'wrong_or_invalid_to_correct': sum(pr[k]['greedy_prediction'] != pr[k]['truth_index'] and cr[k]['greedy_prediction'] == cr[k]['truth_index'] for k in keys),
                             'correct_to_wrong_or_invalid': sum(pr[k]['greedy_prediction'] == pr[k]['truth_index'] and cr[k]['greedy_prediction'] != cr[k]['truth_index'] for k in keys),
                             'candidate_mean_predictions_changed': sum(pr[k]['mean_prediction'] != cr[k]['mean_prediction'] for k in keys)}
    return {'schema': 'paired-z045-saved-comparison-v1', 'parent': p, 'terminal': c,
            'delta': {k: cv[k] - pv[k] for k in pv}, 'decision': decide(pv, cv),
            'prediction_transitions': transitions, 'training_cost': cost_per_update(**training_cost),
            'generalization_proven': False, 'new_training_authorized': False,
            'formal100_used': False, 'interpretation': 'Same-contract exploratory parent/terminal comparison; no causal pairing or independent-patient claim.'}
