"""Training-only candidate likelihood diagnostics; no model imports.

Mean label log probability is the prespecified length-aware diagnostic. It is
not a calibrated class posterior; raw label sums are retained separately.
"""
import math
import hashlib
from pathlib import Path

LABELS = ('FLAIR', 'contrast-enhanced T1')
PREFIX = '<think>Visible MRI evidence.</think><answer>\\boxed{'
SUFFIX = '}</answer>'


def training_pairs(rows, patient_keys, level):
    """Require a complete pair per frozen training patient, never select by score."""
    selected = {}
    for row in rows:
        info=row['extra_info']
        if (info.get('internal_split')!='train' or info.get('patient_lineage')!='BIDS_SUBJECT_VERIFIED'
            or row.get('data_source')!='OpenNeuro_ds007045' or info.get('patient_key') not in patient_keys):
            raise ValueError('Only frozen training patients are allowed')
        if row['ability']!='mri_sequence':continue
        for modality,label in zip(('flair','t1ce'),LABELS):
            key=f"{info['patient_key']}_z{level:03d}_{modality}_sequence"
            if row['id']!=key:continue
            if (row['reward_model']['ground_truth']!=label or len(row['images'])!=1 or
                Path(row['images'][0]).name!=key.removesuffix('_sequence')+'.png'):
                raise ValueError('Image/label identity mismatch')
            pairkey=(info['patient_key'],modality)
            if pairkey in selected:raise ValueError('Duplicate pair member')
            selected[pairkey]=row
    if len(selected)!=2*len(patient_keys):raise ValueError('Incomplete patient pair inventory')
    ordered=sorted(patient_keys,key=lambda p:hashlib.sha256(f'7:mri_sequence:{p}'.encode()).hexdigest())
    result=[selected[(p,m)] for p in ordered for m in ('flair','t1ce')]
    if len({r['extra_info']['question'] for r in result})!=1:raise ValueError('Nonidentical questions')
    return result


def partition_target(tokenizer, label, *, prompt_length):
    if label not in LABELS or type(prompt_length) is not int or prompt_length < 1:
        raise ValueError('Invalid label or prompt length')
    enc = lambda s: list(tokenizer.encode(s, add_special_tokens=False))
    prefix, answer, suffix = enc(PREFIX), enc(label), enc(SUFFIX)
    full = enc(PREFIX + label + SUFFIX)
    if not answer or full != prefix + answer + suffix:
        raise ValueError('Tokenizer boundary crosses fixed prefix/label/suffix')
    start = prompt_length + len(prefix) - 1
    return {'target_ids': full, 'prefix_ids': prefix, 'label_ids': answer,
            'suffix_ids': suffix, 'label_logit_positions': list(range(start, start + len(answer)))}


def summarize_tokens(logps, partition):
    n, a = len(partition['prefix_ids']), len(partition['label_ids'])
    if (len(logps) != len(partition['target_ids']) or not a or
        any(not math.isfinite(x) or x > 1e-6 for x in logps)):
        raise ValueError('Incomplete or nonfinite/positive token log probabilities')
    answer = logps[n:n+a]; fixed = logps[:n] + logps[n+a:]
    return {'label_logp_sum':sum(answer), 'label_logp_mean':sum(answer)/a,
            'label_nll_mean':-sum(answer)/a, 'label_tokens':a,
            'fixed_nll_mean':-sum(fixed)/len(fixed), 'fixed_tokens':len(fixed),
            'prefix_nll_mean':-sum(logps[:n])/n,
            'suffix_nll_mean':-sum(logps[n+a:])/len(logps[n+a:]),
            'whole_target_nll_mean':-sum(logps)/len(logps)}


def compare_candidates(candidates):
    result = {}
    for kind, field in [('raw','label_logp_sum'),('mean','label_logp_mean')]:
        scores = [candidates[label][field] for label in LABELS]
        if not all(math.isfinite(x) for x in scores):raise ValueError('Nonfinite candidate score')
        margin = scores[1]-scores[0]; exps=[math.exp(x-max(scores)) for x in scores]
        probs=[x/sum(exps) for x in exps]
        result.update({kind+'_t1_minus_flair':margin,
            kind+'_preferred':LABELS[1] if margin>0 else LABELS[0] if margin<0 else 'TIE',
            kind+'_candidate_preferences':probs,
            kind+'_candidate_entropy_nats':-sum(p*math.log(p) for p in probs if p)})
    return result
