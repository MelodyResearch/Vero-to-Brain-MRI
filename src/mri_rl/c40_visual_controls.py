
"""Exploratory C40 image-control pairing; no model, pixel or sealed-test access.

The caller authenticates exact source/output/projection/checkpoint receipts.
Saved paired images are permutations of already inferred inputs, not new runs.
"""
from __future__ import annotations
from collections import defaultdict
from statistics import mean
import hashlib
import re
from types import FunctionType
from mri_rl.two_task_analysis import _reference
from mri_rl.vero_mri_reward import _answer_and_format
from mri_rl import answer_likelihood as al

SALT='c40-visual-controls-v1'
CENTRAL_LEVEL=91
LABELS={'mri_sequence':al.LABELS,'mri_visit_correspondence':('different','same')}

def require(ok,message):
    if not ok:raise ValueError(message)

def build_pairs(rows):
    _reference(rows,'internal_dev')
    groups=defaultdict(list)
    for row in rows:
        task=row['ability'];e=row['extra_info']
        if task=='mri_sequence':
            match=re.fullmatch(re.escape(e['patient_key'])+r'_z(\d{3})_(flair|t1ce)_sequence',row['id'])
            require(match is not None,'Sequence source patient/level identifier mismatch')
            require(row['reward_model']['ground_truth']=={'flair':'FLAIR','t1ce':'contrast-enhanced T1'}[match[2]],'Sequence modality reference mismatch')
            group=(task,e['patient_key'],int(match[1]))
        else:group=(task,e['statistical_cluster_key'],e['patient_key'])
        groups[group].append(row)
    result={task:[] for task in LABELS}
    for group,members in sorted(groups.items()):
        task=group[0];labels=LABELS[task]
        require(len(members)==2 and {r['reward_model']['ground_truth'] for r in members}==set(labels),'A paired control requires both exact reference labels')
        by={r['reward_model']['ground_truth']:r for r in members};negative,positive=(by[label] for label in labels)
        require(negative['extra_info']['question']==positive['extra_info']['question'],'Paired image prompt changed')
        require(len(negative['images'])==len(positive['images'])==1 and negative['images']!=positive['images'],'Paired images must be distinct')
        if task=='mri_visit_correspondence':
            require(negative['extra_info']['query_source_sha256']==positive['extra_info']['query_source_sha256'],'Candidate replacement changed query visit')
            require(negative['extra_info']['candidate_source_sha256']!=positive['extra_info']['candidate_source_sha256'],'Candidate replacement must use a different source image')
        result[task].append({'negative':negative,'positive':positive,'unit':group[1],'level':group[2] if task=='mri_sequence' else None})
    return result

def paired_greedy_summary(rows,predictions):
    require(set(predictions)=={r['id'] for r in rows},'Exact frozen output coverage required')
    result={}
    for task,pairs in build_pairs(rows).items():
        labels=tuple(x.lower() for x in LABELS[task]);counts=defaultdict(int);units=defaultdict(list)
        for pair in pairs:
            answers=[]
            for side in ('negative','positive'):
                answer,fmt=_answer_and_format(predictions[pair[side]['id']]['response'])
                answers.append(answer.lower() if fmt>0 and answer.lower() in labels else 'INVALID')
            valid='INVALID' not in answers
            correct=valid and answers==list(labels);reverse=valid and answers==list(reversed(labels))
            counts['correctly_separated_pairs']+=correct
            counts['reversed_pairs']+=reverse
            counts['same_valid_answer_pairs']+=valid and answers[0]==answers[1]
            counts['pairs_with_invalid']+=not valid
            units[pair['unit']].append(int(correct)-int(reverse))
        result[task]={**dict(counts),'pair_count':len(pairs),'independent_units':len(units),
            'unit':'patient' if task=='mri_sequence' else 'matched_two_person_cycle',
            'unit_averaged_direction_score':mean(mean(v) for v in units.values()),
            'direction_score_definition':'+1 correct negative-to-positive label switch; -1 reverse switch; 0 constant or invalid. Invalid pairs remain in denominator.',
            'correct_pair_fraction':counts['correctly_separated_pairs']/len(pairs),
            'input_permutation_only':True,'new_model_generations':0}
    return result

def select_probe_rows(rows):
    """No predictions argument: fixed two people at z091 and one complete cycle."""
    pairs=build_pairs(rows)
    order=lambda key:hashlib.sha256((SALT+'|'+key).encode()).hexdigest()
    people=sorted({p['unit'] for p in pairs['mri_sequence']},key=order)
    cycles=sorted({p['unit'] for p in pairs['mri_visit_correspondence']},key=order)
    require(len(people)>=2 and cycles,'Required frozen dev patients/cycles absent')
    selected=[]
    for task in LABELS:
        for pair in pairs[task]:
            if ((task=='mri_sequence' and pair['unit'] in people[:2] and pair['level']==CENTRAL_LEVEL)
                or (task=='mri_visit_correspondence' and pair['unit']==cycles[0])):
                selected.extend((pair['negative'],pair['positive']))
    require(len(selected)==8 and len({r['id'] for r in selected})==8,'Fixed balanced8-case subset unavailable')
    return sorted(selected,key=lambda r:(r['ability'],order(r['id'])))

def bound_candidate_functions(labels):
    """Bind only the candidate vocabulary to existing exact likelihood code.

    The original LABELS module is never modified. Original comparison field
    names containing 't1_minus_flair' mean candidate2-minus-candidate1 when
    bound to Dallas; the runner must relabel these explicitly in its receipt.
    """
    require(tuple(labels) in set(LABELS.values()),'Unapproved task candidate pair')
    namespace={**al.__dict__,'LABELS':tuple(labels)}
    return tuple(FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
                 for fn in (al.partition_target,al.compare_candidates))
