# Sampling logic adapted from Vero's pinned rl_dataset.py (Apache License 2.0).
# Copyright 2024 Bytedance Ltd. and/or its affiliates;
# Copyright 2023-2024 SGLang Team; Copyright 2025 ModelBest Inc. and/or its affiliates.
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an AS IS basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Finite two-category schedule matching pinned Vero PerBatchDomainSampler.

The official logical batch is256 prompts. Each selected prompt remains a complete
rollout group when executed serially; this helper never alters model inputs or
rewards. A frozen198-occurrence list plus the driver's completed-group cursor is
sufficient for resume; no second sampler-state manager is introduced.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json

import numpy as np

from mri_rl.two_task_pool import validate_model_boundary
from mri_rl.vero_official_launch import expected_batch_counts, VERO_SHA

OFFICIAL_VERO_COMMIT = VERO_SHA
OFFICIAL_CLASS_SHA256 = '472466d5aae9a5b05243d77d914d4869052597d961676f2b926ddec6528b2121'
OFFICIAL_CLASS_PATH = 'vero-rl/verl/utils/dataset/rl_dataset.py:PerBatchDomainSampler'
TASKS = ('mri_sequence', 'mri_visit_correspondence')
LOGICAL_BATCH_SIZE = 256
MAX_GROUPS = 198


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def build_category_schedule(records, *, mode, seed, max_groups=MAX_GROUPS, specialist_category=None):
    """Precompute official sampler order; keep source IDs distinct from occurrences.

    Initial epoch RNG/permutation state matches official _start_epoch(). The
    constructor's earlier permutations are unused there because _start_epoch()
    resets RNG to the same seed; omitting those unused draws preserves the stream.
    Source order is the exact incoming frozen manifest order. No class balancing,
    patient reweighting, unique-subset selection or outcome-dependent choice occurs.
    """
    _require(mode in {'specialist','proportional','uniform'}, 'Unknown sampling mode')
    _require(type(seed) is int and 0 <= seed < 2**63, 'Nonnegative bounded integer seed required')
    _require(type(max_groups) is int and 1 <= max_groups <= MAX_GROUPS, 'Frozen1–198 occurrence budget required')
    _require(bool(records), 'Empty training pool')
    ids=set();counts=Counter();patients={task:set() for task in TASKS}
    for row in records:
        _require(row.get('ability') in TASKS and isinstance(row.get('id'),str) and row['id'],
                 'Two typed categories and source record IDs required')
        _require(row['id'] not in ids and 'source_record_id' not in row,
                 'Duplicate source ID or already-expanded occurrence')
        validate_model_boundary(row,'train')
        ids.add(row['id']);task=row['ability'];counts[task]+=1
        info=row['extra_info'];patients[task].add(info['patient_key'])
        if info.get('candidate_patient_key'):patients[task].add(info['candidate_patient_key'])
    _require(set(counts)==set(TASKS), 'Both frozen source pools required even for specialist comparisons')
    if mode=='specialist':
        _require(specialist_category in TASKS, 'A typed specialist category is required')
        weights={specialist_category:1.}
    else:
        _require(specialist_category is None, 'Specialist category forbidden for mixtures')
        weights={task:(.5 if mode=='uniform' else counts[task]/len(records)) for task in TASKS}
    draws=expected_batch_counts(weights,LOGICAL_BATCH_SIZE)
    names=list(weights)
    domain_indices={name:np.array([i for i,row in enumerate(records) if row['ability']==name],dtype=np.int64) for name in names}
    rng=np.random.default_rng(seed)
    permutation={name:rng.permutation(domain_indices[name]) for name in names}
    position={name:0 for name in names}
    def take(name,count):
        out=[]
        while len(out)<count:
            perm=permutation[name];pos=position[name]
            size=min(count-len(out),len(perm)-pos)
            if size>0:out.extend(perm[pos:pos+size]);pos+=size
            if pos>=len(perm):
                permutation[name]=rng.permutation(domain_indices[name]);position[name]=0;pos=0
            position[name]=pos
        return out
    full=[]
    while len(full)<max_groups:
        batch=[]
        for name in names:
            if draws[name]:batch.extend(take(name,draws[name]))
        rng.shuffle(batch)
        full.extend(int(i) for i in batch)
    selected=full[:max_groups];schedule=[]
    for occurrence,index in enumerate(selected):
        row=json.loads(json.dumps(records[index],allow_nan=False))
        source_id=row['id']
        row['source_record_id']=source_id
        row['id']=_digest({'version':'vero-category-occurrence-v1','mode':mode,'specialist':specialist_category,
                           'seed':seed,'occurrence':occurrence,'source_record_id':source_id})[:24]
        schedule.append(row)
    binding={
        'schema':'vero-category-frozen-schedule-v1','official_vero_commit':OFFICIAL_VERO_COMMIT,
        'official_class_path':OFFICIAL_CLASS_PATH,'official_class_sha256':OFFICIAL_CLASS_SHA256,
        'numpy_version':np.__version__,'mode':mode,'specialist_category':specialist_category,'seed':seed,
        'logical_batch_size':LOGICAL_BATCH_SIZE,'logical_batch_category_counts':draws,'sampling_weights':weights,
        'source_pool_sha256':_digest(records),'source_pool_record_counts':dict(counts),
        'source_pool_patient_counts':{task:len(patients[task]) for task in TASKS},
        'maximum_occurrences':max_groups,'precomputed_logical_blocks':len(full)//LOGICAL_BATCH_SIZE,
        'source_indices_sha256':_digest(selected),'full_logical_blocks_indices_sha256':_digest(full),
        'rng_state_after_logical_blocks_sha256':_digest(rng.bit_generator.state),
        'schedule_sha256':_digest(schedule),'realized_occurrence_counts':dict(Counter(r['ability'] for r in schedule)),
        'selected_unique_source_records':{task:len({r['source_record_id'] for r in schedule if r['ability']==task}) for task in TASKS},
        'resume_rule':'Freeze complete schedule and binding; existing driver completed_groups cursor indexes occurrences. Never regenerate a different seed/pool/order.',
        'logical_block_rng_is_not_per_occurrence_rng':True,
        'patient_count_is_not_occurrence_count':True,'model_prompt_unchanged':True,
    }
    return {'schedule':schedule,'source_indices':selected,'binding':binding}
