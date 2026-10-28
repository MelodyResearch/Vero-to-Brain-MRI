"""Prospective z045-pool evaluation. Saved outputs only; no model imports.

Strict tags/box parser and tie-invalid candidate readout preserve the existing
internal-dev conventions. A diagnostic GO never authorizes another update.
"""
from __future__ import annotations
import math
from mri_rl.vero_mri_reward import _answer_and_format

LABELS={'mri_sequence':('FLAIR','contrast-enhanced T1'),
        'mri_visit_correspondence':('different','same')}
KEYS=('sequence_balanced_accuracy','FLAIR_recall','T1_recall',
      'sequence_candidate_balanced_accuracy','Dallas_accuracy',
      'Dallas_same_recall','Dallas_different_recall')

def require(ok,message):
    if not ok: raise ValueError(message)

def strict_prediction(response,task):
    require(task in LABELS and isinstance(response,str),'Task/text required')
    answer,fmt=_answer_and_format(response)
    return {v.lower():i for i,v in enumerate(LABELS[task])}.get(answer.lower(),-1) if fmt==1. else -1

def candidate_prediction(scores):
    require(len(scores)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in scores),'Two finite canonical scores required')
    return -1 if scores[0]==scores[1] else int(scores[1]>scores[0])

def decision_vector(*values):
    require(len(values)==len(KEYS),'Complete endpoint vector required')
    x=dict(zip(KEYS,values)); _validate_vector(x); return x

def _validate_vector(x):
    require(set(x)==set(KEYS) and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in x.values()),'Complete finite endpoint vector required')

def decide(parent,child):
    _validate_vector(parent);_validate_vector(child)
    improving=('sequence_balanced_accuracy','T1_recall','sequence_candidate_balanced_accuracy')
    checks={k:child[k]>parent[k] if k in improving else child[k]>=parent[k] for k in KEYS}
    return {'decision':'ENGINEERING_GO_REQUIRES_NEW_AUTHORIZATION' if all(checks.values()) else 'STOP_SAME_CONFIGURATION',
            'endpoint_checks':checks,'generalization_proven':False,'training_authorized':False}

def swap_summary(real,swapped,real_truth):
    require(len(real)==len(swapped)==len(real_truth)>0,'Exact paired swap coverage required')
    require(all(v in (-1,0,1) for v in (*real,*swapped)) and all(v in (0,1) for v in real_truth),'Binary sequence predictions required')
    triples=list(zip(real,swapped,real_truth))
    return {'comparisons':len(triples),'changed_predictions':sum(a!=b for a,b,_ in triples),
            'valid_opposite_class_changes':sum(a>=0 and b>=0 and a!=b for a,b,_ in triples),
            'both_images_correct':sum(a==y and b==1-y for a,b,y in triples),
            'swapped_displayed_truth_accuracy':sum(b==1-y for _,b,y in triples)/len(triples),
            'interpretation':'A response change or larger margin alone is not correctness or MRI grounding.'}

def cost_per_update(charged_tokens,seconds,updates):
    require(type(charged_tokens) is int and charged_tokens>=0 and type(updates) is int and updates>=0 and
            type(seconds) in (int,float) and math.isfinite(seconds) and seconds>=0,'Measured finite cost/counters required')
    return {'charged_tokens':charged_tokens,'wall_seconds':seconds,'accepted_updates':updates,
            'tokens_per_update':charged_tokens/updates if updates else None,
            'seconds_per_update':seconds/updates if updates else None,
            'zero_update_interpretation':'Undefined per-update cost when no update is accepted; never report zero as efficiency.'}

def protocol():
    return {'schema':'paired-z045-evaluation-contract-v1','status':'FROZEN_PREPARATION_NOT_RUN',
        'primary':'Sequence strict-greedy balanced accuracy; T1 recall must increase, FLAIR recall must not fall.',
        'candidate':'Separate mean-label balanced accuracy and correct-minus-wrong margin; sum and first secondary, ties INVALID.',
        'decoding':{'temperature':0.,'max_tokens':256,'prompt':'UNCHANGED_ORIGINAL_VERO_SYSTEM_AND_TASK_QUESTION'},
        'strict_format':'Existing Vero full tags, nonempty think/answer, exactly one boxed canonical label; format1 only.',
        'development':{'real_sequence':24,'sequence_people':4,'real_Dallas':16,'Dallas_cycles':4,
            'swap_sequence':8,'swap_rule':'All z045 paired modalities of the same four dev people; displayed-image truth flips.',
            'greedy_calls_per_checkpoint':40,'candidate_queries_per_checkpoint':40,'canonical_labels_per_query':2,
            'swap_new_model_calls':0,'swap_reuse':'Opposite real z045 row has identical pixels/prompt/prefix; reuse its authenticated output, never count aliases as new independent cases.',
            'candidate_LM_calls_per_checkpoint':120,'candidate_calls_interpretation':'80 canonical class forwards plus at most40 prefix-oracle forwards; generation LM token steps counted separately.',
            'status':'Previously explored internal-dev, engineering diagnostic only; no independent-generalization claim.'},
        'numeric_contract':{'baseline_and_child_same_contract':True,'required':'Separately authenticated native rollout/scorer/backward qualification; narrow F32 contexts are insufficient.',
            'BF16_saved_parent_outputs':'Not a matched baseline for an F32 child. Re-evaluate exact Adam60 only under the same newly qualified contract.'},
        'holdout_status':'NOT_ESTABLISHED','future_independent_holdout':{
            'sequence_people':16,'modalities_per_person':2,'level':45,'site_strategy':'Two per each of the eight existing source sites if available; otherwise freeze a revised availability rule before any output.',
            'exclusions':'All historically trained, development, final100, feature/probe and adjudication patients; deduplicate source lineage before claims.',
            'selection':'Source metadata only; deterministic sha256 ordering with salt paired-z045-holdout-v1 within site; no predictions or final100 labels/outcomes.',
            'collection':'Not fetched or qualified in this preparation; independent output remains sealed until method frozen.',
            'Dallas':'Old dev retention is engineering-only; no independent Dallas generalization is claimed.'},
        'uncertainty':'Frozen 2000 patient/cycle-cluster bootstrap seed7; descriptive only on repeated dev.',
        'finite_evaluation_budget':{'checkpoints':2,'new_greedy_generations':80,'maximum_generated_tokens':20480,
            'candidate_score_LM_API_calls':240,'maximum_teacherforced_input_positions':102000,
            'teacherforced_cap_basis':'240 calls x425 input positions; conditional on native prompt<=405, fixed prefix<=16 and label<=5. This is not training-ledger tokens.',
            'operation_start_seconds':2400,'atomic_call_and_release_can_overrun':True,
            'no_repeated_baseline_if_same_contract_already_authenticated':True},
        'go_rule':'Strict sequence balanced accuracy, T1 recall and mean candidate balanced accuracy all strictly improve; FLAIR and each Dallas accuracy/recall do not fall. Otherwise stop this configuration.',
        'efficiency':'Actual charged input+visual+response tokens, completed and accepted groups, wall seconds, token/seconds per accepted update; zero updates => undefined, never divide by zero.',
        'model_training_authorized':False,'formal100_access':False,'optimizer_updates_this_preparation':0}
