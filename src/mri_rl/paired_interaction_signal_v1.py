"""CPU reference and pair interface for an EXPLICIT supervised GSPO auxiliary.

Not connected to any trainer. No model load, sampling, reward or optimizer.
Rows are FLAIR/T1 images; columns are FLAIR/T1 mean label-token logps.
L=softplus(-((s00-s01)-(s10-s11))/2); prospective alpha=.1, tau=1.
Common additive class priors cancel. This does not ensure either class recall,
grounding, clinical validity, or absence of cross-task interference.
"""
import math

LABELS=('FLAIR','contrast-enhanced T1')
ALPHA=.1

def _matrix(scores):
    if len(scores)!=2 or any(len(r)!=2 for r in scores):raise ValueError('Two images by two classes required')
    if not all(math.isfinite(float(x)) for r in scores for x in r):raise ValueError('Finite scores required')
    return [[float(x) for x in r] for r in scores]

def pair_loss(scores):
    s=_matrix(scores);d=((s[0][0]-s[0][1])-(s[1][0]-s[1][1]))/2
    x=-d
    return max(x,0)+math.log1p(math.exp(-abs(x)))

def score_derivatives(scores):
    s=_matrix(scores);d=((s[0][0]-s[0][1])-(s[1][0]-s[1][1]))/2
    p=math.exp(-max(d,0))/(math.exp(-max(d,0))+math.exp(min(d,0)))
    return [[-p/2,p/2],[p/2,-p/2]]

def differentiable_pair_loss(scores,ops):
    """Backend-neutral differentiable scalar; native integration NOT_VERIFIED.

    Caller must independently ensure finite 2x2 scores and label-only alignment.
    No conversion to Python floats or detach of the gradient graph.
    """
    d=((scores[0][0]-scores[0][1])-(scores[1][0]-scores[1][1]))/2
    return ops.logaddexp(0.,-d)

def auxiliary_objective(unchanged_gspo_loss,scores,ops):
    """An inactive original GSPO branch contributes zero, never fake advantage.

    The new supervised branch can be nonzero independently. Such an update must
    be reported as auxiliary-only, not as an accepted original GSPO group.
    Original gates continue to govern the original branch; native integration
    and all training remain unauthorized.
    """
    return unchanged_gspo_loss+ALPHA*differentiable_pair_loss(scores,ops)

def mean_label_score(logps,start,stop):
    if not (type(start) is int and type(stop) is int and 0<=start<stop<=len(logps)):
        raise ValueError('Exact nonempty label span required')
    values=[float(v) for v in logps[start:stop]]
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite label scores')
    return sum(values)/len(values)

def operation_budget(sequence_slots,dallas_slots):
    """Conservative declared positions, including repeated auxiliary forwards.

    Two-pass serial gradient plan: four value forwards, four recompute forwards
    with two image backward calls, plus two independent prefix oracle forwards.
    This is prospective accounting, not measured native throughput or training.
    """
    if any(type(v) is not int or v<0 for v in (sequence_slots,dallas_slots)):raise ValueError('Nonnegative slot counts required')
    base=4*((213+256)*sequence_slots+(405+256)*dallas_slots)
    pair_positions=2*(2*((213+11+2-1)+(213+11+5-1)))+2*(213+11)
    return {'GSPO_reserved_tokens':base,'auxiliary_teacherforced_positions':sequence_slots*pair_positions,
        'combined_accounted_token_positions':base+sequence_slots*pair_positions,
        'auxiliary_forward_calls':10*sequence_slots,'auxiliary_backward_calls':2*sequence_slots,
        'maximum_returned_rollout_tokens':4*256*(sequence_slots+dallas_slots)}

def attach_pairs(slots,records):
    """Private routing metadata only; main GSPO schedule stays byte equivalent.

    IDs/patient/image SHA are never model inputs. Both images share identical
    generic question; candidate labels are scored symmetrically, not in prompt.
    """
    lookup={};pairs={}
    for r in records:
        if r['id'] in lookup or r['label'] not in LABELS or r['level']!=45:raise ValueError('Unique qualified z045 record required')
        lookup[r['id']]=r;key=(r['patient'],r['level']);p=pairs.setdefault(key,{})
        if r['label'] in p:raise ValueError('Duplicate modality')
        p[r['label']]=r
    for pair in pairs.values():
        if set(pair)!=set(LABELS):raise ValueError('Complete modality pair required')
        f,t=(pair[k] for k in LABELS)
        if f['image_sha256']==t['image_sha256'] or f['question']!=t['question']:raise ValueError('Distinct images and identical generic question required')
    result=[]
    for slot in slots:
        ids=None
        if slot['task']=='sequence':
            r=lookup[slot['record_id']];p=pairs[(r['patient'],r['level'])];ids=[p[k]['id'] for k in LABELS]
        result.append({'main':dict(slot),'pair_ids':ids})
    return result
