"""Explicit supervised per-class MRI margin auxiliary; not original Vero GSPO.

Rows: same-person z045 FLAIR, contrast-enhanced T1. Columns: canonical FLAIR,
T1 mean label-only logps under the SAME input/prefix for each comparison.
Fixed-format/prefix/EOS tokens excluded. Sum/first readouts remain separate.
This loss encourages both score-space correct margins; shared parameters and
Adam momentum can still hurt a class. Native derivatives and accuracy require
separate evidence. No model, sampler, reward or optimizer is constructed here.
"""
import math
from mri_rl.paired_interaction_signal_v1 import _matrix,attach_pairs,mean_label_score

ALPHA=.1
LABELS=('FLAIR','contrast-enhanced T1')

def margins(scores):
    s=_matrix(scores)
    return s[0][0]-s[0][1],s[1][1]-s[1][0]

def _softplus(x):return max(x,0.)+math.log1p(math.exp(-abs(x)))
def _negative_sigmoid(m):
    if m>=0:
        e=math.exp(-m);return e/(1+e)
    return 1/(1+math.exp(m))

def classification_loss(scores):
    mf,mt=margins(scores)
    return (_softplus(-mf)+_softplus(-mt))/2

def score_derivatives(scores):
    mf,mt=margins(scores);f=_negative_sigmoid(mf)/2;t=_negative_sigmoid(mt)/2
    return [[-f,f],[t,-t]]

def differentiable_classification_loss(scores,ops):
    """No detach/float conversion; caller checks finite 2x2 canonical scores."""
    mf=scores[0][0]-scores[0][1];mt=scores[1][1]-scores[1][0]
    return (ops.logaddexp(0.,-mf)+ops.logaddexp(0.,-mt))/2

def auxiliary_objective(unchanged_gspo_loss,scores,ops):
    return unchanged_gspo_loss+ALPHA*differentiable_classification_loss(scores,ops)

def update_kind(gspo_active,auxiliary_active):
    if type(gspo_active) is not bool or type(auxiliary_active) is not bool:raise ValueError('Explicit branch eligibility required')
    if gspo_active and auxiliary_active:return 'JOINT'
    if auxiliary_active:return 'AUXILIARY_ONLY'
    if gspo_active:return 'GSPO_ONLY'
    return 'NO_UPDATE'
