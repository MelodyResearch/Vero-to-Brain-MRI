"""Fixed candidate, not pure Vero: supervised v1/v2 mixture after CPU feasibility."""
from mri_rl import paired_interaction_signal_v1 as v1
from mri_rl import paired_classification_signal_v2 as v2

WEIGHT=.457130
ALPHA=.1
FEASIBILITY_SHA256='cdc935fe0f49ae7453334dad6dadd24971c44bbdb850079b0ae128a9745e3966'


def require_fixed_weight(w):
    if w != WEIGHT: raise ValueError('Weight is frozen by the TRAIN-gradient midpoint rule, not an accuracy tuning knob')


def loss(scores):
    return (1-WEIGHT)*v1.pair_loss(scores)+WEIGHT*v2.classification_loss(scores)


def score_derivatives(scores):
    a=v1.score_derivatives(scores);b=v2.score_derivatives(scores)
    return [[(1-WEIGHT)*x+WEIGHT*y for x,y in zip(r,s)] for r,s in zip(a,b)]


def differentiable_loss(scores,ops):
    mf=scores[0][0]-scores[0][1];mt=scores[1][1]-scores[1][0]
    return ((1-WEIGHT)*ops.logaddexp(0.,-(mf+mt)/2)
            +WEIGHT*(ops.logaddexp(0.,-mf)+ops.logaddexp(0.,-mt))/2)
