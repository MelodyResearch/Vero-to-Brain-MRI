"""CPU-only saved-gradient feasibility arithmetic; no model/native imports.

This module does not define or authorize a new training objective. Adam must be
applied to the mixed gradient, never interpolated from endpoint Adam deltas.
"""
import math
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def mix_arrays(one, two, w):
    require(math.isfinite(w) and 0 <= w <= 1, 'Finite convex weight required')
    require(one.shape == two.shape and one.dtype == two.dtype, 'Gradient inventory differs')
    require(np.isfinite(one).all() and np.isfinite(two).all(), 'Nonfinite gradient')
    return np.asarray(1-w, dtype=one.dtype)*one + np.asarray(w, dtype=one.dtype)*two


def adam_delta(g, first, second, lr):
    require(g.shape == first.shape == second.shape and g.dtype == first.dtype == second.dtype,
            'Same gradient/moment shape and dtype required')
    require(math.isfinite(lr) and lr > 0 and all(np.isfinite(x).all() for x in (g, first, second))
            and (second >= 0).all(), 'Invalid Adam input')
    cast = lambda x: np.asarray(x, dtype=g.dtype)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        mm = cast(.9)*first + cast(.1)*g
        vv = cast(.999)*second + cast(.001)*np.square(g)
        delta = -cast(lr)*mm/(np.sqrt(vv)+cast(1e-8))
    require(all(np.isfinite(x).all() for x in (mm, vv, delta)), 'Nonfinite Adam preview')
    return delta


def class_protected(dots, guard=0):
    require(math.isfinite(guard) and guard >= 0, 'Nonnegative numerical guard required')
    values = [dots[k] for k in ('mT', 'mF', 'mDallas')]
    return (all(math.isfinite(v) for v in values) and values[0] > 0
            and all(v >= guard for v in values))


def margin_coefficients(matrix):
    mf = matrix[0][0]-matrix[0][1]
    mt = matrix[1][1]-matrix[1][0]
    def negative_sigmoid(x):
        return math.exp(-x)/(1+math.exp(-x)) if x >= 0 else 1/(1+math.exp(x))
    return {'v1': [-.5*negative_sigmoid((mf+mt)/2)]*2,
            'v2': [-.5*negative_sigmoid(mf), -.5*negative_sigmoid(mt)]}


def coefficient_mix(coefficients, w):
    require(math.isfinite(w) and 0 <= w <= 1, 'Finite convex weight required')
    return [(1-w)*a+w*b for a, b in zip(coefficients['v1'], coefficients['v2'])]


def preview(arrays, coefficients, w, lr, *, dtype=np.float32, order='coefficients'):
    """Same-point F/T margin VJPs; mixed gradient→one Adam61 preview.

    Array values are read-only inputs. F64 variant and gradients-first variant
    are numerical sensitivity checks, never new experimental model runs.
    """
    require(dtype in (np.float32, np.float64) and order in ('coefficients', 'gradients'), 'Unknown arithmetic policy')
    f, t, d, first, second = [np.asarray(arrays[k], dtype=dtype) for k in ('F', 'T', 'D', 'm', 'v')]
    require(f.shape == t.shape == d.shape == first.shape == second.shape, 'Complete packed inventory required')
    cast = lambda x: np.asarray(x, dtype=dtype)
    cf, ct = coefficient_mix(coefficients, w)
    if order == 'coefficients':
        g = cast(cf)*f + cast(ct)*t
    else:
        a, b = coefficients['v1']; c, e = coefficients['v2']
        g = mix_arrays(cast(a)*f+cast(b)*t, cast(c)*f+cast(e)*t, w)
    scaled = cast(.1)*g
    delta = adam_delta(scaled, first, second, lr)
    baseline = adam_delta(np.zeros_like(scaled), first, second, lr)
    targets = {'mF': f, 'mT': t, 'mDallas': d}
    def dot(x, y):
        return float(np.dot(np.asarray(x, np.float64), np.asarray(y, np.float64)))
    directions = {name: {key: dot(vec, gradient) for key, gradient in targets.items()}
                  for name, vec in [('Adam_total', delta), ('Adam_g0_baseline', baseline),
                                    ('Adam_increment', delta-baseline), ('Euclidean', -scaled)]}
    norms = {'scaled_aux_gradient_l2': math.sqrt(dot(scaled, scaled)),
             'Adam_total_delta_l2': math.sqrt(dot(delta, delta)),
             'Adam_increment_l2': math.sqrt(dot(delta-baseline, delta-baseline))}
    require(all(math.isfinite(v) for row in directions.values() for v in row.values())
            and all(math.isfinite(v) for v in norms.values()), 'Nonfinite scalar reduction')
    return {'w': w, 'gradient_coefficients': [cf, ct], 'directions': directions,
            'norms': norms, 'arithmetic': {'dtype': np.dtype(dtype).name, 'order': order}}
