#!/usr/bin/env python3
"""Finite CPU-only mixture feasibility, using preserved native VJPs/Adam60."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mri_rl import mixture_adam_cpu_v3 as mix
from mri_rl.likelihood_cache import _header


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def moment_map(node, path=''):
    out = {}
    if node['kind'] == 'dict':
        d = dict(node['items'])
        if 'm' in d and 'v' in d: out[path] = (d['m']['key'], d['v']['key'])
        else:
            for key, child in node['items']: out.update(moment_map(child, (path+'.'+key).lstrip('.')))
    elif node['kind'] in ('list', 'tuple'):
        for key, child in enumerate(node['items']): out.update(moment_map(child, (path+'.'+str(key)).lstrip('.')))
    return out


def packed(path, keys):
    header, offset = _header(path)
    rows = []
    for key in keys:
        h = header[key]; a, b = h['data_offsets']
        assert h['dtype'] == 'F32' and b-a == 4*math.prod(h['shape'])
        assert 0 <= a <= b <= Path(path).stat().st_size-offset
        rows.append(np.memmap(path, dtype='<f4', mode='r', offset=offset+a, shape=tuple(h['shape'])).ravel())
    result = np.concatenate(rows); result.flags.writeable = False
    return result


def load_saved(base):
    scope = read(base/'scope_v4_PRIVATE.json'); out = Path(scope['output'])
    review = read(base.parents[1]/'results/CLASS_MARGIN_V2_ACTUAL_INDEPENDENT_REVIEW_V1.json')
    assert sha(base.parents[1]/'results/CLASS_MARGIN_V2_ACTUAL_INDEPENDENT_REVIEW_V1.json') == 'daf7829404381f02e06e51754eb7b50cb3b2d459fa17138f48887419e2dba087'
    assert sha(out/'complete_PRIVATE.json') == review['actual_proof_sha256']
    complete = read(out/'complete_PRIVATE.json'); spec = read(base/'spec_v1_PRIVATE.json'); runtime = read(spec['runtime_spec'])
    parent = Path(runtime['parent_checkpoint']); manifest = read(parent/'optimizer/manifest.json')
    statefile = parent/'optimizer/state.safetensors'
    assert sha(statefile) == manifest['tensors_sha256'] == scope['parent']['optimizer_state_sha256']
    assert sha(parent/'optimizer/manifest.json') == scope['parent']['optimizer_manifest_sha256']
    moments = moment_map(dict(manifest['layout']['items'])['optimizer']); keys = sorted(moments)
    arrays = {}; bindings = {str(statefile): sha(statefile), str(parent/'optimizer/manifest.json'): sha(parent/'optimizer/manifest.json')}
    for short, name in [('F','F_REAL'), ('T','T_REAL'), ('D','DALLAS_REAL')]:
        path = out/(name+'_margin_gradient_PRIVATE.safetensors')
        assert sha(path) == review['independent_saved_gradient_inventory'][name]['file_sha256']
        assert set(_header(path)[0]) == set(keys) and len(keys) == 504
        arrays[short] = packed(path, keys); bindings[str(path)] = sha(path)
    arrays['m'] = packed(statefile, [moments[k][0] for k in keys])
    arrays['v'] = packed(statefile, [moments[k][1] for k in keys])
    matrix = complete['analysis']['conditions']['REAL_PAIR']['scores_PRIVATE']
    return arrays, mix.margin_coefficients(matrix), complete, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--private-output', type=Path, required=True)
    parser.add_argument('--public-output', type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    arrays, coefficients, native, bindings = load_saved(args.base)
    lr = native['analysis']['actual_optimizer_profile']['learning_rate']; cache = {}
    def evaluate(w):
        key = float(w)
        if key not in cache:
            cache[key] = mix.preview(arrays, coefficients, key, lr)
        return cache[key]
    endpoints = [evaluate(0.), evaluate(1.)]
    for ver, result in zip(('v1','v2'), endpoints):
        original = native['analysis']['conditions']['REAL_PAIR']['versions'][ver]
        for target in ('mT','mF','mDallas'):
            assert abs(result['directions']['Adam_total'][target]-original['directions']['Adam61_total_delta'][target]) < 1e-10
    # A NEW conservative envelope, not an inherited clipping/safety threshold.
    caps = {key: max(r['norms'][key] for r in endpoints) for key in ('scaled_aux_gradient_l2','Adam_total_delta_l2')}
    caps['Adam_total_delta_l2'] = max(caps['Adam_total_delta_l2'], native['analysis']['conditions']['REAL_PAIR']['versions']['v1']['parameter_direction_l2']['Adam61_zero_new_gradient_baseline'])
    def passes(w, guard):
        r = evaluate(w)
        return mix.class_protected(r['directions']['Adam_total'], guard) and all(r['norms'][k] <= v*(1+1e-6) for k,v in caps.items())
    grid = [k/16 for k in range(17)]
    def refine(left, right, guard):
        initial = passes(left, guard)
        for _ in range(24):
            if right-left <= 1e-7: break
            mid = (left+right)/2
            if passes(mid, guard) == initial: left = mid
            else: right = mid
        return {'bracket': [left,right], 'left_passes': passes(left,guard), 'right_passes': passes(right,guard)}
    def components(guard):
        flags = [passes(w,guard) for w in grid]
        boundaries = [refine(a,b,guard) for a,b,x,y in zip(grid,grid[1:],flags,flags[1:]) if x != y]
        ranges=[]; low=0. if flags[0] else None
        for bound in boundaries:
            left,right=bound['bracket']
            if bound['right_passes']: low=right
            else:
                ranges.append([low,left]); low=None
        if flags[-1]: ranges.append([low,1.])
        return {'guard':guard,'initial_grid_passes':flags,'transition_brackets':boundaries,'verified_inside_components':ranges}
    zero = components(0.); robust = components(1e-6)
    candidate = None; sensitivity = {}; interior=[]
    if robust['verified_inside_components']:
        low,high=max(robust['verified_inside_components'],key=lambda pair:pair[1]-pair[0])
        interior=[low+(high-low)*k/32 for k in range(33)]
        assert all(passes(w,1e-6) for w in interior)
        w=round((low+high)/2,6)
        assert passes(w,1e-6); candidate=evaluate(w)
        for name,dtype,order in [('F64',np.float64,'coefficients'),('F32_gradients_first',np.float32,'gradients')]:
            sensitivity[name]=mix.preview(arrays,coefficients,w,lr,dtype=dtype,order=order)
        sensitivity['guard_1e-5_candidate_passes']=passes(w,1e-5)
        sensitivity['guard_1e-4_candidate_passes']=passes(w,1e-4)
    result={'schema':'saved-Adam60-mixture-feasibility-cpu-v3','status':'FEASIBLE_LOCAL_TRAIN_DIRECTION' if candidate else 'NO FEASIBLE CLASS-PROTECTED DIRECTION',
        'source_parent_commit':'a6ebe8d12e0dc7a8e1633acb64e5730cc5c6897b','coefficients':coefficients,'alpha':.1,
        'zero_boundary_analysis':zero,'robust_analysis':robust,'candidate':candidate,'arithmetic_sensitivity':sensitivity,
        'new_endpoint_norm_envelope':caps,'original_explicit_norm_cap':'NOT_PRESENT_FINITE_NONZERO_ONLY',
        'initial_grid':grid,'interior_33_checked_points':interior,'evaluations':len(cache),'wall_seconds':time.monotonic()-started,
        'model_loads':0,'new_native_forwards':0,'new_native_backwards':0,'optimizer_updates':0,'training_runs':0,
        'limitations':['ONE fixed TRAIN modality pair and ONE Dallas case, not wider TRAIN accuracy or heldout benefit.',
            'TOTAL retained-Adam protection is distinct from new-aux increment protection; report both.',
            'Finite grid and bisection find sampled connected components, not a global proof excluding narrow unsampled components.',
            'Direction dot products are not a finite evaluated update or accuracy change; no native training qualification.',
            'Endpoint norm envelope is a NEW analysis guard, not an inherited physical/model safety qualification.'],
        'final_or_holdout_reads':False,'training_authorized':False}
    private={**result,'evidence_sha256_PRIVATE':bindings,'all_evaluations':[cache[k] for k in sorted(cache)]}
    for path,value in [(args.private_output,private),(args.public_output,result)]:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    assert not any(k.split('.')[0] in ('mlx','mlx_vlm','torch','transformers') for k in sys.modules)
    print(json.dumps({'status':result['status'],'candidate':candidate,'robust_analysis':robust,'evaluations':len(cache),'wall_seconds':result['wall_seconds'],'public_sha256':sha(args.public_output)}))


if __name__ == '__main__': main()
