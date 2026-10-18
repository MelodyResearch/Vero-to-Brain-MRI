"""Researcher-authorized admission ablation; rewards and GSPO math are unchanged."""
import math

def accuracy_group_is_informative(grades):
    if len(grades) < 2:
        return False
    if any(not all(math.isfinite(float(g[k])) for k in ('accuracy','format','total')) or
           not math.isfinite(float(g.get('overlong',0))) for g in grades):
        raise ValueError('Nonfinite adaptive grade')
    valid = [g for g in grades if g['format'] == 1]
    return (len({g['accuracy'] for g in valid}) > 1
            and len({g['accuracy'] for g in grades}) > 1
            and len({g.get('overlong',0) for g in grades}) == 1
            and max(g['total'] for g in grades)-min(g['total'] for g in grades) > 1e-12)

def parent_anchor(path, expected_sha256):
    from pathlib import Path
    import json
    from mri_rl.continuous_rl import digest, require
    path=Path(path)
    require(path.is_dir() and not path.is_symlink(), 'Invalid parent checkpoint')
    require(digest(path/'receipt.json') == expected_sha256, 'Parent receipt digest changed')
    r=json.loads((path/'receipt.json').read_text())
    groups=r.get('completed_groups');updates=r.get('optimizer_updates')
    require(r.get('schema')=='continuous-rl-boundary-v1' and type(groups) is int and
            type(updates) is int and 0 <= updates <= groups and
            path.name==f'boundary_{groups:04d}', 'Invalid parent checkpoint counters')
    for name in ('adapter.safetensors','adapter_config.json','optimizer/manifest.json','optimizer/state.safetensors'):
        require(digest(path/name)==r['files_sha256'].get(name), 'Parent artifact digest changed')
    return {'receipt_sha256':expected_sha256, 'adapter_sha256':r['files_sha256']['adapter.safetensors'],
            'optimizer_manifest_sha256':r['files_sha256']['optimizer/manifest.json'],
            'optimizer_state_sha256':r['files_sha256']['optimizer/state.safetensors'],
            'groups':groups,'updates':updates,'original_binding':r['binding']}
