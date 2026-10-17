"""CPU-only pilot launcher checks on synthetic rows; imports no MLX/model runtime."""
import json
import sys
import pytest
from mri_rl import continuous_rl_pilot as pilot


def test_import_loads_no_native_runtime():
    assert not any(m in sys.modules for m in ('mlx.core', 'mlx_vlm', 'torch'))


def test_schedule_is_explicit_ordered_and_rejects_missing_or_duplicate(tmp_path):
    train = tmp_path / 'train.jsonl'
    train.write_text('\n'.join(json.dumps({'id': f'r{i}'}) for i in range(3)) + '\n')
    assert [r['id'] for r in pilot.load_schedule(train, ['r2', 'r0'])] == ['r2', 'r0']
    with pytest.raises(ValueError):
        pilot.load_schedule(train, ['r0', 'r0'])
    with pytest.raises(ValueError):
        pilot.load_schedule(train, ['r0', 'missing'])


def test_launch_configuration_binds_every_pilot_source(tmp_path):
    cfg = pilot.launch_configuration(output=tmp_path, seed=7, record_ids=['a', 'b'], group_size=2, mode='train')
    assert set(cfg['source_sha256']) == set(pilot.SOURCES)
    assert {'continuous_rl.py', 'mlx_continuous_gspo.py', 'continuous_rl_pilot.py'} <= set(cfg['source_sha256'])
    assert cfg['schedule_record_ids'] == ['a', 'b']
    changed = pilot.launch_configuration(output=tmp_path, seed=8, record_ids=['a', 'b'], group_size=2, mode='train')
    assert changed != cfg
