from types import SimpleNamespace
from pathlib import Path
import pytest
import mri_rl.mlx_gspo_smoke as smoke


def test_task_activation_excludes_format_and_length_only_variance():
    correct = {'accuracy': 1.0, 'format': 1.0, 'overlong': 0.0}
    incorrect = {'accuracy': 0.0, 'format': 1.0, 'overlong': 0.0}
    invalid = {'accuracy': 0.0, 'format': 0.0, 'overlong': 0.0}
    assert smoke.task_group_is_informative([correct, incorrect])
    assert not smoke.task_group_is_informative([correct, invalid])
    assert not smoke.task_group_is_informative([correct, correct])
    assert not smoke.task_group_is_informative([incorrect, invalid])
    assert not smoke.task_group_is_informative([correct, dict(incorrect, overlong=-0.2)])


def test_smoke_policy_rejection_precedes_mlx_import_and_output(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, 'load_contract', lambda p: SimpleNamespace(
        system_instruction='matched', system_instruction_sha256='sha'))
    called = []
    def reject(**kwargs):
        called.append(kwargs)
        raise ValueError('review refused')
    monkeypatch.setattr(smoke, 'require_launch_preflight', reject)
    root = Path(smoke.__file__).resolve().parents[2]
    output = root / 'local_data/test_never_created_activation'
    with pytest.raises(ValueError, match='review refused'):
        smoke.run(tmp_path/'base',tmp_path/'train',output,'mri_sequence',2,256,7,
                  adapter_checkpoint=tmp_path/'adapter',reward_profile='direct_vero',
                  contract_path=tmp_path/'contract')
    assert not output.exists()
    assert called[0]['execution_tier'] == 'B'
    assert called[0]['launch_budget']['optimizer_steps'] == 1
    assert called[0]['expected_temperature'] == 1.0
