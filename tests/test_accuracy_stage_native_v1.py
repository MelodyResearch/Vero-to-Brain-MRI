"""Synthetic bridge checks. No MLX, model or real record is imported/read."""
import copy
from pathlib import Path
import pytest
from mri_rl import accuracy_stage_native_v1 as bridge
from mri_rl import paired_prompt_groups_v1 as ppg


def test_historical_loader_generate_attribute_cannot_shadow_stage_method(monkeypatch):
    def historical_init(self, *args, **kwargs):
        self.generate = lambda *a, **k: 'WRONG_LEGACY_ENTRY'
    monkeypatch.setattr(bridge.SharedRolloutBackend, '__init__', historical_init)
    backend = bridge.StageNativeBackend('adapter', {})
    assert getattr(backend.generate, '__func__', None) is bridge.StageNativeBackend.generate


def test_dallas_routing_is_per_response_without_rewriting_rewards():
    from mri_rl.multistep_gspo_core_v1 import DContext
    ctx = DContext('row', 'pair', 'D', *(['a'*64]*5))
    grade = ppg.Grade(0., 1., .2, 0., 0.)
    # Use simple attributes; the routing helper is intentionally independent of tensors.
    from types import SimpleNamespace
    rs = [SimpleNamespace(response_id='D1_0', context=ctx), SimpleNamespace(response_id='D2_0', context=ctx)]
    assert [bridge.context_key(r) for r in rs] == ['D1', 'D2']
    assert ctx.modality == 'D'
    with pytest.raises(ValueError): bridge.context_key(SimpleNamespace(response_id='D3_0', context=ctx))


def test_unit_binding_cannot_silently_reuse_previous_patient():
    fake = bridge.StageNativeBackend.__new__(bridge.StageNativeBackend)
    fake._active = {'old': True}; fake._active_modality = 'F'; fake._recorded = {'old': 1}
    fake._stage_prompt_lengths = {'mri_sequence': 213}
    unit = {'index': 1, 'contexts': [{'key': 'F', 'row': {'id':'one','ability':'mri_sequence'}}]}
    fake.bind_unit(unit)
    assert fake.rows == {'F': unit['contexts'][0]['row']}
    assert fake._active is None and fake._active_modality is None and fake._recorded is None


def test_arm_boundary_restore_does_not_reinitialize_parent_per_unit(tmp_path):
    # Orchestration is tested with the actual core + its deterministic CPU backend.
    import test_multistep_gspo_v1 as fixture
    from mri_rl import multistep_gspo_core_v1 as ms
    units=ms.make_units(fixture.make_plan(2)); lengths=fixture.LENGTHS
    budget=ms.derive_arm_budget(units,lengths,maximum_seconds=60)
    b=fixture.FakeBackend(always=None)
    parent=b.snapshot()
    common=ms.make_common_rollout(units[0],b,budget,prompt_lengths=lengths,reward_policy_sha256=fixture.REWARD,output=tmp_path/'common')
    b.restore(parent)
    result=bridge.run_arms(backend=b,units=units,budget=budget,prompt_lengths=lengths,
        reward_policy_sha256=fixture.REWARD,common=common,parent_snapshot=parent,output=tmp_path/'run',before=lambda name:None)
    assert set(result)=={'A','B'}
    assert all(r['units_committed']==4 for r in result.values())
    assert b.policy_sha()==fixture.FakeBackend().policy_sha()
    assert result['A']['optimizer_updates']<=4
    terminal=tmp_path/'run'/'arm_A'/'unit_0003'
    assert (terminal/'own_old_PRIVATE.json').is_file()
    import json
    receipt=json.loads((terminal/'receipt_PRIVATE.json').read_text())
    assert 'own_old_PRIVATE.json' in receipt['files_sha256']


def test_native_entry_refuses_missing_authorization_before_runtime_read():
    with pytest.raises(ValueError, match='authorization'):
        bridge.run_stage({}, {}, None, None)
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='authorization'):
        bridge.run_stage({}, {}, SimpleNamespace(check=lambda phase: True), lambda phase:None)


def test_close_drops_full_parameter_and_adam_snapshot_references(monkeypatch):
    monkeypatch.setattr(bridge.SharedRolloutBackend, 'close', lambda self: setattr(self, 'closed', True))
    b=bridge.StageNativeBackend.__new__(bridge.StageNativeBackend)
    b._unit_snapshot={'parameters':object(), 'optimizer':object()}
    b._active={'inputs':object()}
    b.close()
    assert b.closed and b._unit_snapshot is None and b._active is None
