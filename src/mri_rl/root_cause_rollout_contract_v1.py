"""CPU-only validation of a future rollout/length-reward configuration.

This does not change the historical reward, choose a new length penalty, launch
models or authorize training. Callers must explicitly disable an unreachable
penalty or supply a separately justified reachable contract.
"""
from __future__ import annotations


def validate_length_contract(*, rollout_cap: int, penalty_max_tokens: int,
                             buffer_tokens: int, mode: str) -> dict:
    for name, value in (('rollout_cap', rollout_cap),
                        ('penalty_max_tokens', penalty_max_tokens),
                        ('buffer_tokens', buffer_tokens)):
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if buffer_tokens > penalty_max_tokens:
        raise ValueError('buffer_tokens exceeds penalty_max_tokens')
    if mode not in {'enabled', 'disabled_explicit'}:
        raise ValueError('mode must explicitly enable or disable the penalty')
    first_negative = penalty_max_tokens - buffer_tokens + 1
    reachable = rollout_cap >= first_negative
    if mode == 'enabled' and not reachable:
        raise ValueError('enabled length penalty is unreachable within rollout_cap')
    return {
        'rollout_cap': rollout_cap,
        'penalty_max_tokens': penalty_max_tokens,
        'buffer_tokens': buffer_tokens,
        'first_negative_token': first_negative,
        'reachable': reachable,
        'effective_penalty_enabled': mode == 'enabled',
        'mode': mode,
        'training_authorized': False,
        'historical_reward_modified': False,
        'scope': 'Configuration consistency only; not evidence of better MRI accuracy.',
    }
