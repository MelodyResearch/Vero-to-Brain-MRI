"""Finite matched A/B preparation, not a native training implementation/permission."""

def proposal():
    return {'schema':'paired-prompt-matched-ab-preparation-v1','status':'PROPOSED_CPU_PREPARED_NATIVE_NOT_IMPLEMENTED_NOT_AUTHORIZED',
        'method_label':'PROPOSED VERO-STYLE ADAPTATION',
        'A':{'name':'SAME_PROMPT_LOCAL_VERO_STYLE_CONTROL','sequence_groups':[['F']*4,['T']*4]},
        'B':{'name':'PAIRED_PROMPT_ADAPTATION','sequence_groups':[['F','F','T','T']]*2},
        'both_arms_Dallas_groups':[['DALLAS_SAME']*4,['DALLAS_DIFFERENT']*4],
        'per_arm':{'attempted_groups':4,'responses_per_group':4,'sequence_responses':8,'Dallas_responses':8,
            'maximum_updates':4,'minimum_updates':0,'generated_response_tokens_cap':4096,
            'prompt_positions_reservation':4944,'charged_prompt_plus_returned_cap':9040,
            'generation_LM_API_call_reservation':4160,'full_TF_API_calls_cap':64,'backward_API_calls_cap':16,
            'TF_positions_cap':42240,'vision_embedding_API_calls_cap':80,
            'operation_start_deadline_seconds':1200,'simultaneous_models':1,'simultaneous_autograd_image_graphs':1},
        'all_four_attempts_required_for_complete_exposure':True,
        'native_per_group_response_loss_weight':.25,'offline_fixed_eight_block_response_loss_weight':.125,
        'per_group_policy':'Collect four true on-policy responses before that group update. Next group uses current policy; never silently call stale eight-response collection sequentially on-policy.',
        'matching':'Same parent/fullAdam/RNG, TRAIN inputs, F32, reward/calibration, inheritedAdam60 gate, LR/LoRA, class exposure and hard ceilings. Realized tokens/updates/trajectories may differ; report them, no refill.',
        'gate':'Unchanged uniform_balance_continuation_gate_v1.accuracy_group_is_informative, inherited Adam60 valid-format-subset rule; not historical18 all-format-strict or official Vero equality.',
        'reward_or_auxiliary_change':False,'native_execution_authorized':False,
        'partial_failure':'Preserve partial state/cost; unequal incomplete exposure invalidates complete matched contrast. No automatic retry/replacement/resampling or extra updates.',
        'native_blockers':['Mixed-context F32 generation/scorer/gradient backend and crash-consistent transaction implementation are not implemented by this CPU preparation.',
            'Each response must retain its own image/prompt/oldlogps/reward; historical single-self.active backend cannot be reused unchanged.',
            'Bound real input lengths, source/checkpoint/state hashes, all API counters/resources and matched F32 evaluation; independent exact scope review and fresh explicit human permission required.'],
        'accuracy_claim':False,'model_loads':0,'optimizer_updates':0}


def refuse_native_execution(_untrusted_receipt=None):
    raise ValueError('PREPARATION_ONLY: no native execution implementation or human launch authorization; local approval flags are not permission')
