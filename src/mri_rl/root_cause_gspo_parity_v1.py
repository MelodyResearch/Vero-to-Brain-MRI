"""CPU source audit and mechanism counterexamples; never imports a model runtime.

Reuses only AST-selected NumPy AD definitions from the accepted 31-case audit.
The new tests concern causal interpretation, not another loss equivalence sweep.
No patients, model weights, inference, optimizer steps, or evaluation rows are read.
"""
from __future__ import annotations
import ast
import hashlib
import math
from pathlib import Path
import subprocess
import types

OFFICIAL_SHA = '025ca3c800fc95224c86323c1f7fd33dceed3090'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cpu_loss(root, official_root):
    """Extract audited tiny AD helpers and actual native/official loss ASTs only."""
    import numpy as np
    helper = Path(root) / 'scripts/diagnose_gspo_loss_cpu.py'
    tree = ast.parse(helper.read_text())
    keep = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
            and n.name in {'D', 'API', 'extracted'}]
    if len(keep) != 3:
        raise ValueError('Accepted CPU AD helper layout changed')
    ns = {'np': np, 'ast': ast, 'hashlib': hashlib,
          'NATIVE': Path(root) / 'src/mri_rl/mlx_continuous_gspo.py',
          'OFF': Path(official_root) / 'vero-rl/verl/trainer/ppo/core_algos.py'}
    exec(compile(ast.Module(body=keep, type_ignores=[]), '<CPU_AD_definitions_only>', 'exec'), ns)
    native, official, hashes = ns['extracted']()
    return ns['D'], native, official, hashes


def native_fixture(root, official_root, lengths, advantages, sensitivities, mean_log_ratios=None):
    """Differentiate actual loss wrt one scalar toy parameter, with detached old logp.

    sensitivities[i][t] = d log p(token_t) / d theta at theta=0.
    Stop-gradient values must stay fixed when taking finite differences.
    """
    import numpy as np
    D, native, official, hashes = load_cpu_loss(root, official_root)
    D.dim = 1
    n, width = len(lengths), max(lengths)
    ratios = mean_log_ratios or [0.] * n
    old = np.full((n, width), -4.)
    current = old.copy(); tangent = np.zeros((n, width, 1)); mask = np.zeros((n, width))
    result = D(0.)
    for i, length in enumerate(lengths):
        current[i, :length] += ratios[i]
        tangent[i, :length, 0] = sensitivities[i]
        mask[i, :length] = 1
    value = D(current, tangent)
    for i, length in enumerate(lengths):
        native['self'] = types.SimpleNamespace(_logps=lambda _, i=i, length=length: value[i, :length])
        result += native['loss_fn'](None, None, D(old[i, :length]), advantages[i]) / n
    reference = official(D(old), value, np.array(advantages)[:, None], mask, .0003, .0004)
    assert abs(float(result.v-reference.v)) < 1e-12
    assert abs(float(result.j[0]-reference.j[0])) < 1e-12
    return {'loss': float(result.v), 'gradient': float(result.j[0]),
            'forward_ratios': [math.exp(min(x, 10.)) for x in ratios],
            'native_official_gradient_error': float(abs(result.j[0]-reference.j[0])),
            'ast_sha256': hashes}


def sample_advantages(rewards):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((x-mean)**2 for x in rewards) / (len(rewards)-1))
    return [(x-mean)/(std+1e-6) for x in rewards]


def detached_surrogate_fd(advantages, sensitivities, step=1e-7):
    """At on-policy theta=0 freeze detached forward values before perturbing theta."""
    def f(theta):
        return sum(-a*sum(math.exp(theta*s) for s in seq)/len(seq)
                   for a, seq in zip(advantages, sensitivities))/len(advantages)
    return (f(step)-f(-step))/(2*step)


def mechanisms(root, official_root):
    on = native_fixture(root, official_root, [1, 1], [1., -1.], [[2.], [-1.]])
    on['frozen_detach_finite_difference'] = detached_surrogate_fd([1., -1.], [[2.], [-1.]])
    on['naive_redetach_each_perturbation_loss'] = 0.
    assert on['loss'] == 0. and abs(on['gradient'] + 1.5) < 1e-12
    assert abs(on['gradient']-on['frozen_detach_finite_difference']) < 1e-8
    stale = native_fixture(root, official_root, [1, 1], [1., -1.], [[2.], [-1.]], [.001, -.001])
    assert stale['gradient'] == 0.
    short = native_fixture(root, official_root, [2, 2], [1., -1.], [[0., 1.], [0., -1.]])
    long = native_fixture(root, official_root, [10, 10], [1., -1.], [[0.]*9+[1.], [0.]*9+[-1.]])
    assert abs(short['gradient']/long['gradient']-5.) < 1e-12
    cancel = native_fixture(root, official_root, [1, 1], [1., -1.], [[1.], [1.]])
    assert cancel['gradient'] == 0.
    format_a = sample_advantages([0., .2, 0., .2])
    task_a = sample_advantages([0., .8, 0., .8])
    assert min(abs(x) for x in format_a) > .86
    return {
        'ratio_one_loss_zero_gradient_nonzero': {**on, 'interpretation':
            'Forward ratio=1 and mean advantage=0 do not imply zero gradient. Current logp derivative survives detach; clipping is inactive.'},
        'stale_minibatch_clipping_counterexample': {**stale, 'interpretation':
            'For favorable movement of 0.001 in mean logp, both examples clip and gradient=0; the matching on-policy fixture has gradient=-1.5. Scheduling can matter, but no local bug is established.'},
        'sequence_credit_dilution': {'length2': short['gradient'], 'length10': long['gradient'], 'factor': 5.,
            'interpretation': 'Holding only label-token sensitivity nonzero, adding invariant wrapper tokens dilutes label gradient by response length. This is expected seq-mean-token-mean behavior in both sources.'},
        'image_derivative_does_not_prove_rl_signal': {**cancel,
            'individual_logp_image_derivatives': [1., 1.], 'single_target_ce_image_derivative': -1.,
            'interpretation': 'A nonzero per-response/CE image derivative can cancel under signed group advantages. CE evidence cannot establish full visual GSPO credit.'},
        'format_only_group_normalization': {'format_only_advantages': format_a, 'task_only_advantages': task_a,
            'max_difference': max(abs(a-b) for a,b in zip(format_a,task_a)),
            'interpretation': 'A varying .2 format term can produce near-unit standardized credit even when task accuracy is fixed. Tier C all-full-format/task-varies gate excludes this fixture; historical/unfiltered groups differ.'},
        'group_size_active_probability': {'assumption': 'IID binary task success, all full format, equal overlong; analytic illustration only',
            'p_0_1_K4': 1-.1**4-.9**4, 'p_0_1_K8': 1-.1**8-.9**8,
            'p_0_5_K4': 1-2*.5**4, 'p_0_5_K8': 1-2*.5**8,
            'strict_all_format_q_0_8_p_0_5_K4': .8**4*(1-2*.5**4),
            'strict_all_format_q_0_8_p_0_5_K8': .8**8*(1-2*.5**8),
            'strict_gate_assumption': 'Each response full-format independently with probability q=.8, independent of IID binary success p=.5; eligibility=q^K*[1-p^K-(1-p)^K].',
            'binary_label_information_upper_bound_bits': 1.,
            'uniform_attempts_active_share_p_0_1_vs_0_5_K4': (1-.1**4-.9**4)/((1-.1**4-.9**4)+(1-2*.5**4)),
            'interpretation': 'K changes mixed-success probability. Requiring every response full-format can make K8 LESS eligible than K4 under independent q=.8; uniform attempted categories can yield unequal active updates. Binary categorical label entropy is at most1bit, which does not bound the image reasoning difficulty. These are not observed MRI rates.'},
        'teacher_force_identity_blind_spot': {'generation_prompt_ids': [1,2,3], 'teacher_force_prompt_ids': [1,9,3],
            'equal_length_check_passes': True, 'old_current_recompute_error': 0.,
            'interpretation': 'Equal prompt length plus recomputed old=current cannot detect different generation inputs. This is a synthetic counterexample to sufficiency of checks, not evidence actual generation inputs differ.'}}


def source_ledger(root, official_root):
    """Return exact source hashes and locations without importing either runtime."""
    root, official_root = Path(root), Path(official_root)
    head = subprocess.check_output(['git','-C',str(official_root),'rev-parse','HEAD'], text=True).strip()
    if head != OFFICIAL_SHA:
        raise ValueError('Official checkout HEAD differs from historical contract pin')
    subprocess.run(['git','-C',str(official_root),'diff','--quiet','HEAD','--'], check=True)
    historical_tree = ast.parse((root/'src/mri_rl/mlx_gspo_smoke.py').read_text())
    historical_pin = next(n.value.value for n in historical_tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'VERO_SHA' for t in n.targets))
    if historical_pin != OFFICIAL_SHA:
        raise ValueError('Historical training source pin changed')
    records = []
    def record(claim, opath, needle, local, judgment):
        text = (official_root/opath).read_text(); lines = text.splitlines()
        line = next(i for i,s in enumerate(lines,1) if needle in s)
        records.append({'claim': claim, 'evidence_state': 'DIRECT_SOURCE_OBSERVATION',
            'official_url': f'https://github.com/zlab-princeton/vero/blob/{head}/{opath}#L{line}',
            'official_source_sha256': sha(official_root/opath), 'official_location': f'{opath}:{line}',
            'local_source': local, 'judgment': judgment})
    base = 'vero-rl/'
    record('Sequence geometric mean ratio, current derivative with detached forward value; asymmetric clipping; sequence mean aggregation',base+'verl/trainer/ppo/core_algos.py','def compute_policy_loss_gspo','src/mri_rl/mlx_continuous_gspo.py:update.loss_fn','FORMULA_PARITY under CPU AD; real MLX visual backward NOT_VERIFIED')
    record('Official on-policy fast path needs one minibatch and one epoch',base+'verl/workers/actor/dp_actor.py','on_policy =','src/mri_rl/mlx_continuous_gspo.py:update','Official default has 256 prompts /128 prompt minibatch=2 minibatches, ppo_epochs1; local one K-group per update. Clipping initially inactive locally; gradient survives.')
    record('Official minibatch size normalized by K and data parallel world size',base+'verl/workers/fsdp_workers.py','self.config.actor.ppo_mini_batch_size *=','src/mri_rl/tier_c_entry.py:train_arm','Default8 GPUs/K8 gives128 response minibatch per rank; 256 response rollout per rank ->2 actor minibatches. No CUDA run performed.')
    record('K8, LR1e-6, warmup40, uniform six categories, freeze_vision_tower=false',base+'examples/model_runs/config/gspo_llmjudge_shared.yaml','per_batch_domain_weights:','src/mri_rl/tier_c_entry.py:train_arm','Local K4 (plan), LR1e-5 constant, language LoRA4/alpha8 frozen4bit base; engineering differences, not proven defects.')
    record('Official AdamW, weight decay defaults .01, constant schedule with warmup',base+'verl/workers/fsdp_workers.py','actor_optimizer = optim.AdamW','src/mri_rl/mlx_continuous_gspo.py:__init__','Local Adam no bias correction, no decay/scheduler; materially different update geometry. No optimizer steps in audit.')
    record('Category floor/largest-remainder blocks with exhaustion reshuffle',base+'verl/utils/dataset/rl_dataset.py','class PerBatchDomainSampler','src/mri_rl/vero_category_sampler.py; two_task_training.py:build_schedule','Tier C uses256-prompt logical blocks; serial updates are not distributed minibatch parity; prefix realized balance must be reported.')
    record('Data curation removes exclusively binary, <1k examples and avg<200k pixels','docs/DATA.md','Heuristic filtering','MRI task-design distinction','Original general-vision data heuristics not satisfied merely by binary MRI verifier validity; recipe transfer is restricted, not official data parity.')
    record('Accuracy gated to zero when format=0',base+'vero_reward/math_verify_reward_type_boxed.py','if formatting == 0.0:','src/mri_rl/vero_mri_reward.py; continuous_rl.py','Local reward routes are MRI-specific; Tier C adds all-full-format/task-varies/equal-overlong update filter. Uniform attempted categories need not mean uniform active updates.')
    record('Official Qwen3 max pixels16,777,216 and prompt/response18432',base+'examples/model_runs/config/gspo_qwen3vl.yaml','max_pixels:','src/mri_rl/mlx_continuous_gspo.py:rollout/_logps','Local prompt bound2048; generation max_tokens supplied by plan, visual processing token counts require actual runtime receipt. Response loss averages only response tokens; prompt/image count does not enter its denominator.')
    record('Official configured overlong reward uses max_resp_len16384 minus2048 buffer, independently of data response cap18432',base+'verl/workers/reward_manager/vero_vllm_judge.py','expected_len = self.max_resp_len - overlong_buffer_len','src/mri_rl/mlx_gspo_smoke.py:overlong_penalty','SOURCE_CONFIG_DISCREPANCY: shared reward max16384 => first negative14337; Qwen3 data cap18432 does not override reward max. Local direct profile18432 => first negative16385. Both zero for current local short caps. No historical length rerun; no frozen edits.')
    record('Qwen3 launcher selects gspo_qwen3vl without reward max length CLI override',base+'examples/model_runs/run_gspo_qwen3vl_instruct_mix_all_llmjudge.sh',"--config-name='gspo_qwen3vl'",'src/mri_rl/tier_c_entry.py:train_arm','Direct source observation of default launcher, not a recovered resolved Hydra config from original authors training.')
    record('Training documentation summarizes source recipe','docs/TRAINING.md','Rollout samples per prompt','scripts/diagnose_gspo_loss_cpu.py','Historical local VERO_SHA constant and docs URLs match actual clean official HEAD025ca3; no newer source substituted.')
    return {'actual_official_head':head, 'historical_training_contract_sha':historical_pin,
            'same_historical_and_actual_head':True, 'records':records,
            'local_sha256': {p:sha(root/p) for p in ['src/mri_rl/mlx_continuous_gspo.py','src/mri_rl/tier_c_entry.py','src/mri_rl/two_task_training.py','src/mri_rl/continuous_rl.py','src/mri_rl/vero_category_sampler.py','src/mri_rl/mlx_gspo_smoke.py']}}


def build_report(root, official_root):
    return {'schema':'root-cause-gspo-parity-v1','status':'TESTED_CPU','research_tag':'PRIMARY_VERO_TRANSFER',
        'evidence_tier':'SOURCE_AUDIT_AND_TINY_CPU_AST_COUNTEREXAMPLES',
        'mission_alignment': {'main_question':'Separate Vero recipe mismatch from mathematical GSPO defects before controlled MRI transfer.',
            'claim_falsifier':'On-policy ratio1 can have nonzero gradient; falsified by zero extracted derivative for heterogeneous response sensitivities.',
            'isolation':'New audit files only; historical/frozen sources and experiments unchanged.',
            'provenance':'Pinned source hashes and toy fixtures; no medical or evaluation rows read.',
            'baseline':'Actual pinned official loss AST versus actual native loss AST plus analytic finite differences.',
            'resource_decision':'CPU one-thread only; no heavyweight runtime, inference, optimizer update or training.'},
        'source_ledger':source_ledger(root, official_root), 'counterexamples':mechanisms(root, official_root),
        'minimal_proven_fixes':[], 'fix_decision':'NO_FROZEN_SOURCE_EDIT: no native loss defect proved by these fixtures.',
        'native_rollout_scorer_visual_gspo_end_to_end':'NOT_VERIFIED',
        'limitations':['CPU AD preserves source stop-gradient expressions but does not execute MLX/CUDA backward or nonsmooth backend ties.',
            'Recomputed old/current logp alignment cannot certify sampling-distribution logp, KV-cache equivalence, generation prompt token identity, processor equivalence or visual gradient paths.',
            'Teacher-forced scoring includes captured visual kwargs and causal response positions; real generation/teacher-force equality requires a separately authorized runtime intervention.',
            'CE image-gradient evidence does not prove signed advantage-weighted full visual GSPO credit.',
            'K/LR/LoRA/vision/batching/token differences are documented recipe differences, not established causes of past MRI negative results.',
            'No model weights, patient records, final100 or heldout test content accessed; no real optimizer update, inference or training performed.']}
