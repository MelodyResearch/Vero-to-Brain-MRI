"""Synthetic CPU tests for V2_NATIVE_READONLY_V1. No MLX/torch/JAX, no model, no generation, no gradient/optimizer.

A meta-path blocker is installed BEFORE any project import: if any code path tried to import a heavyweight
runtime, the import fails loudly instead of loading it. Fake backends/ops only; nothing here is MRI evidence.
"""
from __future__ import annotations
import ast
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

HEAVY = ('mlx', 'mlx_vlm', 'mlx_lm', 'torch', 'jax', 'jaxlib', 'tensorflow')


class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in HEAVY:
            raise ImportError('blocked in synthetic CPU tests: ' + name)
        return None


assert not [m for m in sys.modules if m.split('.')[0] in HEAVY], 'heavy runtime already imported before tests'
sys.meta_path.insert(0, _Block())

from mri_rl import v2_native_readonly_v1 as nr                    # noqa: E402
from mri_rl import v2_native_readonly_entry_v1 as ne              # noqa: E402
from mri_rl import v2_native_readonly_backend_v1 as nb            # noqa: E402
from mri_rl import shared_rollout_gspo_core_v1 as shared          # noqa: E402
from mri_rl import safe_micro_dynamic_resources_v1 as resources   # noqa: E402
from mri_rl.safe_micro_journal_v1 import Journal, Indeterminate   # noqa: E402
from mri_rl.v2_regression_diagnosis_v1 import TokenDecoder, bytes_to_unicode   # noqa: E402

SRC = Path(nr.__file__).resolve().parent
SEQ, DAL = 'mri_sequence', 'mri_visit_correspondence'
LENGTHS = {SEQ: 230, DAL: 405}


# ================================================================ synthetic tokenizer (saved token ids -> bytes)
PIECES = ['<think>', 'short', '</think>', '<answer>', '\\boxed{', 'FL', 'AIR', '}', '</answer>', 'contrast', '-en', 'hanced', ' T', '1',
          'same', 'different', 'maybe', '\\boxed', '{FL', 'AIR}', 'x', '1}', '\\text{', ' ']
EOS = 999


def _encode(piece):
    inverse = {b: ch for ch, b in bytes_to_unicode().items()}
    return ''.join(inverse[b] for b in piece.encode())


DECODER = TokenDecoder({_encode(p): i for i, p in enumerate(PIECES)}, [{'id': EOS, 'content': '<|im_end|>'}])
ID = {p: i for i, p in enumerate(PIECES)}


def toks(*pieces, eos=True):
    return [ID[p] for p in pieces] + ([EOS] if eos else [])


FLAIR_OK = toks('<think>', 'short', '</think>', '<answer>', '\\boxed{', 'FL', 'AIR', '}', '</answer>')
T1_OK = toks('<think>', 'short', '</think>', '<answer>', '\\boxed{', 'contrast', '-en', 'hanced', ' T', '1', '}', '</answer>')
NO_BOX = toks('<think>', 'short', '</think>', '<answer>', 'maybe', '</answer>')
MERGED = toks('<think>', 'short', '</think>', '<answer>', '\\boxed', '{FL', 'AIR}', '</answer>')
SAME = toks('<think>', 'short', '</think>', '<answer>', '\\boxed{', 'same', '}', '</answer>')
THINK, ANSWER = ('<think>', 'short', '</think>', '<answer>'), ('</answer>',)
F_BOX, T_BOX = ('\\boxed{', 'FL', 'AIR', '}'), ('\\boxed{', 'contrast', '-en', 'hanced', ' T', '1', '}')
REASONING_BOX_THEN_T1 = toks('<think>', *F_BOX, '</think>', '<answer>', *T_BOX, '</answer>')      # reasoning box + ONE valid answer box
MULTI_IN_ANSWER = toks(*THINK, *F_BOX, ' ', *F_BOX, *ANSWER)                                    # 2 boxes INSIDE <answer>: format .5
MALFORMED = toks(*THINK, '\\boxed{', 'FL', 'AIR', *ANSWER)                                       # unclosed box
EXTRA_MARKUP = toks(*THINK, *F_BOX, ' ', '\\boxed{', 'x', *ANSWER)                               # parser: 1 balanced box (fmt 1) + stray \boxed{
T1_LABEL_BRACE_BPE = toks(*THINK, '\\boxed{', 'contrast', '-en', 'hanced', ' T', '1}', *ANSWER)  # last label byte shares a token with '}'
WRAPPED = toks(*THINK, '\\boxed{', '\\text{', 'FL', 'AIR', '}', '}', *ANSWER)
EMPTY = toks(*THINK, '\\boxed{', '}', *ANSWER)
NO_TAGS = toks(*F_BOX)                                                                           # format 0, canonical content
TRUNCATED = toks(*THINK, *F_BOX, eos=False)                                                      # no </answer>: format 0


def text_of(ids):
    return b''.join(DECODER.piece(t) for t in ids if t != EOS).decode()


# ================================================================ 1. import isolation
def test_imports_load_no_heavy_runtime():
    assert not [m for m in sys.modules if m.split('.')[0] in HEAVY]
    for module in (nr, ne, nb):
        tree = ast.parse(Path(module.__file__).read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = [a.name for n in top for a in n.names] + [n.module or '' for n in top if isinstance(n, ast.ImportFrom)]
        assert not [x for x in names if x.split('.')[0] in HEAVY]


# ================================================================ 2. exact budgets
def test_exact_totals_and_per_call_pins():
    assert nr.CHECKPOINTS == ('A53', 'B53') and nr.TF_PER_CHECKPOINT == 32 and nr.DEV_PER_CHECKPOINT == 4
    t = nr.TOTALS
    assert t['teacher_forced_scorer_forwards'] == 64 and t['greedy_generations'] == 8 and t['returned_tokens_max'] == 2048
    assert t['LM_max'] == 2080 and t['vision_max'] == 8 and t['sampler_max'] == 2056
    assert t['backward_calls'] == t['optimizer_updates'] == t['new_train_generation'] == t['recoveries'] == 0 and t['model_loads_max'] == 2
    assert nr.GREEDY_LIMITS == {'generations': 4, 'generated_tokens': 1024, 'LM': 1040, 'vision': 4, 'sampler': 1028}
    # Same per-greedy-call ceilings as the executed V2 DEV24 counter (source text pin; module not imported here)
    assert "PER_CALL = {'LM': 260, 'sampler': 257}" in (SRC / 'accuracy_pilot_v2_native.py').read_text()
    assert nr.RUN_LIMITS['per_checkpoint_window_seconds'] == 1200 and nr.RUN_LIMITS['total_window_seconds'] == 2400
    assert nr.RUN_LIMITS['wait_counts_against_deadlines'] is True and nr.RUN_LIMITS['automatic_retry'] is False
    assert nr.RUN_LIMITS['windows_reset'] is False and not hasattr(nr, 'WORST_CASE_SECONDS_PER_CHECKPOINT')
    assert nr.PLANNING_ESTIMATE_SECONDS_PER_CHECKPOINT == 90 + 32 * 5 + 4 * (256 * .25 + 3)
    assert nr.COST_MODEL['kind'] == 'PLANNING_ESTIMATE_FROM_HISTORICAL_RATES_NOT_CALIBRATED_NOT_A_WALLTIME_GUARANTEE'


def test_tf_positions_exact_and_length_counterexample():
    jobs = [{'task': SEQ, 'prompt_length': 230, 'tokens': 1}, {'task': SEQ, 'prompt_length': 230, 'tokens': 17},
            {'task': DAL, 'prompt_length': 405, 'tokens': 256}]
    exact = (230 + 1 - 1) + (230 + 17 - 1) + (405 + 256 - 1)
    assert nr.tf_job_positions(jobs, LENGTHS) == exact
    assert exact != sum(j['prompt_length'] + j['tokens'] for j in jobs)        # P+n estimate is NOT the charged count
    with pytest.raises(ValueError):                                          # recorded P differs from the frozen ledger
        nr.tf_job_positions([{'task': SEQ, 'prompt_length': 213, 'tokens': 5}], LENGTHS)
    with pytest.raises(ValueError):
        nr.tf_job_positions([{'task': DAL, 'prompt_length': 405, 'tokens': 257}], LENGTHS)
    counters = shared.Counters(nr.tf_counter_limits(1, 230 + 17 - 1))
    counters.charge_many({'detached_old_forwards': 1, 'scorer_vision_embedding_calls': 1, 'scorer_language_model_calls': 1,
                          'teacherforced_input_positions': 230 + 17 - 1})
    with pytest.raises(ValueError):
        counters.charge('teacherforced_input_positions', 1)
    with pytest.raises(ValueError):
        counters.charge('backward_calls', 1)


def test_finite_counter_fresh_only_and_refuses(tmp_path):
    c = nr.FiniteCounter({'generations': 1, 'generated_tokens': 3}, tmp_path / 'c.json')
    c.charge('generated_tokens', 3)
    with pytest.raises(ValueError):
        c.charge('generated_tokens', 1)
    with pytest.raises(ValueError):
        c.charge('generations', -1)
    with pytest.raises(ValueError):
        nr.FiniteCounter({'generations': 1, 'generated_tokens': 3}, tmp_path / 'c.json')   # never resumed


# ================================================================ 3. answer-position arithmetic on saved ids
def span(ids, task=SEQ, text=None):
    return nr.answer_span(ids, DECODER, text_of(ids) if text is None else text, task, eos_last=ids[-1] == EOS)


def test_answer_span_label_only_eligibility_strict_canonical():
    f = span(FLAIR_OK)
    assert f['status'] == 'STRICT_CANONICAL' and f['label'] == 'FLAIR' and f['label_metric_eligible'] is True
    assert f['measurement_kind'] == 'LABEL_ONLY' and f['label_positions'] == f['overlap_positions'] == [5, 6] and f['decision_index'] == 5
    assert f['ineligibility_reason'] is None and not f['boundary_tokens_include_non_answer_bytes'] and f['parser_format'] == 1.
    t = span(T1_OK)
    assert t['label_metric_eligible'] and t['label_positions'] == [5, 6, 7, 8, 9] and len(T1_OK) - 1 not in t['overlap_positions']
    d = span(SAME, DAL)
    assert d['label_metric_eligible'] and d['label'] == 'same' and d['label_positions'] == [5]
    w = span(WRAPPED)                                                    # \text{} wrapper: label bytes FLAIR are token-aligned
    assert w['label_metric_eligible'] and w['label_positions'] == [6, 7]
    r = span(REASONING_BOX_THEN_T1)                                      # a reasoning box does NOT invalidate the single answer box
    assert r['status'] == 'STRICT_CANONICAL' and r['label'] == 'contrast-enhanced T1' and r['label_metric_eligible']
    assert r['label_positions'] == [8, 9, 10, 11, 12] and not set(r['overlap_positions']) & {1, 2, 3, 4}   # reasoning FL/AIR = 2,3
    for s_ in (f, t, d, w, r):
        assert 'content' not in s_ and 'text' not in s_                   # no raw answer text retained


def test_answer_span_ineligible_spans_keep_only_marked_overlap_diagnostics():
    m = span(MERGED)                                                     # '{FL' / 'AIR}': label bytes share tokens with braces
    assert m['status'] == 'STRICT_CANONICAL' and m['label_metric_eligible'] is False and m['ineligibility_reason'] == 'PARTIAL_BPE_BOUNDARY'
    assert m['label_positions'] == [] and m['overlap_positions'] == [5, 6] and m['measurement_kind'] == 'WHOLE_OVERLAPPING_TOKEN_DIAGNOSTIC'
    assert m['boundary_tokens_include_non_answer_bytes'] and m['decision_index'] is None
    b = span(T1_LABEL_BRACE_BPE)                                         # mixed label/brace final token '1}'
    assert not b['label_metric_eligible'] and b['ineligibility_reason'] == 'PARTIAL_BPE_BOUNDARY' and b['overlap_positions'] == [5, 6, 7, 8, 9]
    mu = span(MULTI_IN_ANSWER)                                           # >1 box inside <answer>: parser format .5, ambiguous
    assert mu['status'] == 'MULTIPLE_ANSWER_BOXES' and not mu['label_metric_eligible'] and mu['parser_format'] == .5
    assert mu['overlap_positions'] == [10, 11] and mu['label_positions'] == [] and mu['measurement_kind'] == 'WHOLE_OVERLAPPING_TOKEN_DIAGNOSTIC'
    mal = span(MALFORMED)
    assert mal['status'] == 'MALFORMED_BOX' and not mal['label_metric_eligible'] and mal['overlap_positions'] == [] and mal['measurement_kind'] == 'NONE'
    ex = span(EXTRA_MARKUP)                                              # historical parser grades it strict-valid; measurement refuses it
    assert ex['status'] == 'STRICT_CANONICAL' and ex['parser_format'] == 1. and not ex['label_metric_eligible']
    assert ex['ineligibility_reason'] == 'EXTRA_BOX_MARKUP_IN_ANSWER' and ex['overlap_positions'] == [5, 6]
    n = span(NO_BOX)
    assert n['status'] == 'NO_ANSWER_BOX' and n['overlap_positions'] == [] and not n['label_metric_eligible']
    e = span(EMPTY)
    assert e['status'] == 'EMPTY_ANSWER_BOX' and e['overlap_positions'] == []
    for ids in (NO_TAGS, TRUNCATED):                                     # missing/invalid tags: format 0, content kept only as overlap
        x = span(ids)
        assert x['status'] == 'FORMAT_INVALID' and x['parser_format'] == 0. and not x['label_metric_eligible'] and x['content_canonical'] is True
        assert x['measurement_kind'] == 'WHOLE_OVERLAPPING_TOKEN_DIAGNOSTIC' and x['label_positions'] == [] and x['label'] is None
    assert span(NO_TAGS)['overlap_positions'] == [1, 2] and span(TRUNCATED)['overlap_positions'] == [5, 6]
    wrong = span(SAME, SEQ)                                              # strict-valid format, non-label content
    assert wrong['status'] == 'STRICT_NONCANONICAL' and wrong['overlap_positions'] == [5] and not wrong['label_metric_eligible']
    assert wrong['content_canonical'] is False and wrong['parser_answer_canonical'] is False
    for text in (text_of(T1_OK), 'no tags \\boxed{FLAIR}'):              # decoded saved ids differ from the saved text
        mm = span(FLAIR_OK, text=text)
        assert mm['status'] == 'DECODE_MISMATCH' and mm['overlap_positions'] == [] and not mm['label_metric_eligible']


def test_response_measures_eos_answer_and_ratio():
    old = [-1.0, -2.0, -0.5, -0.25, -0.1]
    new = [-0.5, -2.5, -0.5, -0.05, -0.2]
    m = nr.response_measures(new, old, eos_last=True, overlap_positions=[2, 3], label_positions=[2, 3])
    assert m['delta_first'] == pytest.approx(0.5) and m['delta_sum'] == pytest.approx(0.1) and m['delta_mean'] == pytest.approx(0.02)
    assert m['delta_eos'] == pytest.approx(-0.1) and m['delta_mean_excluding_eos'] == pytest.approx(0.2 / 4)
    assert m['label_metric_eligible'] is True and m['label_tokens'] == 2 and m['delta_label_sum'] == pytest.approx(0.2)
    assert m['delta_label_mean'] == pytest.approx(0.1) and m['delta_label_first'] == pytest.approx(0.0) and m['raw_ratio'] == pytest.approx(math.exp(0.02))
    assert m['overlap_tokens'] == 2 and m['delta_overlap_sum'] == pytest.approx(0.2)
    diag_only = nr.response_measures(new, old, eos_last=True, overlap_positions=[2, 3], label_positions=None)   # ineligible span
    assert diag_only['label_metric_eligible'] is False and diag_only['delta_label_mean'] is None and diag_only['label_tokens'] == 0
    assert diag_only['delta_overlap_mean'] == pytest.approx(0.1)
    assert not any(k.startswith('delta_answer') for k in m)                       # no unlabeled generic answer metric remains
    none = nr.response_measures(new, old, eos_last=False, overlap_positions=[])
    assert none['delta_eos'] is None and none['overlap_tokens'] == 0 and none['delta_overlap_mean'] is None and none['delta_label_mean'] is None
    with pytest.raises(ValueError):                                                # label positions must be exactly the overlap positions
        nr.response_measures(new, old, eos_last=True, overlap_positions=[2, 3], label_positions=[2])
    with pytest.raises(ValueError):
        nr.response_measures(new, old, eos_last=True, overlap_positions=[], label_positions=[])
    capped = nr.response_measures([0.] * 2, [-30.] * 2, eos_last=False, overlap_positions=[])
    assert capped['raw_ratio'] == pytest.approx(math.exp(10.)) and capped['log_ratio_capped'] is True
    for bad in ([-1., float('nan')], [-1., 0.5], [-1.]):
        with pytest.raises(ValueError):
            nr.response_measures(bad, [-1., -1.], eos_last=False, overlap_positions=[])
    with pytest.raises(ValueError):
        nr.response_measures([-1., -1.], [-1., -1.], eos_last=False, overlap_positions=[2])


def test_post_update_scores_are_not_rejected_by_old_current_parity(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('training parity gate must not be applied to post-update scores')
    monkeypatch.setattr(shared, 'selected_logp_parity', forbidden)
    m = nr.response_measures([-0.1] * 4, [-0.9] * 4, eos_last=True, overlap_positions=[1], label_positions=[1])
    assert m['delta_mean'] == pytest.approx(0.8)                                     # >> 1e-5 / .002 training gates


def test_clipped_gspo_surrogate_branches():
    lo, hi = 1 - 3e-4, 1 + 4e-4
    assert nr.gspo_surrogate(1.0, 0.7) == pytest.approx(-0.7)
    assert nr.gspo_surrogate(1.01, 0.5) == pytest.approx(-0.5 * hi)                 # A>0 above the clip: clipped (flat)
    assert nr.gspo_surrogate(1.01, -0.5) == pytest.approx(0.5 * 1.01)              # A<0 above: unclipped (pessimistic)
    assert nr.gspo_surrogate(0.99, 0.5) == pytest.approx(-0.5 * 0.99)              # A>0 below: unclipped
    assert nr.gspo_surrogate(0.99, -0.5) == pytest.approx(0.5 * lo)                # A<0 below: clipped
    assert nr.gspo_surrogate(1.2, 0.) == 0.
    with pytest.raises(ValueError):
        nr.gspo_surrogate(float('inf'), 1.)


# ================================================================ synthetic scope / checkpoint fixtures
def _job(i, unit, task, key, rid, tokens, *, raw, fmt, plans, answer):
    P = LENGTHS[task]
    old = [-0.5] * len(tokens)
    return {'index': i, 'unit': unit, 'task': task, 'context_key': key, 'response_id': rid, 'prompt_length': P, 'tokens': len(tokens),
            'tf_positions': P + len(tokens) - 1, 'token_ids': tokens, 'token_sha256': shared.identity(tuple(tokens)), 'old_logps_PRIVATE': old,
            'old_logps_sha256': shared.identity(old), 'raw_accuracy': raw, 'format': fmt, 'reward_total': .8 * raw + .2 * fmt, 'eos_last': True,
            'image_class': {'F': 'FLAIR_IMAGE', 'T': 'T1_IMAGE'}.get(key, 'DALLAS'), 'answer': answer, 'plans': plans,
            'recorded_context_sha256': shared.identity({'unit': unit, 'key': key})}


def synthetic_jobs():
    jobs, i = [], 0
    for unit in range(4):
        task, keys = (SEQ, ('F', 'T')) if unit % 2 == 0 else (DAL, ('D1', 'D2'))
        for key in keys:
            for r in range(4):
                rid = f'{key}{r}' if task == SEQ else f'{key}_{r}'
                tokens = (FLAIR_OK if key in ('F', 'D1') else T1_OK) if task == SEQ else SAME
                raw = int(r % 2 == 0)
                adv = .75 if raw else -.75
                plans = {'A': {'advantage': adv, 'weight': 1 / 8, 'admitted': True, 'group': 0},
                         'B': {'advantage': adv if unit == 0 else 0., 'weight': 1 / 8 if unit == 0 else 0., 'admitted': unit == 0, 'group': 0}}
                answer = nr.answer_span(tokens, DECODER, text_of(tokens), task, eos_last=True)
                jobs.append(_job(i, unit, task, key, rid, tokens, raw=raw, fmt=1, plans=plans, answer=answer)); i += 1
    return jobs


def synthetic_dev():
    out = []
    for k, (role, modality) in enumerate((('REGRESSED', 'FLAIR'), ('PARTNER', 'contrast-enhanced T1'), ('PARTNER', 'FLAIR'), ('REGRESSED', 'contrast-enhanced T1'))):
        row = {'id': f'dev{k}', 'ability': SEQ, 'images': [f'/img{k}.png'], 'reward_model': {'ground_truth': modality}, 'extra_info': {'question': 'q'}}
        ref = {'text': text_of(FLAIR_OK), 'token_ids': FLAIR_OK, 'raw_selected_logps': [-0.1] * len(FLAIR_OK)}
        out.append({'index': k, 'dev_case_index_PRIVATE': [3, 4, 10, 11][k], 'pair': k // 2, 'role': role, 'truth': modality, 'row_PRIVATE': row,
                    'image_sha256': shared.identity({'image': k}), 'reference_PRIVATE': {'Parent': ref, 'A54': ref, 'B54': ref}})
    return out


def make_checkpoint(directory, label):
    ck = Path(directory) / label
    (ck / 'optimizer').mkdir(parents=True)
    (ck / 'adapter.safetensors').write_bytes(b'adapter-' + label.encode())
    (ck / 'optimizer' / 'state.safetensors').write_bytes(b'state-' + label.encode())
    run_binding = {'scope_sha256': 'a' * 64, 'parent_receipt_sha256': 'b' * 64, 'stage': label[0]}
    (ck / 'optimizer' / 'manifest.json').write_text(json.dumps({'bindings': {'run': run_binding, 'adapter_sha256': shared.file_sha256(ck / 'adapter.safetensors')}}))
    files = {str(p): shared.file_sha256(p) for p in sorted(ck.rglob('*')) if p.is_file()}
    receipt = {'policy_sha_after': 'policy-' + label, 'adam_digest_after': 'adam-' + label, 'rng_digest_after': 'rng-' + label, 'adam_step_after': 53}
    return {'checkpoint_dir': str(ck), 'receipt_PRIVATE': receipt, 'run_binding': run_binding, 'payload_sha256': files}


def make_scope(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bound = tmp_path / 'bound.txt'
    bound.write_text('source')
    checkpoints = {label: make_checkpoint(tmp_path / 'ck', label) for label in nr.CHECKPOINTS}
    units = [{'index': u, 'kind': 'SEQ_PAIR' if u % 2 == 0 else 'DALLAS_BLOCK', 'contexts': []} for u in range(4)]
    return nr.assemble_scope(sources={str(bound): shared.file_sha256(bound)}, files={}, run_output=str(tmp_path / 'run_PRIVATE'),
                             checkpoints=checkpoints, units=units, recorded_contexts={str(u): {} for u in range(4)}, tf_jobs=synthetic_jobs(),
                             dev_jobs=synthetic_dev(), selection={'rule': nr.DEV_SELECTION_RULE, 'regressed_PRIVATE': [3, 11], 'selected_PRIVATE': [3, 4, 10, 11]},
                             lengths=dict(LENGTHS), runtime={'runtime_spec': str(bound), 'global_lock': str(bound)}, pins={})


def fresh_human(scope, **over):
    now = datetime.now(timezone.utc)
    human = {'authorized': True, 'action': nr.ACTION, 'scope_sha256': nr.digest(scope), 'included_checkpoints': list(nr.CHECKPOINTS),
             'human_instruction_sha256': 'c' * 64, 'approved_by': 'researcher', 'issued_at_utc': (now - timedelta(seconds=5)).isoformat(),
             'expires_at_utc': (now + timedelta(hours=2)).isoformat()}
    human.update(over)
    return human


def review_for(scope, **over):
    review = {'verdict': 'PASS', 'scope_sha256': nr.digest(scope), 'source_sha256': scope['source_sha256'], 'input_sha256': scope['input_sha256'],
              'reviewer_kind': 'isolated_codex', 'reviewer_id': 'codex-isolated-1'}
    review.update(over)
    return review


def issue(tmp_path, scope, label):
    token = ne.authorize(scope, fresh_human(scope), review_for(scope), consumed=shared.consume_once(tmp_path / ('ledger_' + label)))
    run = Path(scope['run_output'])
    run.mkdir(parents=True, exist_ok=True)
    clock = run / f'{label}_clock_PRIVATE.json'
    if not clock.exists():
        shared.publish_once(clock, {'started_monotonic': token.start})
    return token, token.for_checkpoint(label)


class FakeGate:
    def __init__(self, scope, authorization):
        self.scope, self.authorization, self.calls, self.loads = scope, authorization, [], 0
    def __call__(self, name):
        self.authorization.check('resources_' + name); self.calls.append(name)
    def claim_model_load(self):
        self.loads += 1
        if self.loads > 1:
            raise ValueError('one load per checkpoint child')
    def peaks(self):
        return {'peak_process_rss_gib': 1.0, 'peak_mlx_reported_gib': None}


class TrainingCalled(AssertionError):
    pass


class FakeBackend:
    """Every training / TRAIN-generation surface raises if touched."""
    def __init__(self):
        self.model = object(); self.params = None; self.adam = None; self.rng = None; self.step = None; self.stage_binding = None; self.counters = None
    def _train(self, *a, **k):
        raise TrainingCalled('training or TRAIN-generation method called')
    gradient = arm_gradient = apply_update = generate_shared_rollout = score_old = save_checkpoint = generate = _train


class FakeOps:
    def __init__(self, *, mutate_weights=False, keep_rng_consumed=False, release_ok=True, logp=-0.4, wrong_identity=False, bad_logp=None, tokens=12):
        self.mutate_weights, self.keep_rng_consumed, self.release_ok, self.logp = mutate_weights, keep_rng_consumed, release_ok, logp
        self.wrong_identity, self.bad_logp, self.tokens = wrong_identity, bad_logp, tokens
        self.scored, self.generated, self.loaded_bindings = [], [], []
    def construct(self, scope, job, resident, gate, directory):
        b = FakeBackend(); resident[0] = b; return b
    def load_and_verify(self, backend, checkpoint_dir, receipt, run_binding):
        manifest = json.loads((Path(checkpoint_dir) / 'optimizer' / 'manifest.json').read_text())
        backend.stage_binding = run_binding
        if manifest['bindings']['run'] != backend.stage_binding:
            raise ValueError('Checkpoint binding differs')
        self.loaded_bindings.append(run_binding)
        backend.params, backend.adam, backend.rng, backend.step = receipt['policy_sha_after'], receipt['adam_digest_after'], receipt['rng_digest_after'], 53
        if self.wrong_identity:
            backend.rng = 'some-other-rng'
        identity = self.full_identity(backend)
        if (identity['policy'], identity['adam'], identity['rng'], identity['optimizer_step']) != (receipt['policy_sha_after'], receipt['adam_digest_after'], receipt['rng_digest_after'], 53):
            raise ValueError('Loaded checkpoint full policy/Adam/RNG identity differs from receipt')
        return identity
    def full_identity(self, b):
        return {'parameters': {'all': b.params}, 'optimizer': {'all': b.adam}, 'policy': b.params, 'adam': b.adam, 'rng': b.rng, 'optimizer_step': b.step}
    def rng_digest(self, b):
        return b.rng
    def readonly(self, backend, run, directory):
        params, adam, rng = backend.params, backend.adam, backend.rng
        try:
            return run()
        finally:
            if not self.keep_rng_consumed:
                backend.rng = rng
    def score(self, backend, unit, recorded, key, tokens, prompt_length):
        backend.counters.charge_many({'detached_old_forwards': 1, 'scorer_vision_embedding_calls': 1, 'scorer_language_model_calls': 1,
                                      'teacherforced_input_positions': prompt_length + len(tokens) - 1})
        self.scored.append((unit['index'], key, tuple(tokens)))
        if self.mutate_weights:
            backend.params = 'mutated'
        values = [self.logp] * len(tokens)
        if self.bad_logp is not None:
            values[-1] = self.bad_logp
        return values
    def generate(self, backend, row, image_sha, lengths, counter, before):
        n = self.tokens
        counter.charge('generations'); counter.charge('vision')
        for _ in range(n + 1):
            counter.charge('LM'); counter.charge('sampler')
        counter.charge('generated_tokens', n)
        backend.rng = backend.rng + '+'                                      # greedy may consume RNG; readonly restores it
        self.generated.append(row['id'])
        ids = (FLAIR_OK + [ID['x']] * n)[:n]
        return {'response': {'text': text_of(FLAIR_OK), 'token_ids': ids, 'raw_selected_logps': [-0.05] * n, 'finish_reason': 'stop'},
                'image_sha256': image_sha, 'prompt_length': lengths[SEQ], 'prompt_sha256': 'd' * 64, 'input_fingerprints_PRIVATE': {}}
    def release(self, backend, directory):
        if not self.release_ok:
            raise ValueError('Recovery allocator release failed')
        backend.model = None
        p = Path(directory) / 'operation_release_SAFE.json'
        shared.publish_once(p, {'model_is_none': True, 'active_MLX_bytes': 0, 'release_verified': True})
        return p
    def run_with_release(self, call, resident, directory):
        try:
            return call()
        finally:
            ok = resident[0] is not None and resident[0].model is None
            if ok:
                shared.publish_once(Path(directory) / 'release_SAFE.json', {'model_is_none': True, 'active_MLX_bytes': 0, 'release_verified': True})
            else:
                shared.publish_once(Path(directory) / 'release_failure_PRIVATE.json', {'error': 'release failed'})


def run_one(tmp_path, label, ops, scope=None):
    scope = scope or make_scope(tmp_path)
    token, phase = issue(tmp_path, scope, label)
    gate = FakeGate(phase.scope, phase)
    return scope, nb.run_checkpoint(label, scope=phase.scope, authorization=phase, resource_gate=gate, ops=ops), gate


# ================================================================ 4. native control flow with fakes
def test_fake_checkpoint_run_exact_accounting_and_no_training(tmp_path):
    ops = FakeOps()
    scope, result, gate = run_one(tmp_path, 'A53', ops)
    out = Path(scope['run_output']) / 'A53_PRIVATE'
    assert result['status'] == 'PASS' and result['tf_scored'] == 32 and result['dev_generated'] == 4
    expected = sum(j['tf_positions'] for j in scope['tf_jobs_PRIVATE'])
    assert result['tf_counters']['teacherforced_input_positions'] == expected == scope['budget']['tf_positions_per_checkpoint']
    assert result['tf_counters']['detached_old_forwards'] == 32 and result['tf_counters']['backward_calls'] == 0
    assert result['tf_counters']['optimizer_updates'] == 0 and result['tf_counters']['generated_response_tokens'] == 0
    g = result['greedy_counters']
    assert g['generations'] == 4 and g['vision'] == 4 and g['generated_tokens'] <= 1024 and g['LM'] <= 1040 and g['sampler'] <= 1028
    assert all(v >= 0 for v in {**result['tf_counters'], **g}.values())
    assert len(ops.scored) == 32 and len(ops.generated) == 4 and gate.loads == 1
    # ownership: every score routed with its OWN unit/context and saved token ids
    assert [(u, k, t) for u, k, t in ops.scored] == [(j['unit'], j['context_key'], tuple(j['token_ids'])) for j in scope['tf_jobs_PRIVATE']]
    assert ops.loaded_bindings == [scope['checkpoints_PRIVATE']['A53']['run_binding']]         # ORIGINAL run binding, not the new scope
    assert result['rng_after_readonly'] == result['loaded_identity_PRIVATE']['rng'] and result['rng_consumed_during_readonly_observed'] is True
    rows = sorted((out / 'tf_PRIVATE').glob('*_PRIVATE.json'))
    assert len(rows) == 32
    for p in rows:
        rec = json.loads(p.read_text())
        assert all(math.isfinite(v) and v <= 0 for v in rec['logps']) and len(rec['logps']) == rec['tokens']
    assert (out / 'native_complete_PRIVATE.json').is_file()


def test_nonfinite_or_positive_scorer_output_rejected(tmp_path):
    for bad in (float('nan'), 0.25):
        with pytest.raises(ValueError):
            run_one(tmp_path / str(bad), 'A53', FakeOps(bad_logp=bad))


def test_loaded_identity_must_match_original_receipt(tmp_path):
    ops = FakeOps(wrong_identity=True)
    with pytest.raises(ValueError):
        run_one(tmp_path, 'A53', ops)
    assert ops.scored == [] and ops.generated == []


def test_checkpoint_binding_is_the_original_manifest_binding(tmp_path):
    scope = make_scope(tmp_path)
    bad = json.loads(json.dumps(scope))
    bad['checkpoints_PRIVATE']['A53']['run_binding'] = {'scope_sha256': nr.digest(scope)}        # a NEW-scope binding is wrong
    with pytest.raises(ValueError):
        nr.check_checkpoint_entry(bad['checkpoints_PRIVATE']['A53'])
    nr.check_checkpoint_entry(scope['checkpoints_PRIVATE']['A53'])


def test_weight_mutation_detected_and_not_committed(tmp_path):
    ops = FakeOps(mutate_weights=True)
    scope = make_scope(tmp_path)
    with pytest.raises(ValueError):
        run_one(tmp_path, 'A53', ops, scope)
    assert not (Path(scope['run_output']) / 'A53_PRIVATE' / 'native_complete_PRIVATE.json').exists()


def test_rng_not_restored_is_detected(tmp_path):
    with pytest.raises(ValueError):
        run_one(tmp_path, 'A53', FakeOps(keep_rng_consumed=True))


def test_checkpoint_disk_change_detected(tmp_path):
    scope = make_scope(tmp_path)
    Path(scope['checkpoints_PRIVATE']['A53']['checkpoint_dir'], 'adapter.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError):
        run_one(tmp_path, 'A53', FakeOps(), scope)


def test_partial_operation_is_never_replayed(tmp_path):
    scope = make_scope(tmp_path)
    out = Path(scope['run_output']) / 'A53_PRIVATE'
    j = Journal(out / 'journal_PRIVATE', {'x': 1})
    def crash(opdir):
        raise RuntimeError('crash after claim')
    with pytest.raises(RuntimeError):
        j.run('tf_00', crash, reservation={'teacherforced_input_positions': 10})
    with pytest.raises(Indeterminate):
        j.assert_recoverable()
    j.close()
    ops = FakeOps()
    with pytest.raises(ValueError):
        run_one(tmp_path, 'A53', ops, scope)                         # existing output: one-time claim, no resume/replay
    assert ops.scored == []


def test_release_failure_prevents_commit(tmp_path):
    scope = make_scope(tmp_path)
    with pytest.raises(ValueError):
        run_one(tmp_path, 'A53', FakeOps(release_ok=False), scope)
    out = Path(scope['run_output']) / 'A53_PRIVATE'
    assert not (out / 'native_complete_PRIVATE.json').exists()
    assert list(out.rglob('release_failure_PRIVATE.json'))


def test_backend_source_calls_no_training_or_train_generation_api():
    tree = ast.parse(Path(nb.__file__).read_text())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    forbidden = {'gradient', 'arm_gradient', 'apply_update', 'value_and_grad', 'update', 'generate_shared_rollout', 'score_old', 'save_checkpoint',
                 'run_segment', 'make_common', 'grad', 'backward'}
    assert not (attrs & forbidden) and not (names & forbidden)


# ================================================================ 5. DEV selection (frozen rule, existing outputs only)
LABELS = ('FLAIR', 'contrast-enhanced T1')


def dev_fixture():
    records = []
    for p in range(12):
        for label in LABELS:
            records.append({'id': f'r{len(records)}', 'row_PRIVATE': {'id': f'r{len(records)}', 'extra_info': {'patient_key': f'pt{p // 3}', 'level': str(p % 3)},
                                                                       'reward_model': {'ground_truth': label}}})
    return records


def rows_from(preds, records):
    def text(i, label):
        return text_of(FLAIR_OK) if label == 'FLAIR' else text_of(T1_OK)
    return [{'text': text(i, LABELS[preds[i]])} for i in range(len(records))]


def test_dev_selection_regressions_plus_same_pair_partners():
    records = dev_fixture()
    truth = [LABELS.index(r['row_PRIVATE']['reward_model']['ground_truth']) for r in records]
    parent = list(truth); parent[5] = 0                       # one parent error (wrong kept)
    a = list(parent); a[2] = 1; a[19] = 0; a[5] = 1           # regressions at 2 (FLAIR) and 19 (T1); 5 improves (must NOT be selected)
    b = list(a)
    rows = {'Parent': rows_from(parent, records), 'A': rows_from(a, records), 'B': rows_from(b, records)}
    sel = nr.select_dev_cases(records, rows)
    assert sel['regressed_PRIVATE'] == [2, 19] and sel['selected_PRIVATE'] == [2, 3, 18, 19]
    assert sel['roles_PRIVATE'] == {'2': 'REGRESSED', '3': 'PARTNER', '18': 'PARTNER', '19': 'REGRESSED'}
    assert 5 not in sel['selected_PRIVATE'] and 4 not in sel['selected_PRIVATE']
    assert sel['rule'] == nr.DEV_SELECTION_RULE and sel['uses_new_outputs'] is False
    # deterministic, union over A/B
    b2 = list(parent); b2[2] = 1
    sel2 = nr.select_dev_cases(records, {'Parent': rows['Parent'], 'A': rows_from(parent[:19] + [0] + parent[20:], records), 'B': rows_from(b2, records)})
    assert sel2['selected_PRIVATE'] == [2, 3, 18, 19]


def test_dev_selection_fails_closed_outside_predeclared_design():
    records = dev_fixture()
    truth = [LABELS.index(r['row_PRIVATE']['reward_model']['ground_truth']) for r in records]
    three = list(truth); three[2] = 1; three[19] = 0; three[7] = 0
    with pytest.raises(ValueError):
        nr.select_dev_cases(records, {'Parent': rows_from(truth, records), 'A': rows_from(three, records), 'B': rows_from(three, records)})
    partners = list(truth); partners[2] = 1; partners[3] = 0                 # regressions are each other's partners -> only 2 cases
    with pytest.raises(ValueError):
        nr.select_dev_cases(records, {'Parent': rows_from(truth, records), 'A': rows_from(partners, records), 'B': rows_from(partners, records)})


# ================================================================ 6. authority
def test_authorization_denials(tmp_path):
    scope = make_scope(tmp_path)
    ok_h, ok_r = fresh_human(scope), review_for(scope)
    cases = [
        (None, ok_r), ({**ok_h, 'authorized': False}, ok_r), ({**ok_h, 'scope_sha256': 'e' * 64}, ok_r), ({**ok_h, 'included_checkpoints': ['A53']}, ok_r),
        ({**ok_h, 'issued_at_utc': (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()}, ok_r),
        ({**ok_h, 'expires_at_utc': (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}, ok_r),
        (ok_h, None), (ok_h, {**ok_r, 'reviewer_kind': 'claude_code'}), (ok_h, {**ok_r, 'reviewer_id': 'claude-subagent'}),
        (ok_h, {**ok_r, 'verdict': 'FAIL'}), (ok_h, {**ok_r, 'scope_sha256': 'e' * 64}), (ok_h, {**ok_r, 'input_sha256': 'e' * 64})]
    for human, review in cases:
        with pytest.raises((ValueError, TypeError, AttributeError, KeyError)):
            ne.authorize(scope, human, review, consumed=shared.consume_once(tmp_path / 'deny'))
    consumed = shared.consume_once(tmp_path / 'once')
    ne.authorize(scope, ok_h, ok_r, consumed=consumed)
    with pytest.raises(ValueError):
        ne.authorize(scope, ok_h, ok_r, consumed=consumed)            # consumed instruction cannot authorize twice
    tampered = json.loads(json.dumps(scope)); tampered['budget']['tf_positions_per_checkpoint'] += 1
    with pytest.raises(ValueError):
        ne.authorize(tampered, fresh_human(tampered), review_for(tampered), consumed=shared.consume_once(tmp_path / 't'))
    with pytest.raises(ValueError):
        ne.Authorization(object(), scope, datetime.now(timezone.utc) + timedelta(hours=1))   # only exact admission issues tokens


def test_bound_source_change_revokes_authority(tmp_path):
    scope = make_scope(tmp_path)
    token, phase = issue(tmp_path, scope, 'A53')
    Path(next(iter(scope['source_sha256']))).write_text('changed')
    with pytest.raises(ValueError):
        phase.check('model_load')


def test_checkpoint_window_counts_waits(tmp_path, monkeypatch):
    scope = make_scope(tmp_path)
    token, phase = issue(tmp_path, scope, 'A53')
    import time as _time
    real = _time.monotonic
    monkeypatch.setattr(ne.time, 'monotonic', lambda: real() + 1201)
    with pytest.raises(ValueError):
        phase.check('resources_x')


# ================================================================ 7. resources / serial controller
def test_emergency_resource_protections_unchanged(tmp_path):
    scope = make_scope(tmp_path)
    assert scope['resource_policy'] == resources.POLICY and nr.RESOURCE_POLICY == resources.POLICY
    from types import SimpleNamespace as S
    base = dict(available_gib=20., pressure_free_percent=60., swap_used_gib=1., process_rss_gib=4., battery_percent=80., thermal_state='NOMINAL',
                other_heavy_processes=0, power_source='BATTERY', mlx_active_gib=5.)
    assert resources.Controller().decide(S(**base), 1, 100., 0.)['action'] == 'CONTINUE'
    assert resources.Controller().decide(S(**base), 2, 100., 0.)['action'] == 'CONTINUE'          # YELLOW continues
    assert resources.Controller().decide(S(**base), 4, 100., 0.)['action'] == 'PAUSE_RELEASE'     # RED releases own child
    for k, v, disk in (('process_rss_gib', 14.5, 100.), ('mlx_active_gib', 12.5, 100.), ('available_gib', 1.5, 100.), ('battery_percent', 15., 100.),
                       ('pressure_free_percent', 10., 100.), ('available_gib', 20., 20.)):
        assert resources.Controller().decide(S(**{**base, k: v}), 1, disk, 0.)['action'] == 'STOP'
    c = resources.Controller()
    c.decide(S(**{**base, 'available_gib': 3.}), 1, 100., 0.)
    assert c.decide(S(**{**base, 'available_gib': 3.}), 1, 100., 31.)['action'] == 'STOP'        # sustained <4GiB for 30 s
    assert ne.MODEL_LOADS_MAX == 2


def _write_child_result(scope, label, *, release_ok=True):
    out = Path(scope['run_output']) / f'{label}_PRIVATE'
    (out / 'resident_PRIVATE').mkdir(parents=True)
    shared.publish_once(out / 'native_complete_PRIVATE.json', {'status': 'PASS', 'release_verified': release_ok, 'checkpoint': label})
    name = 'release_SAFE.json' if release_ok else 'release_failure_PRIVATE.json'
    shared.publish_once(out / 'resident_PRIVATE' / name, {'release_verified': release_ok})


def _witness(code=0):
    return {'native_exit_code': code, 'child_reaped': True, 'SIGTERM_sent': False, 'SIGKILL_after_grace': False, 'monitor_log_errors': [],
            'post_child_global_lock': {'no_open_holders': True}}


def test_serial_controller_runs_A_then_B_and_analyzes(tmp_path):
    scope = make_scope(tmp_path)
    Path(scope['run_output']).mkdir(parents=True)
    launched, analyzed = [], []
    def launch(label):
        launched.append(label); _write_child_result(scope, label); return _witness(0)
    status = ne.run_serial(scope, launch=launch, observe=lambda: {'action': 'CONTINUE', 'reasons': []}, analyze=lambda: analyzed.append(1) or {'ok': True},
                           sleep=lambda s: None)
    assert status['status'] == 'COMPLETE' and launched == ['A53', 'B53'] and analyzed == [1]


def test_A_release_failure_blocks_B(tmp_path):
    scope = make_scope(tmp_path)
    Path(scope['run_output']).mkdir(parents=True)
    launched = []
    def launch(label):
        launched.append(label); _write_child_result(scope, label, release_ok=False); return _witness(0)
    status = ne.run_serial(scope, launch=launch, observe=lambda: {'action': 'CONTINUE', 'reasons': []}, analyze=lambda: 1 / 0, sleep=lambda s: None)
    assert launched == ['A53'] and status['status'] == 'STOPPED_NO_RETRY' and status['blocked'] == ['B53']


def test_A_nonzero_exit_or_forced_stop_blocks_B_and_never_retries(tmp_path):
    for w in (_witness(75), {**_witness(0), 'SIGTERM_sent': True}, {**_witness(0), 'post_child_global_lock': {'no_open_holders': False}}):
        scope = make_scope(tmp_path / str(id(w)))
        Path(scope['run_output']).mkdir(parents=True)
        launched = []
        def launch(label, w=w, scope=scope):
            launched.append(label); _write_child_result(scope, label); return w
        status = ne.run_serial(scope, launch=launch, observe=lambda: {'action': 'CONTINUE', 'reasons': []}, analyze=lambda: 1 / 0, sleep=lambda s: None)
        assert launched == ['A53'] and status['status'] == 'STOPPED_NO_RETRY'
        with pytest.raises(ValueError):                                # second controller launch on the same output is refused
            ne.run_serial(scope, launch=launch, observe=lambda: {'action': 'CONTINUE', 'reasons': []}, analyze=lambda: 1 / 0, sleep=lambda s: None)
        assert launched == ['A53']


def test_nonresident_wait_is_bounded_and_stop_is_fatal(tmp_path):
    scope = make_scope(tmp_path)
    Path(scope['run_output']).mkdir(parents=True)
    t = [0.]
    def clock():
        t[0] += 10.; return t[0]
    with pytest.raises(ValueError):
        ne.run_serial(scope, launch=lambda label: 1 / 0, observe=lambda: {'action': 'WAIT', 'reasons': ['other_heavy_process']}, analyze=lambda: None,
                      sleep=lambda s: None, clock=clock)
    scope2 = make_scope(tmp_path / 'b')
    Path(scope2['run_output']).mkdir(parents=True)
    with pytest.raises(resources.ResourceFatal):
        ne.run_serial(scope2, launch=lambda label: 1 / 0, observe=lambda: {'action': 'STOP', 'reasons': ['RSS_above14GiB']}, analyze=lambda: None,
                      sleep=lambda s: None)


# ================================================================ 8. CPU analysis (descriptive 2x2, strata, DEV timing)
def test_train_analysis_2x2_surrogate_and_strata():
    jobs = synthetic_jobs()
    new = {'A53': {j['index']: [v + (0.01 if j['raw_accuracy'] else -0.01) for v in j['old_logps_PRIVATE']] for j in jobs},
           'B53': {j['index']: list(j['old_logps_PRIVATE']) for j in jobs}}
    a = nr.analyze_train(jobs, new)
    for ck in nr.CHECKPOINTS:
        for plan in ('A', 'B'):
            cell = a['surrogate_2x2'][ck][plan]
            assert set(cell) >= {'objective_old', 'objective_new', 'delta', 'delta_unclipped', 'admitted', 'outside_clip'}
            assert cell['label'].startswith('DESCRIPTIVE')
    assert a['surrogate_2x2']['B53']['A']['delta'] == pytest.approx(0.)            # unchanged scores -> unchanged objective
    assert a['surrogate_2x2']['A53']['A']['delta'] < 0                               # correct up / wrong down lowers the surrogate
    assert a['surrogate_2x2']['A53']['A']['admitted'] == 32 and a['surrogate_2x2']['A53']['B']['admitted'] == 8
    strata = a['strata']['A53']
    assert strata['raw']['CORRECT']['n'] == 16 and strata['raw']['CORRECT']['delta_mean']['mean'] == pytest.approx(0.01)
    assert strata['raw']['RAW_ZERO']['delta_mean']['mean'] == pytest.approx(-0.01) and 'WRONG' not in strata['raw']
    assert set(strata) >= {'task', 'image', 'format', 'raw', 'raw_outcome', 'advantage_sign_plan_A', 'advantage_sign_plan_B', 'answer_status'}
    assert strata['answer_status']['STRICT_CANONICAL']['delta_label_mean']['n'] == 32
    assert strata['raw_outcome']['RAW0_CANONICAL_WRONG_CLASS']['n'] == 16
    with pytest.raises(ValueError):
        nr.analyze_train(jobs, {'A53': new['A53']})                                  # both checkpoints required, all 32 each


def test_dev_analysis_locates_regression_timing_and_pairs():
    dev = synthetic_dev()            # roles: 0 REGRESSED FLAIR, 1 PARTNER T1, 2 PARTNER FLAIR, 3 REGRESSED T1 ; pairs (0,1),(2,3)
    def gen(label):
        ids = FLAIR_OK if label == 'FLAIR' else T1_OK
        return {'text': text_of(ids), 'token_ids': ids, 'raw_selected_logps': [-0.1] * len(ids)}
    for d in dev:
        d['reference_PRIVATE'] = {'Parent': gen(d['truth']), 'A54': gen('FLAIR' if d['role'] == 'PARTNER' else ('contrast-enhanced T1' if d['truth'] == 'FLAIR' else 'FLAIR')),
                                  'B54': None}
        d['reference_PRIVATE']['B54'] = d['reference_PRIVATE']['A54']
    new = {'A53': [gen('contrast-enhanced T1'), gen('contrast-enhanced T1'), gen('FLAIR'), gen('contrast-enhanced T1')],   # case0 wrong at 53, case3 still right
           'B53': [gen('FLAIR'), gen('contrast-enhanced T1'), gen('FLAIR'), gen('contrast-enhanced T1')]}
    out = nr.analyze_dev(dev, new, decoder=DECODER)
    assert out['A53']['regression_timing']['REGRESSION_PRESENT_AT_53'] == 1 and out['A53']['regression_timing']['REGRESSION_APPEARS_AT_54'] == 1
    assert out['B53']['regression_timing']['REGRESSION_APPEARS_AT_54'] == 2
    assert out['A53']['correct'] == 3 and out['B53']['correct'] == 4 and out['A53']['invalid'] == 0
    assert out['A53_vs_B53']['identical_predictions'] == 3
    assert out['B53']['pair_categories'] == {'DISCRIMINATES_CORRECTLY': 2}
    assert out['scope_label'].startswith('DIAGNOSTIC_ONLY')


def test_end_to_end_fake_A_then_B_then_cpu_analysis(tmp_path):
    scope = make_scope(tmp_path)
    for label in nr.CHECKPOINTS:
        run_one(tmp_path, label, FakeOps(logp=-0.4 if label == 'A53' else -0.6), scope)
    result = nr.run_analysis(scope, decoder=DECODER)
    pub = result['public']
    assert result['model_calls_in_analysis'] == 0 and pub['boundaries'] == nr.TOTALS
    assert pub['train']['surrogate_2x2']['A53']['A']['admitted'] == 32 and pub['dev']['A53']['images'] == 4
    assert (Path(scope['run_output']) / 'analysis_PRIVATE.json').is_file() and (Path(scope['run_output']) / 'AGGREGATE_PUBLIC_CANDIDATE.json').is_file()
    with pytest.raises(FileExistsError):
        nr.run_analysis(scope, decoder=DECODER)                          # outputs are write-once


def mixed_jobs():
    """MRI strict-correct, canonical-wrong, format-invalid (raw 0 by the gate), ambiguous/boundary spans; Dallas strict. 12 + 20 = 32."""
    cases = [  # (task, key, tokens, raw, fmt)
        (SEQ, 'F', FLAIR_OK, 1, 1), (SEQ, 'F', FLAIR_OK, 1, 1), (SEQ, 'T', FLAIR_OK, 0, 1), (SEQ, 'T', FLAIR_OK, 0, 1),
        (SEQ, 'F', NO_TAGS, 0, 0), (SEQ, 'T', TRUNCATED, 0, 0), (SEQ, 'F', MERGED, 1, 1), (SEQ, 'F', MULTI_IN_ANSWER, 1, .5),
        (SEQ, 'T', T1_LABEL_BRACE_BPE, 1, 1), (SEQ, 'F', SAME, 0, 1), (SEQ, 'T', NO_BOX, 0, .5), (SEQ, 'F', REASONING_BOX_THEN_T1, 0, 1)]
    cases += [(DAL, 'D1', SAME, r % 2, 1) for r in range(20)]
    jobs = []
    for i, (task, key, tokens, raw, fmt) in enumerate(cases):
        adv = .5 if raw else -.5
        plans = {'A': {'advantage': adv, 'weight': 1 / 8, 'admitted': True, 'group': 0},
                 'B': {'advantage': adv if task == SEQ else 0., 'weight': 1 / 8 if task == SEQ else 0., 'admitted': task == SEQ, 'group': 0}}
        answer = nr.answer_span(tokens, DECODER, text_of(tokens), task, eos_last=tokens[-1] == EOS)
        jobs.append(_job(i, i // 8, task, key, f'r{i}', tokens, raw=raw, fmt=fmt, plans=plans, answer=answer))
        jobs[-1]['eos_last'] = tokens[-1] == EOS
    return jobs


def _shift(jobs, ineligible_shift):
    """+0.01 on every token; +ineligible_shift on overlap tokens of label-INELIGIBLE spans (must never reach label metrics)."""
    out = {}
    for j in jobs:
        v = [x + 0.01 for x in j['old_logps_PRIVATE']]
        if not j['answer']['label_metric_eligible']:
            for p in j['answer']['overlap_positions']:
                v[p] = min(0., j['old_logps_PRIVATE'][p] + ineligible_shift)
        out[j['index']] = v
    return out


def test_label_metrics_never_pool_ineligible_spans_and_overlap_is_separate():
    jobs = mixed_jobs()
    a = nr.analyze_train(jobs, {'A53': _shift(jobs, 0.4), 'B53': _shift(jobs, 0.4)})
    am = a['answer_measurement']['A53']
    assert am['label_metric_eligible'] == 25 and am['label_metric_ineligible'] == 7
    assert am['by_status']['STRICT_CANONICAL'] == {'n': 27, 'label_metric_eligible': 25, 'ineligibility_reasons': {'PARTIAL_BPE_BOUNDARY': 2}}
    assert am['measurement_kind_counts'] == {'LABEL_ONLY': 25, 'NONE': 1, 'WHOLE_OVERLAPPING_TOKEN_DIAGNOSTIC': 6}
    assert am['by_task']['MRI'] == {'n': 12, 'label_metric_eligible': 5} and am['by_task']['DALLAS'] == {'n': 20, 'label_metric_eligible': 20}
    # pure-label family: eligible spans only (all +0.01), never the +0.4 ineligible overlap tokens
    label = am['label_only']
    assert label['n'] == 25 and label['delta_label_mean']['mean'] == pytest.approx(0.01) and label['delta_label_mean']['max'] == pytest.approx(0.01)
    for name, stratum in a['strata']['A53'].items():
        for g, cell in stratum.items():
            if cell['delta_label_mean']['n']:
                assert cell['delta_label_mean']['max'] == pytest.approx(0.01), (name, g)
            assert not any(k.startswith('delta_overlap') or k.startswith('delta_answer') for k in cell)
    # whole-overlapping-token diagnostics stay available, explicitly named, split by status, ineligible spans only
    ov = am['whole_overlapping_token_diagnostic_by_status']
    assert set(ov) == {'STRICT_CANONICAL', 'MULTIPLE_ANSWER_BOXES', 'FORMAT_INVALID', 'STRICT_NONCANONICAL'}
    assert ov['STRICT_CANONICAL']['n'] == 2 and ov['FORMAT_INVALID']['n'] == 2
    assert ov['FORMAT_INVALID']['delta_overlap_mean']['mean'] == pytest.approx(0.4)
    assert ov['FORMAT_INVALID']['label'].startswith('CONTENT_OVERLAP_DIAGNOSTIC') and 'NOT visual accuracy' in ov['FORMAT_INVALID']['label']


def test_joint_task_format_outcome_strata_with_fixed_plan_terms():
    jobs = mixed_jobs()
    new = {'A53': _shift(jobs, 0.), 'B53': {j['index']: list(j['old_logps_PRIVATE']) for j in jobs}}
    a = nr.analyze_train(jobs, new)
    joint = a['joint_strata']['A53']
    assert joint['MRI|STRICT_VALID|CORRECT']['n'] == 4                        # FLAIR_OK x2, MERGED, T1 brace-BPE
    assert joint['MRI|STRICT_VALID|RAW0_CANONICAL_WRONG_CLASS']['n'] == 3     # canonical but wrong class (incl. reasoning-box case)
    assert joint['MRI|FORMAT_INVALID|RAW0_FORMAT_GATE']['n'] == 2             # raw 0 because of format, NOT a wrong MRI class
    assert joint['MRI|STRICT_VALID|RAW0_NONCANONICAL_OR_MISSING_ANSWER']['n'] == 1
    assert joint['MRI|PARTIAL_FORMAT|CORRECT']['n'] == 1 and joint['MRI|PARTIAL_FORMAT|RAW0_NONCANONICAL_OR_MISSING_ANSWER']['n'] == 1
    assert joint['DALLAS|STRICT_VALID|CORRECT']['n'] == 10 and joint['DALLAS|STRICT_VALID|RAW0_CANONICAL_WRONG_CLASS']['n'] == 10
    assert sum(c['n'] for c in joint.values()) == len(jobs)
    inv = joint['MRI|FORMAT_INVALID|RAW0_FORMAT_GATE']
    assert inv['label_metric_eligible'] == 0 and inv['delta_label_mean']['n'] == 0 and inv['measurement_kind_counts'] == {'WHOLE_OVERLAPPING_TOKEN_DIAGNOSTIC': 2}
    cor = joint['MRI|STRICT_VALID|CORRECT']
    assert cor['label_metric_eligible'] == 2 and cor['delta_label_mean']['n'] == 2
    # same-checkpoint fixed-plan advantage/admission and surrogate terms per joint cell; cells sum exactly to the plan objective
    pa = cor['plans']['A']
    assert (pa['admitted'], pa['advantage_positive'], pa['advantage_negative'], pa['advantage_zero']) == (4, 4, 0, 0)
    assert joint['DALLAS|STRICT_VALID|CORRECT']['plans']['B']['admitted'] == 0
    for plan in nr.PLANS:
        assert math.fsum(c['plans'][plan]['surrogate_delta'] for c in joint.values()) == pytest.approx(a['surrogate_2x2']['A53'][plan]['delta'])
        assert sum(c['plans'][plan]['admitted'] for c in joint.values()) == a['surrogate_2x2']['A53'][plan]['admitted']
    assert all(c['plans'][p]['surrogate_delta'] == pytest.approx(0.) for c in a['joint_strata']['B53'].values() for p in nr.PLANS)
    # the coarse raw stratum no longer calls raw 0 'WRONG'
    assert set(a['strata']['A53']['raw']) == {'CORRECT', 'RAW_ZERO'}
    assert a['strata']['A53']['raw_outcome']['RAW0_FORMAT_GATE']['n'] == 2


def test_public_aggregate_has_no_private_fields(tmp_path):
    jobs = synthetic_jobs()
    new = {ck: {j['index']: list(j['old_logps_PRIVATE']) for j in jobs} for ck in nr.CHECKPOINTS}
    train = nr.analyze_train(jobs, new)
    dev = synthetic_dev()
    gen = {'text': text_of(FLAIR_OK), 'token_ids': FLAIR_OK, 'raw_selected_logps': [-0.1] * len(FLAIR_OK)}
    dev_out = nr.analyze_dev(dev, {ck: [gen] * 4 for ck in nr.CHECKPOINTS}, decoder=DECODER)
    pub = nr.public_aggregate({'train': train, 'dev': dev_out, 'boundaries': nr.TOTALS})
    assert set(pub['train']) >= {'joint_strata', 'answer_measurement'}
    mixed = mixed_jobs()
    mixed_pub = nr.public_aggregate({'train': nr.analyze_train(mixed, {ck: _shift(mixed, .2) for ck in nr.CHECKPOINTS}), 'dev': dev_out,
                                     'boundaries': nr.TOTALS})
    text = json.dumps(pub) + json.dumps(mixed_pub)
    for forbidden in ('response_id', 'token_ids', 'row_PRIVATE', '_PRIVATE', 'dev_case_index', 'image_sha256', '/img', 'patient'):
        assert forbidden not in text
