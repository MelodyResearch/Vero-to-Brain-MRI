"""CPU TDD for same-context native probability resolution; no native imports."""
import math
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from mri_rl import native_probability_resolution_v1 as p


class ProbabilityResolutionCPU(unittest.TestCase):
    def test_constant_logit_offset_does_not_change_categorical_distribution(self):
        left = np.array([[1., 0., -2.], [3., 2., 1.]])
        right = left+np.array([[64.], [-8.]])
        result = p.distribution_comparison(left, right, [0, 1])
        self.assertLess(result['max_TV'], 1e-14)
        self.assertLess(result['max_JS'], 1e-14)
        self.assertLess(result['max_centered_logit_delta'], 1e-14)
        self.assertLess(result['max_full_vocab_normalized_logp_delta'], 1e-14)
        self.assertAlmostEqual(result['max_raw_logit_delta'], 64.)

    def test_unselected_vocab_difference_cannot_pass_selected_only_gate(self):
        left = np.zeros((1, 4)); right = np.array([[0., .08, -.08, 0.]])
        result = p.distribution_comparison(left, right, [0])
        self.assertLess(result['max_selected_normalized_logp_delta'], .002)
        self.assertGreater(result['max_full_vocab_normalized_logp_delta'], .07)
        self.assertFalse(p.acceptance(result, dtype_observed=True, context_equal=True)['B2_numeric_acceptable'])

    def test_bf16_native_logps_renormalization_separates_offset_and_rounding(self):
        base = np.array([[-.2, -2., -3.]])
        pure_shift = base+.1
        self.assertLess(p.distribution_comparison(base, pure_shift, [0])['max_TV'], 1e-14)
        rounded = np.array([[0., -1.75, -2.75]])
        self.assertGreater(p.distribution_comparison(base, rounded, [0])['max_TV'], .001)

    def test_future_B3_gate_requires_explicit_dtype_context_and_numeric_evidence(self):
        result = p.distribution_comparison(np.array([[1., 0.]]), np.array([[1., 0.]]), [0])
        proof = {'legacy_reproduction': {'native_selected_max_abs_delta': 0., 'stable_selected_max_abs_delta': 0., 'frozen_scorer_selected_max_abs_delta': 0.},
                 'direct_stable': {'max_full_vocab_logp_delta': 0., 'max_selected_logp_delta': 0.}}
        for dtype, context in [(False, True), (True, False), (False, False)]:
            self.assertFalse(p.acceptance(result, dtype_observed=dtype, context_equal=context, **proof)['B2_numeric_acceptable'])
        good = p.acceptance(result, dtype_observed=True, context_equal=True, **proof)
        self.assertTrue(good['B2_numeric_acceptable'])
        self.assertFalse(good['B3_authorized'])

    def test_gate_rejects_nonreproduction_or_direct_F32_failure(self):
        result = p.distribution_comparison(np.array([[1., 0.]]), np.array([[1., 0.]]), [0])
        legacy = {'native_selected_max_abs_delta': 1e-4, 'stable_selected_max_abs_delta': 0., 'frozen_scorer_selected_max_abs_delta': 0.}
        direct = {'max_full_vocab_logp_delta': 0., 'max_selected_logp_delta': 0.}
        self.assertFalse(p.acceptance(result, dtype_observed=True, context_equal=True,
                                      legacy_reproduction=legacy, direct_stable=direct)['B2_numeric_acceptable'])
        legacy['native_selected_max_abs_delta'] = 0.; direct['max_full_vocab_logp_delta'] = .003
        self.assertFalse(p.acceptance(result, dtype_observed=True, context_equal=True,
                                      legacy_reproduction=legacy, direct_stable=direct)['B2_numeric_acceptable'])
        direct['max_full_vocab_logp_delta'] = 0.; legacy['frozen_scorer_selected_max_abs_delta'] = 1e-4
        self.assertFalse(p.acceptance(result, dtype_observed=True, context_equal=True,
                                      legacy_reproduction=legacy, direct_stable=direct)['B2_numeric_acceptable'])

    def test_full_route_saves_actual_frozen_centered_scorer_from_same_logits(self):
        import ast
        path = Path('/Users/vv/Documents/Codex/2026-10-15/files-pasted-by-the-user-codex-2/work/mlx-venv/lib/python3.13/site-packages/mlx/nn/layers/activations.py')
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'log_softmax')
        node.decorator_list = []
        def logsumexp(x, axis, keepdims):
            maximum = x.max(axis=axis, keepdims=keepdims)
            return maximum+np.log(np.exp(x-maximum).sum(axis=axis, keepdims=keepdims))
        mx = SimpleNamespace(float32=np.float32, ones_like=np.ones_like, max=np.max, all=np.all, isfinite=np.isfinite,
            stop_gradient=lambda v: v, logsumexp=logsumexp, eval=lambda *args: None, clear_cache=lambda: None)
        namespace = {'mx': mx}; exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        calls = []; stored = {}
        logits = np.array([[[1e7, 1e7-1], [1e7, 1e7-1]]], dtype=np.float32)
        def language(*args, **kwargs):
            calls.append(True)
            return SimpleNamespace(logits=logits)
        def save(path, arrays):
            stored.update({key: value.copy() for key, value in arrays.items()}); Path(path).write_bytes(b'CPU_ARRAY_FIXTURE')
        backend = SimpleNamespace(mx=mx, nn=SimpleNamespace(log_softmax=namespace['log_softmax']),
            model=SimpleNamespace(language_model=language), save_arrays=save)
        utility = SimpleNamespace(_fsync=lambda path: None, durable=lambda *args: None)
        with tempfile.TemporaryDirectory() as directory:
            p._full_route(backend, utility, np.array([[1, 2]]), {'inputs_embeds': np.zeros((1, 2, 3), np.float32)},
                None, {}, [0, 1], 1, 'CPU_FULL', Path(directory), lambda name: None, lambda name: None)
        self.assertEqual(len(calls), 1)
        self.assertIn('frozen_scorer_logps_F32', stored)
        self.assertGreater(float(np.max(np.abs(stored['stable_logps_F32']-stored['frozen_scorer_logps_F32']))), .3)
        expected = np.array([[-math.log1p(math.exp(-1)), -1-math.log1p(math.exp(-1))]]*2)
        np.testing.assert_allclose(stored['frozen_scorer_logps_F32'], expected, atol=1e-6)

    def test_real_route_collector_uses_legacy_whole_vs_installed_split_and_saved_positions(self):
        class Cache:
            def __init__(self): self.offset = 0
            @property
            def state(self): return self.keys, self.values
        class Language:
            _rope_deltas = _position_ids = None
            def __init__(self): self.calls = []
            def __call__(self, y=None, *, inputs=None, inputs_embeds=None, cache=None, **kwargs):
                ids = inputs if inputs is not None else y
                self.calls.append((ids.shape[-1], kwargs.get('position_ids'), kwargs.get('n_to_process')))
                if kwargs.get('rope_deltas') is not None: self._rope_deltas = kwargs['rope_deltas']
                for c in cache:
                    c.offset += ids.shape[-1]
                    c.keys = c.values = np.zeros((1, 1, c.offset, 1), dtype=np.float32)
                return SimpleNamespace(logits=np.zeros((1, ids.shape[-1], 4), np.float32), cross_attention_states=None, encoder_outputs=None)
        mx = SimpleNamespace(float32=np.float32, array=np.array, broadcast_to=np.broadcast_to, array_equal=lambda a, b: np.array(np.array_equal(a, b)),
            all=np.all, isfinite=np.isfinite, stack=np.stack, eval=lambda *args: None, stop_gradient=lambda v: v,
            clear_cache=lambda: None, logsumexp=lambda values, axis, keepdims: np.log(np.exp(values).sum(axis=axis, keepdims=keepdims)))
        language = Language(); backend = SimpleNamespace(mx=mx, model=SimpleNamespace(language_model=language),
            save_arrays=lambda path, arrays: Path(path).write_bytes(b'CPU_FAKE_DIAGNOSTIC_ARRAYS'))
        utility = SimpleNamespace(_fsync=lambda path: None, durable=lambda path, value: None)
        features = {'inputs_embeds': np.zeros((1, 7, 2), np.float32), 'position_ids': np.broadcast_to(np.arange(7)[None, None, :], (3, 1, 7)),
                    'rope_deltas': np.array([[0]])}
        inputs = {'input_ids': np.array([[0, 1, 2, 3, 0]], np.int32), 'pixel_values': None, 'attention_mask': None}
        with tempfile.TemporaryDirectory() as directory:
            for route, lengths in [(p.ROUTES[0], [5, 1, 1]), (p.ROUTES[1], [4, 1, 1, 1]), (p.ROUTES[2], [4, 1, 1, 1])]:
                language.calls = []
                p._kv_route(backend, utility, inputs, p.prefix_features(features, 5), features, [1, 2, 3], route,
                            Path(directory), lambda name: None, lambda name: None, lambda *args, **kwargs: [Cache()])
                self.assertEqual([c[0] for c in language.calls], lengths)
                if route == p.ROUTES[2]:
                    self.assertEqual(language.calls[-2][1][0, 0, 0], 5)
                    self.assertEqual(language.calls[-1][1][0, 0, 0], 6)

    def test_causal_position_schedule_scores_all_saved_tokens_including_EOS(self):
        schedule = p.causal_schedule(213, 17)
        self.assertEqual(schedule[0], {'response_index': 0, 'input_start': 0, 'input_end': 213, 'score_position': 212})
        self.assertEqual(schedule[1]['input_start'], 213)
        self.assertEqual(schedule[-1]['input_start'], 228)
        self.assertEqual(schedule[-1]['score_position'], 228)
        self.assertEqual(len(schedule), 17)
        with self.assertRaises(ValueError): p.causal_schedule(0, 17)

    def test_f32_activation_control_preserves_metadata_and_source_features(self):
        features = {'inputs_embeds': np.ones((1, 8, 3), np.float16), 'deepstack_visual_embeds': [np.ones((2, 3), np.float16)],
            'position_ids': np.arange(8).reshape(1, 1, 8), 'rope_deltas': np.array([[-4]]),
            'visual_pos_masks': np.array([[0, 1, 1, 0, 0, 0, 0, 0]], bool)}
        controlled = p.activation_control(features, np.float32)
        self.assertEqual(controlled['inputs_embeds'].dtype, np.float32)
        self.assertEqual(controlled['deepstack_visual_embeds'][0].dtype, np.float32)
        self.assertEqual(features['inputs_embeds'].dtype, np.float16)
        self.assertIs(controlled['position_ids'], features['position_ids'])
        self.assertIs(controlled['rope_deltas'], features['rope_deltas'])
        prefix = p.prefix_features(features, 5)
        self.assertEqual(prefix['inputs_embeds'].shape, (1, 5, 3))
        self.assertEqual(prefix['position_ids'].shape[-1], 5)
        self.assertIs(prefix['deepstack_visual_embeds'], features['deepstack_visual_embeds'])

    def test_nonfinite_and_misaligned_distributions_fail_closed(self):
        for left, right, tokens in [(np.array([[math.nan, 0]]), np.array([[0., 0.]]), [0]),
            (np.ones((2, 3)), np.ones((1, 3)), [0]), (np.ones((1, 3)), np.ones((1, 3)), [3])]:
            with self.assertRaises(ValueError): p.distribution_comparison(left, right, tokens)

    def test_budget_contains_no_generation_backward_or_update(self):
        result = p.budget(17)
        self.assertEqual(result['vision_embedding_calls'], 2)
        self.assertEqual(result['language_model_calls'], 55)
        self.assertEqual(result['generation_calls'], 0)
        self.assertEqual(result['backward_calls'], 0)
        self.assertEqual(result['optimizer_updates'], 0)
        with self.assertRaises(ValueError): p.budget(18)

    def test_import_is_CPU_only(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys; import mri_rl.native_probability_resolution_v1; assert not any(k.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'} for k in sys.modules)"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
