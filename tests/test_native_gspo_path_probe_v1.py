"""CPU-only qualification of bounded native probe contracts and fail-closed paths."""
import ast
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest
import gc
import json
import tempfile
import weakref
from types import SimpleNamespace
from unittest.mock import patch

from mri_rl import native_gspo_path_probe_v1 as p


class PathProbeCPU(unittest.TestCase):
    def test_import_never_loads_native_runtime(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys; import mri_rl.native_gspo_path_probe_v1; assert not any(k == 'mlx' or k.startswith('mlx.') or k == 'mlx_vlm' or k.startswith('mlx_vlm.') for k in sys.modules)"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_first_groups_are_deterministic_and_outcome_independent(self):
        def row(task, index, accepted=True):
            return {'task': task, 'group_index': index, 'accepted': accepted, 'informative_advantages': accepted,
                    'advantages': [.5, .5, .5, -1.5], 'lengths': [17]*4}
        rows = [row('mri_sequence', 0, False), row('mri_sequence', 1), row('mri_sequence', 2),
                row('mri_visit_correspondence', 3), row('mri_visit_correspondence', 4)]
        self.assertEqual([r['group_index'] for r in p.select_first_groups(rows)], [1, 3])
        self.assertEqual(p.select_first_groups(rows), p.select_first_groups(rows[:4]))
        with self.assertRaises(ValueError): p.select_first_groups(rows[:3])

    def test_bound_total_budget_and_no_generation(self):
        budget = p.budget([{'lengths': [17, 21, 18, 21]}, {'lengths': [17, 18, 16, 16]}])
        self.assertEqual(budget['B3_backward_calls'], 24)
        self.assertEqual(budget['KV_prefills'], 8)
        self.assertEqual(budget['KV_token_forwards'], 136)
        self.assertEqual(budget['optimizer_updates'], 0)
        self.assertEqual(budget['new_generated_tokens'], 0)
        with self.assertRaises(ValueError): p.budget([{'lengths': [257]*4}])
        with self.assertRaises(ValueError): p.budget([{'lengths': [1]*4}]*3)

    def test_each_saved_position_and_eos_are_compared(self):
        for changed in range(3):
            candidate = [-1., -2., -3.]; candidate[changed] += .003
            result = p.finite_logp_comparison([-1., -2., -3.], candidate)
            self.assertFalse(result['within_frozen_tolerance'])
        with self.assertRaises(ValueError): p.finite_logp_comparison([-1.], [-1., -2.])
        with self.assertRaises(ValueError): p.finite_logp_comparison([-1.], [math.nan])

    def test_strict_label_projection_does_not_salvage_nonclass(self):
        text = '<think>Evidence</think><answer>\\boxed{contrast-enhanced\\ T1}</answer>'
        result = p.label_support(list(range(len(text))), lambda ids: text[:len(ids)], 'mri_sequence')
        self.assertEqual(result['status'], 'NOT_SEPARABLE_STRICT_NONCLASS')
        self.assertEqual(sum(result['mask']), 0)
        valid = '<think>Evidence</think><answer>\\boxed{\\text{FLAIR}}</answer>'
        result = p.label_support(list(range(len(valid))), lambda ids: valid[:len(ids)], 'mri_sequence')
        self.assertEqual(result['label_tokens'], 5)
        self.assertEqual(''.join(ch for ch, use in zip(valid, result['mask']) if use), 'FLAIR')

    def test_native_token_overlap_includes_shared_boundary_token(self):
        text = '<think>Evidence</think><answer>\\boxed{same}</answer>'
        start = text.index('same'); chunks = [text[:start-1], text[start-1:start+2], text[start+2:]]
        result = p.label_support([0, 1, 2], lambda ids: ''.join(chunks[i] for i in ids), 'mri_visit_correspondence')
        self.assertEqual(result['mask'], [0, 1, 1])

    def test_same_lengths_do_not_certify_prepared_tensor_identity(self):
        class Utility:
            @staticmethod
            def _fingerprints(backend, values): return values
        values = {key: 'same-shape-content' for key in ('input_ids', 'pixel_values', 'attention_mask', 'image_grid_thw')}
        for key in values:
            other = {**values, key: 'different-same-shape-content'}
            with self.assertRaises(ValueError): p.strict_tensor_identity(None, values, other, Utility)
        self.assertEqual(p.strict_tensor_identity(None, values, dict(values), Utility), values)

    def test_native_loss_ast_has_identical_ratio_clipping_expression(self):
        source = Path(p.__file__).with_name('mlx_continuous_gspo.py')
        module = ast.parse(source.read_text())
        node = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == 'loss_fn')
        # Source closure binds existing AST. No copied CE loss or optimizer API is present.
        helper = ast.parse(Path(p.__file__).read_text())
        calls = [n.func for n in ast.walk(helper) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(f, ast.Attribute) and (f.attr in ('generate', 'save_adapter', 'save_weights') or
            (f.attr == 'update' and isinstance(f.value, ast.Attribute) and f.value.attr == 'optimizer')) for f in calls))
        self.assertIn('stop_gradient', ast.dump(node))
        self.assertIn('clip', ast.dump(node))

    def test_actual_helper_failure_frames_release_model_and_preserve_error(self):
        class Model:
            def eval(self): pass
        backend = SimpleNamespace(model=Model(), mx=SimpleNamespace(clear_cache=lambda: None), active=None,
                                  optimizer_state=lambda: {}, rng_state=lambda: {})
        reference = weakref.ref(backend.model)
        records = {}
        utility = SimpleNamespace(_weights=lambda backend: {}, _fingerprints=lambda backend, values: {},
                                  durable=lambda path, value: records.__setitem__(Path(path).name, value))
        group = {'raw': {'responses': [{'token_ids': [1]} for _ in range(4)]},
                 'row': {'extra_info': {'internal_split': 'dev'}}}
        with tempfile.TemporaryDirectory() as directory:
            try:
                p.run_group(backend, group, Path(directory)/'group', utility, lambda name: None)
            except ValueError as error:
                self.assertEqual(str(error), 'Actual TRAIN image changed')
                self.assertTrue(records['exit_integrity_PRIVATE.json']['weights_unchanged'])
                self.assertTrue(records['exit_integrity_PRIVATE.json']['optimizer_Python_MLX_RNG_unchanged'])
                backend.model = None; gc.collect()
                self.assertIsNotNone(reference(), 'Regression must first reproduce traceback retention')
                p.clear_unwound_exception_frames(error); gc.collect()
                self.assertIsNone(reference(), 'Completed helper traceback still retains model')
                self.assertEqual(str(error), records['STOP_PRIVATE.json']['error'])
            else:
                self.fail('Expected actual TRAIN guard failure')

    def test_native_and_stable_KV_logps_share_forward_but_keep_dtypes(self):
        class Tensor:
            def __init__(self, values, dtype='float16'):
                self.values, self.dtype = values, dtype
            def __getitem__(self, key):
                if len(key) == 3: return self
                return Tensor(self.values[key[1]], self.dtype)
            def astype(self, dtype): return Tensor(self.values, dtype)
            def __sub__(self, right):
                values = [v-right.values for v in self.values]
                if self.dtype == 'float16': values = [round(v, 2) for v in values]
                return Tensor(values, self.dtype)
            def item(self): return self.values
        forwards = []
        def language_model(*args, **kwargs):
            forwards.append(True)
            return SimpleNamespace(logits=Tensor([1., 1.1]), cross_attention_states=None, encoder_outputs=None)
        def logsumexp(tensor, **kwargs):
            value = math.log(sum(math.exp(v) for v in tensor.values))
            return Tensor(round(value, 2) if tensor.dtype == 'float16' else value, tensor.dtype)
        mx = SimpleNamespace(float32='float32', logsumexp=logsumexp, eval=lambda *args: None,
                             array=lambda *args, **kwargs: None, clear_cache=lambda: None)
        backend = SimpleNamespace(mx=mx, model=SimpleNamespace(language_model=language_model,
            get_input_embeddings=lambda *args, **kwargs: SimpleNamespace(inputs_embeds=None, to_dict=lambda: {})))
        cache = SimpleNamespace(make_prompt_cache=lambda *args, **kwargs: [SimpleNamespace(state=[])])
        modules = {'mlx_vlm': SimpleNamespace(), 'mlx_vlm.models': SimpleNamespace(), 'mlx_vlm.models.cache': cache}
        with patch.dict(sys.modules, modules):
            result = p.forced_kv_logps(backend, {'input_ids': Tensor([]), 'pixel_values': None, 'attention_mask': None},
                                      [0, 1], lambda name: None, lambda name: None)
        self.assertEqual(len(forwards), 2)
        self.assertNotEqual(result['native_dtype_selected_logps'], result['fp32_stable_selected_logps'])
        self.assertTrue(all(d['native_logp'] == 'float16' and d['stable_logp'] == 'float32' for d in result['per_token_dtypes']))

    def test_resource_hold_cannot_start_work_after_deadline(self):
        runner = Path(p.__file__).parents[2]/'scripts/run_native_gspo_path_probe_v1.py'
        source = ast.parse(runner.read_text())
        execute = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
        node = next(n for n in execute.body if isinstance(n, ast.FunctionDef) and n.name == 'before')
        clock = [1799.]
        def gate(name, loading):
            clock[0] = 1801.
            return 'CONTINUE'
        namespace = {'require': p.require, 'time': SimpleNamespace(monotonic=lambda: clock[0]), 'started': 0.,
            'probe': p, 'gate': SimpleNamespace(before_unit=gate), 'is_loading_unit': lambda name, cache: False, 'cache': {}}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(runner), 'exec'), namespace)
        with self.assertRaisesRegex(ValueError, 'budget exhausted during resource hold'):
            namespace['before']('native_forward')


if __name__ == '__main__': unittest.main()
