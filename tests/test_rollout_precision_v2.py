"""CPU TDD for future AR normalization precision; no model/native imports."""
import ast
import math
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
import numpy as np

from mri_rl.rollout_precision_v2 import FP32LogitsProcessor, append_fp32_logits_processor, cpu_bf16_normalization_counterexample


class RolloutPrecisionCPU(unittest.TestCase):
    def test_hook_upcasts_same_logits_and_preserves_processor_order(self):
        calls = []
        def existing(tokens, logits):
            calls.append((tokens, logits))
            return logits+np.float32(.125)
        prior = [existing]
        processors = append_fp32_logits_processor(prior, SimpleNamespace(float32=np.float32))
        self.assertEqual(prior, [existing])
        self.assertIs(processors[0], existing)
        self.assertIsInstance(processors[-1], FP32LogitsProcessor)
        source = np.array([[64., 63.]], dtype=np.float16)
        original = source.copy(); value = source
        tokens = object()
        for processor in processors: value = processor(tokens, value)
        self.assertEqual(value.dtype, np.dtype('float32'))
        np.testing.assert_array_equal(source, original)
        np.testing.assert_array_equal(value, (source+np.float32(.125)).astype(np.float32))
        self.assertIs(calls[0][0], tokens)

    def test_processor_rejects_non_f32_cast_result(self):
        wrong = SimpleNamespace(dtype='bf16', shape=(1, 2), astype=lambda dtype: wrong)
        with self.assertRaisesRegex(ValueError, 'float32'):
            FP32LogitsProcessor(SimpleNamespace(float32='float32'))(None, wrong)

    def test_bf16_rounding_loses_probability_information_restored_by_upcast(self):
        result = cpu_bf16_normalization_counterexample()
        self.assertEqual(result['logits_exact_in_bf16'], [64., 63.])
        self.assertGreater(result['bf16_max_abs_logp_error'], .18)
        self.assertLess(result['hook_fp32_max_abs_logp_error'], 5e-6)
        self.assertGreater(abs(result['bf16_probability_sum']-1.), .1)
        self.assertLess(abs(result['hook_fp32_probability_sum']-1.), 5e-6)
        self.assertEqual(result['model_forwards'], 0)
        self.assertEqual(result['native_runtime_executed'], False)

    def test_fp32_preserves_ranking_and_extreme_logits_are_finite(self):
        values = np.array([[1000., 999., -1000.]], dtype=np.float16)
        processed = FP32LogitsProcessor(SimpleNamespace(float32=np.float32))(None, values)
        maximum = processed.max(axis=-1, keepdims=True)
        logps = processed-(maximum+np.log(np.exp(processed-maximum).sum(axis=-1, keepdims=True)))
        self.assertTrue(np.isfinite(logps).all())
        self.assertEqual(int(np.argmax(logps)), int(np.argmax(values)))

    def test_module_import_does_not_import_native_runtime(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys; import mri_rl.rollout_precision_v2; assert not any(k.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'} for k in sys.modules)"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
