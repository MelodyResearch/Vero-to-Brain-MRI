"""Future AR logits processor: upcast before the existing normalization.

CPU import only. The caller supplies its existing array API explicitly; this
module never imports MLX, loads a model, samples, or changes weights. This hook
has not been executed on a native model. It targets the inspected ordinary
mlx_vlm.generate.ar.generate_step processor loop only.
"""
from __future__ import annotations


class FP32LogitsProcessor:
    """Use as the final custom AR logits processor; do no score normalization.

    Existing ar._step then computes its usual direct logsumexp on float32 logits.
    Any later processor that downcasts can invalidate this guarantee. This hook
    cannot restore information already lost when model logits were formed in
    BF16, and it does not address differences between KV and full forward paths.
    """
    def __init__(self, array_api):
        self.array_api = array_api

    def __call__(self, tokens, logits):
        converted = logits.astype(self.array_api.float32)
        if converted.dtype != self.array_api.float32:
            raise ValueError('AR logits upcast must return float32')
        if converted.shape != logits.shape:
            raise ValueError('AR logits upcast changed shape')
        return converted


def append_fp32_logits_processor(existing, array_api):
    """Copy the supplied custom list and append the upcast as its final hook.

    Installed AR appends this list after built-in score processors. This function
    does not install or launch anything; future callers must pass the returned
    list as logits_processors under a separately reviewed native execution scope.
    """
    return [*(existing or ()), FP32LogitsProcessor(array_api)]


def cpu_bf16_normalization_counterexample():
    """IEEE BF16 round-to-nearest-even arithmetic counterexample, CPU only.

    The logits [64,63] are exactly representable in BF16, isolating rounding of
    logsumexp/subtraction. This models BF16 output rounding of the mathematical
    reduction, not the unobserved implementation details of MLX's reduction
    kernel. The hook path uses NumPy float32 after exactly the same input logits.
    """
    import math
    from types import SimpleNamespace
    import numpy as np

    def bf16_rne(values):
        array = np.asarray(values, dtype=np.float32)
        bits = array.view(np.uint32)
        rounded = (bits+np.uint32(0x7fff)+((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xffff0000)
        return rounded.view(np.float32)

    class BF16Mock:
        dtype = 'CPU_BF16_RNE_MOCK'
        def __init__(self, values):
            self.values = bf16_rne(values)
            self.shape = self.values.shape
        def astype(self, dtype):
            return self.values.astype(dtype)

    logits = np.array([[64., 63.]], dtype=np.float32)
    reference64 = logits.astype(np.float64)
    maximum64 = reference64.max(axis=-1, keepdims=True)
    reference = reference64-(maximum64+np.log(np.exp(reference64-maximum64).sum(axis=-1, keepdims=True)))
    lse_rounded = bf16_rne(maximum64+np.log(np.exp(reference64-maximum64).sum(axis=-1, keepdims=True)))
    old_logps = bf16_rne(bf16_rne(logits)-lse_rounded)
    processed = FP32LogitsProcessor(SimpleNamespace(float32=np.float32))(None, BF16Mock(logits))
    maximum32 = processed.max(axis=-1, keepdims=True)
    new_logps = processed-(maximum32+np.log(np.exp(processed-maximum32).sum(axis=-1, keepdims=True)))
    return {'status': 'TESTED_CPU_ARITHMETIC_COUNTEREXAMPLE', 'logits_exact_in_bf16': logits[0].tolist(),
        'bf16_selected_logps_fixture_only': old_logps[0].tolist(),
        'hook_fp32_logps_fixture_only': new_logps[0].tolist(),
        'reference_float64_logps_fixture_only': reference[0].tolist(),
        'bf16_max_abs_logp_error': float(np.max(np.abs(old_logps-reference))),
        'hook_fp32_max_abs_logp_error': float(np.max(np.abs(new_logps-reference))),
        'bf16_probability_sum': math.fsum(math.exp(float(x)) for x in old_logps[0]),
        'hook_fp32_probability_sum': math.fsum(math.exp(float(x)) for x in new_logps[0]),
        'model_forwards': 0, 'optimizer_updates': 0, 'native_runtime_executed': False,
        'limitation': 'BF16 rounding fixture; no claim of bitwise MLX kernel reproduction or native hook execution.'}
