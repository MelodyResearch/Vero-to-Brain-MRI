# Public Vero metadata counterexamples: bounded CPU reproduction

Status: **`TESTED_CPU`**. Independent Codex reviewer `/root/overnight_method_stats`. This is a separate executed follow-up to the immutable initial `CODEX_PUBLIC_VERO_REVIEW.md`; Claude's response was not read before preparing or running these fixtures. No upstream or local training implementation was modified.

## Exact source and execution boundary

`scripts/reproduce_public_vero_metadata_edges.py` verifies the original public-only prompt SHA-256 `29f33b1b207be1a8ae329ff2121310d8d0bb74ae7ee01cc157766d67a527f11d`, reconstructs its numbered reward excerpt, verifies the complete1–1148 line inventory, and extracts the exact ASTs of five public functions. It does not import the upstream module or its training dependencies. The aggregate receipt `PUBLIC_VERO_COUNTEREXAMPLES.json` records each extracted function's source hash and original line span.

Source: Vero commit `025ca3c800fc95224c86323c1f7fd33dceed3090`, `vero-rl/vero_reward/math_verify_reward_type_boxed.py`.

- `_extract_image_size` and `_extract_tolerance` execute without dependency stubs affecting their logic.
- `_coerce_float`, `_numeric_candidates` and `grade_numeric` retain their exact public bodies. The external `parse` dependency is explicitly stubbed to return no additional candidates. All tested strings are ordinary decimals, so the upstream direct Python `float` path supplies the relevant candidates before that parser call.
- Sympy and `verify` are not imported; their stand-ins raise if reached. No such call occurred. The parser stub was called10 times across the numeric cases.
- No Torch, MLX, GPU, model, MRI, real `math_verify` parser, grounding handler or complete reward manager executed. This is not numerical framework parity or a training test.

## Executed results

All13 assertions passed: five image-size cases, two tolerance-extraction cases and six numeric-grading cases.

| Image-size input | Actual public helper result |
|---|---|
| `image_size=[100,200]` | `None` |
| `image_size=(100,200)` | `(100,200)` |
| `image_sizes=[[100,200]]` | `(100,200)` |
| `image_size={"width":100,"height":200}` | `(100,200)` |
| Missing size | `None` |

The flat-list inconsistency is reproduced directly from [lines386–404](https://github.com/zlab-princeton/vero/blob/025ca3c800fc95224c86323c1f7fd33dceed3090/vero-rl/vero_reward/math_verify_reward_type_boxed.py#L386-L404). Its downstream effect remains conditional on metadata shape and image-path recovery in the unexecuted grounding/clicking handlers.

Both positive infinity and NaN are accepted by the exact tolerance extractor at [lines330–342](https://github.com/zlab-princeton/vero/blob/025ca3c800fc95224c86323c1f7fd33dceed3090/vero-rl/vero_reward/math_verify_reward_type_boxed.py#L330-L342).

| Numeric fixture | Actual grade |
|---|---:|
| Gold1, prediction999, tolerance `+inf` | **1** |
| Gold1, prediction999, tolerance0.1 |0|
| Gold1, prediction1.05, tolerance0.1 |1|
| Gold0, prediction0.05, tolerance0.1 |1|
| Gold1, prediction999, tolerance `NaN` |0|
| Gold1, prediction999, tolerance `-inf` |-1|

The executed infinite-tolerance result reaches the exact comparison at [lines910–931](https://github.com/zlab-princeton/vero/blob/025ca3c800fc95224c86323c1f7fd33dceed3090/vero-rl/vero_reward/math_verify_reward_type_boxed.py#L910-L931). External parser behavior was deliberately not reproduced. For the particular infinite-tolerance counterexample, the ordinary finite direct candidates already make the comparison succeed; adding other parser candidates cannot remove that successful pair in the displayed loop.

## Interpretation and remaining tests

These fixtures upgrade the two initial source counterexamples to executed **helper-level** evidence. They do not demonstrate that upstream training ever supplied malformed metadata, nor that its final task scores or results were affected. No complete-manager or actual dataset result follows.

Suggested minimal corrections remain: recognize flat numeric size pairs before unwrapping an outer list; require finite nonnegative tolerances and explicitly validate invalid metadata. Those corrections were not applied to upstream or local code in this task. A future patched-helper regression should reverse only the invalid cases while preserving tuple/nested/dictionary geometry and ordinary finite tolerance behavior. The actual grounding manager should then test equivalent coordinate scores for all supported size shapes.

Reproduction:

```sh
python3 scripts/reproduce_public_vero_metadata_edges.py \
  --prompt /private/tmp/vero-public-claude-review-20261015/prompt.txt \
  --output /private/tmp/vero-metadata-edge-replay.json
```

The frozen public prompt is required and its digest is checked; the output destination must be new. The script does not fetch source or send data externally. **Scoped verdict: `PASS_WITH_LIMITATIONS` for reproducing the two helper defects; complete recipe/manager correctness remains `NOT_VERIFIED`.**
