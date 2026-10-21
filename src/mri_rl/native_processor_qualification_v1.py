"""CPU qualification of the PINNED native Qwen3-VL processor on the TRAIN rows (no weights, no MLX, no torch).

Loads mlx_vlm's own torch-free numpy ``Qwen3VLProcessor`` and ``prompt_utils`` FROM FILE (the package __init__ would
initialise Metal), with only ``models.base`` replaced by a stub whose ``load_chat_template`` is the pinned function
source and whose ``to_mlx`` returns numpy. It replays ``mlx_vlm.utils.prepare_inputs`` -> ``process_inputs`` exactly
(``process_image``/``load_image``: EXIF transpose + RGB; keyword selection by signature; ``return_tensors`` popped inside
the processor). What this proves is the CPU processor contract (pixels, grid, pad expansion, prompt tokens). It does NOT
prove the backend-F32 model inputs (those are still NOT_RUN) and the prompt text is rendered with the real pinned
template, never inferred from text counts.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import re
import sys
import types
import numpy as np

STATUS = 'CPU_PROCESSOR_CONTRACT_ONLY_NOT_BACKEND_F32_MODEL_INPUTS'
IMAGE_PAD, VISION_START, VISION_END = '<|image_pad|>', '<|vision_start|>', '<|vision_end|>'
PATCH, MERGE, TPS = 16, 2, 2
CAPSULE_REQUIRED_KEYS = ('expected_png_sha256', 'pair_key', 'split')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ---------------------------------------------------------------- pinned loading
def _mlx_vlm_dir():
    spec = importlib.util.find_spec('mlx_vlm')     # locating a top-level package does not execute it
    require(spec is not None and spec.submodule_search_locations, 'mlx_vlm package files unavailable')
    return Path(next(iter(spec.submodule_search_locations)))


def _function_source(path, name):
    source = Path(path).read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno])
    raise ValueError(f'{name} not found in pinned {path.name}')


def pinned_sources():
    root = _mlx_vlm_dir()
    files = {'processor': root / 'models/qwen3_vl/processing_qwen3_vl.py', 'models_base': root / 'models/base.py',
             'prompt_utils': root / 'prompt_utils.py', 'utils': root / 'utils.py'}
    import transformers
    return {**{k: sha256_file(v) for k, v in files.items()}, 'transformers_version': transformers.__version__,
            'numpy_version': np.__version__}


def load_formatter():
    path = _mlx_vlm_dir() / 'prompt_utils.py'
    spec = importlib.util.spec_from_file_location('mri_prompt_utils_cpu_pq', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pinned_processor(model_dir):
    """Qwen3VLProcessor built exactly as its own from_pretrained does, from file, without importing mlx."""
    root = _mlx_vlm_dir()
    names = ('_mri_pq', '_mri_pq.models', '_mri_pq.models.qwen3_vl', '_mri_pq.models.base')
    for name in names:
        module = types.ModuleType(name)
        if name != '_mri_pq.models.base':
            module.__path__ = []
        sys.modules[name] = module
    base = sys.modules['_mri_pq.models.base']
    scope = {}
    exec(_function_source(root / 'models/base.py', 'load_chat_template'), scope)    # pinned source, no mlx names used
    base.load_chat_template = scope['load_chat_template']

    def to_numpy(data):          # replaces to_mlx: identical conversion rules, numpy instead of mx.array
        result = {}
        for key, value in data.items():
            if isinstance(value, list):
                try:
                    result[key] = np.array(value)
                except (ValueError, TypeError):
                    result[key] = value
            else:
                result[key] = value
        return result
    base.to_mlx = to_numpy
    base.install_auto_processor_patch = lambda *a, **k: None
    path = root / 'models/qwen3_vl/processing_qwen3_vl.py'
    spec = importlib.util.spec_from_file_location('_mri_pq.models.qwen3_vl.processing_qwen3_vl', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    processor = module.Qwen3VLProcessor.from_pretrained(str(model_dir), local_files_only=True)
    require('mlx' not in sys.modules and 'mlx.core' not in sys.modules, 'MLX was imported by CPU processor qualification')
    return processor


# ---------------------------------------------------------------- replay of prepare_inputs for one image
def load_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert('RGB')


def run_processor(processor, prompt, image):
    """Same call mlx_vlm.utils.prepare_inputs/process_inputs makes for one image and one rendered prompt."""
    require(isinstance(prompt, str) and prompt.count(IMAGE_PAD) == 1, 'Exactly one image placeholder required in the rendered prompt')
    parameters = inspect.signature(getattr(processor, 'process', processor)).parameters
    args = {'text': [prompt], 'images': [image], 'padding': True, 'return_tensors': 'mlx'}
    if 'padding_side' in parameters:
        args['padding_side'] = 'left'
    if 'add_special_tokens' in parameters:
        args['add_special_tokens'] = True
    out = processor(**args)
    result = {key: np.asarray(out[key]) for key in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')}
    result['input_ids'] = result['input_ids'][0]
    result['attention_mask'] = result['attention_mask'][0]
    return result


def expected_visual_tokens(height, width, *, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280):
    """Independent re-derivation of the pinned smart_resize and merged visual-token count."""
    factor = PATCH * MERGE
    require(max(height, width) / min(height, width) <= 200, 'Aspect ratio out of range')
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h, w = max(factor, math.floor(height / beta / factor) * factor), max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h, w, (h // PATCH) * (w // PATCH) // MERGE ** 2


# ---------------------------------------------------------------- leakage / joins
def leak_hits(text, forbidden):
    """Names of forbidden identifier classes present in model-visible text. Short codes need word boundaries."""
    hits = []
    for name, value in forbidden.items():
        if not value:
            continue
        pattern = re.escape(str(value))
        found = re.search(r'(?<![A-Za-z0-9])' + pattern + r'(?![A-Za-z0-9])', text, flags=re.I if len(str(value)) >= 6 else 0)
        if found:
            hits.append(name)
    return hits


def join_capsule(rows, capsule):
    """Row-by-row join against the curated capsule. Missing keys are a REAL gap, never defaulted."""
    entries = capsule.get('rows') if isinstance(capsule, dict) else None
    missing, ok = {}, isinstance(entries, dict)
    if not ok:
        return {'ok': False, 'reason': 'capsule has no rows mapping keyed by row id', 'missing_keys': {}}
    for row in rows:
        entry = entries.get(row['id'])
        absent = [k for k in CAPSULE_REQUIRED_KEYS if not isinstance(entry, dict) or k not in entry]
        if absent:
            missing[row['id']] = absent
    return {'ok': not missing, 'missing_keys': missing, 'rows_joined': len(rows) - len(missing)}


# ---------------------------------------------------------------- per-row and pair qualification
def render(formatter, processor, model_config, system_instruction, question, *, enable_thinking):
    return formatter.apply_chat_template(processor, model_config,
        [{'role': 'system', 'content': system_instruction}, {'role': 'user', 'content': question}],
        add_generation_prompt=True, num_images=1, enable_thinking=enable_thinking)


def _row_record(processor, row, prompt, image, forbidden):
    out = run_processor(processor, prompt, image)
    ids, grid = out['input_ids'], out['image_grid_thw']
    require(grid.shape == (1, 3) and int(grid[0, 0]) == 1, 'One still-image grid required')
    merged = int(grid[0].prod()) // MERGE ** 2
    pads = int((ids == processor.image_token_id).sum())
    starts, ends = int((ids == processor.vision_start_token_id).sum()), int((ids == processor.vision_end_token_id).sum())
    pixels = out['pixel_values']
    height, width = image.size[1], image.size[0]
    ip = processor.image_processor
    eh, ew, etokens = expected_visual_tokens(height, width, min_pixels=ip.min_pixels, max_pixels=ip.max_pixels)
    decoded = processor.tokenizer.decode(ids.tolist())
    record = {'row_id': row['id'], 'source_size_wh': [width, height], 'image_grid_thw': grid[0].tolist(), 'image_pad_tokens': pads,
              'visual_tokens_from_grid': merged, 'visual_tokens_independent': etokens, 'vision_start': starts, 'vision_end': ends,
              'prompt_length': int(ids.shape[0]), 'attention_all_ones': bool(out['attention_mask'].min() == 1),
              'pixel_shape': list(pixels.shape), 'pixel_dtype': str(pixels.dtype), 'pixel_finite': bool(np.isfinite(pixels).all()),
              'pixel_std': float(pixels.std()), 'pixel_sha256': sha256_array(pixels), 'input_ids_sha256': sha256_array(ids.astype(np.int64)),
              'grid_matches_independent': grid[0].tolist() == [1, eh // PATCH, ew // PATCH],
              'leak_hits': leak_hits(prompt, forbidden) + leak_hits(decoded, forbidden)}
    record['ok'] = (pads == merged == etokens and starts == ends == 1 and record['attention_all_ones'] and record['pixel_finite'] and
                    pixels.shape == (int(grid[0].prod()), 3 * TPS * PATCH * PATCH) and record['pixel_std'] > 0 and
                    record['grid_matches_independent'] and not record['leak_hits'])
    return record, ids, pixels


def qualify_rows(processor, rows, capsule_rows, *, system_instruction, assistant_prefix, enable_thinking, model_config, formatter=None):
    """Stream rows one image at a time. ``capsule_rows``: id -> {'forbidden': {...}, 'pair': pair id}. Counts-only summary."""
    formatter = formatter or load_formatter()
    per_row, prompts, by_pair, held = [], set(), {}, {}
    for row in rows:
        entry = capsule_rows[row['id']]
        require(len(row['images']) == 1, 'Exactly one image per row')
        prompt = render(formatter, processor, model_config, system_instruction, row['extra_info']['question'], enable_thinking=enable_thinking)
        require(prompt.endswith(assistant_prefix) and prompt.count(IMAGE_PAD) == 1, 'Native prompt placeholder/prefix differs')
        prompts.add(hashlib.sha256(prompt.encode()).hexdigest())
        record, ids, pixels = _row_record(processor, row, prompt, load_rgb(row['images'][0]), entry.get('forbidden', {}))
        record['pair'] = entry['pair']
        per_row.append(record)
        by_pair.setdefault(entry['pair'], []).append((row, prompt, record))
        held[row['id']] = (prompt, record['pixel_sha256'], record['input_ids_sha256'])
    swap_pixel, swap_text, swapped_pairs = 0, 0, 0
    for pair, members in by_pair.items():
        if len(members) != 2:
            continue
        (ra, pa, rec_a), (rb, pb, rec_b) = members
        # genuine swap: image of A under the prompt of B and vice versa
        swapped_a, ids_a, px_a = _row_record(processor, ra, pb, load_rgb(ra['images'][0]), {})
        swapped_b, ids_b, px_b = _row_record(processor, rb, pa, load_rgb(rb['images'][0]), {})
        swapped_pairs += 1
        swap_pixel += int(swapped_a['pixel_sha256'] == rec_a['pixel_sha256'] and swapped_b['pixel_sha256'] == rec_b['pixel_sha256'] and
                          swapped_a['pixel_sha256'] != swapped_b['pixel_sha256'])
        swap_text += int(swapped_a['input_ids_sha256'] == rec_b['input_ids_sha256'] and swapped_b['input_ids_sha256'] == rec_a['input_ids_sha256'])
        del ids_a, ids_b, px_a, px_b
    ok = all(r['ok'] for r in per_row) and len(prompts) == 1 and swap_pixel == swapped_pairs == swap_text
    return {'ok': bool(ok), 'rows': len(per_row), 'prompt_text_variants': len(prompts), 'swap_pairs': swapped_pairs,
            'swap_pixel_changes': swap_pixel, 'swap_text_unchanged': swap_text, 'per_row': per_row}


def aggregate(per_row, *, expected_rows):
    require(len(per_row) == expected_rows, f'Exact {expected_rows}-row inventory required; got {len(per_row)}')
    return {'rows': len(per_row), 'rows_ok': sum(bool(r.get('ok')) for r in per_row)}
