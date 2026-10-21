"""CPU processor-qualification tests. Synthetic noise images only; no MRI, weights or MLX."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from PIL import Image
from mri_rl import native_processor_qualification_v1 as q

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parents[1] / 'local_data/models/qwen3vl8b-base-recovered-v1'
needs_model = pytest.mark.skipif(not (MODEL / 'tokenizer.json').is_file(), reason='pinned tokenizer/processor files absent')


def png(tmp_path, name, seed, size=(364, 436)):
    rng = np.random.default_rng(seed)
    path = tmp_path / name
    Image.fromarray((rng.random((size[1], size[0], 3)) * 255).astype('uint8')).save(path)
    return path


# ------------------------------------------------------------------ pure checks
def test_smart_resize_matches_pinned_formula_for_364x436():
    # 364x436 -> rounded to 32: 352 x 448 ; patch16 -> 22 x 28 grid ; merge2 -> 154 visual tokens
    assert q.expected_visual_tokens(436, 364) == (448, 352, 154)


def test_leak_scan_flags_filename_patient_site_and_label_and_ignores_generic_words():
    forbidden = {'patient_key': 'sub-0123abcd', 'file_stem': 'z045_flair_sequence', 'site': 'KA'}
    clean = 'What MRI sequence is this image? Answer with FLAIR or contrast-enhanced T1.'
    assert q.leak_hits(clean, forbidden) == []
    assert q.leak_hits('scan sub-0123abcd', forbidden) == ['patient_key']
    assert q.leak_hits('file Z045_FLAIR_sequence', forbidden) == ['file_stem']
    assert q.leak_hits('site KA here', forbidden) == ['site']
    assert q.leak_hits('KAnsas is not a site code', forbidden) == []


def test_join_gap_reports_specific_missing_keys_and_never_invents():
    rows = [{'id': 'a', 'images': ['/x.png']}]
    gap = q.join_capsule(rows, {'rows': {'a': {'expected_png_sha256': 'f' * 64, 'pair_key': 'p'}}})
    assert gap['ok'] is False and gap['missing_keys']['a'] == ['split']


def test_aggregate_refuses_partial_inventory():
    with pytest.raises(ValueError):
        q.aggregate([{'ok': True}] * 55, expected_rows=56)


# ------------------------------------------------------------------ real pinned processor, CPU, synthetic noise
@needs_model
def test_processor_expands_image_pads_to_grid_and_pixels_follow_the_image(tmp_path):
    proc = q.load_pinned_processor(MODEL)
    a, b = png(tmp_path, 'a.png', 1), png(tmp_path, 'b.png', 2)
    prompt = '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Which?<|im_end|>\n<|im_start|>assistant\n'
    ra = q.run_processor(proc, prompt, q.load_rgb(a))
    rb = q.run_processor(proc, prompt, q.load_rgb(b))
    grid = ra['image_grid_thw'][0].tolist()
    assert grid == [1, 28, 22] and ra['pixel_values'].shape == (28 * 22, 1536) and ra['pixel_values'].dtype == np.float32
    assert int((ra['input_ids'] == proc.image_token_id).sum()) == 154
    assert np.array_equal(ra['input_ids'], rb['input_ids'])          # text stream is image independent
    assert not np.array_equal(ra['pixel_values'], rb['pixel_values'])  # pixels follow the image
    assert np.isfinite(ra['pixel_values']).all()


@needs_model
def test_prompt_with_wrong_placeholder_count_is_rejected(tmp_path):
    proc = q.load_pinned_processor(MODEL)
    with pytest.raises(ValueError):
        q.run_processor(proc, 'no image placeholder here', q.load_rgb(png(tmp_path, 'a.png', 3)))


@needs_model
def test_qualify_rows_end_to_end_synthetic_pairs(tmp_path):
    proc = q.load_pinned_processor(MODEL)
    rows, capsule = [], {}
    system, question = 'SYSTEM', 'Which MRI sequence is shown?'
    for i, (patient, mod) in enumerate((('p1', 'FLAIR'), ('p1', 'T1'), ('p2', 'FLAIR'), ('p2', 'T1'))):
        p = png(tmp_path, f'opaque{i}.png', 10 + i)
        rows.append({'id': f'r{i}', 'images': [str(p)], 'extra_info': {'question': question, 'patient_key': patient},
                     'reward_model': {'ground_truth': mod}})
        capsule[f'r{i}'] = {'forbidden': {'patient_key': patient}, 'pair': patient}
    result = q.qualify_rows(proc, rows, capsule, system_instruction=system, assistant_prefix='<|im_start|>assistant\n',
                            enable_thinking=False, model_config=json.loads((MODEL / 'config.json').read_text()))
    assert result['ok'] and result['rows'] == 4
    assert result['prompt_text_variants'] == 1 and result['swap_pixel_changes'] == 2 and result['swap_text_unchanged'] == 2
    assert all(r['image_pad_tokens'] == 154 for r in result['per_row'])
