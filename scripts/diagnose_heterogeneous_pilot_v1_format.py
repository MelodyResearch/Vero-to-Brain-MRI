#!/usr/bin/env python3
"""CPU-only bounded diagnosis of the v1 heterogeneous pilot gate (all 6 greedy answers format 0.5, no \\boxed). Counts-only stdout.

Reads ONLY saved v1 artifacts (gate records, pilot input, scope) plus the pinned contract/processor/tokenizer FILES through the
existing torch-free native_processor_qualification_v1 helper. No MLX import, no weights, no model call, no replay of the 6 calls.
  * re-renders each delivered prompt and checks it against the saved prompt SHA / prompt length;
  * checks the saved input fingerprints (SWAP pixels == other TRUE pixels; BORDER pixels differ, input_ids equal);
  * looks for any output cleaning on the pinned processor (clean_output) and runs a tokenizer round-trip counterexample showing a
    '\\boxed{yes}' answer survives encode/decode;
  * re-encodes each saved response text and compares with the saved token count (EOS-inclusive by capture_stream's invariant).
The full private proof goes to --private-out; a sanitized summary to --safe-out.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mri_rl import heterogeneous_mri_pilot_v1 as v1                      # noqa: E402
from mri_rl import heterogeneous_mri_pilot_entry_v1 as v1entry           # noqa: E402
from mri_rl import native_processor_qualification_v1 as q                # noqa: E402
from mri_rl.vero_mri_reward import _answer_and_format                   # noqa: E402

RUN = v1entry.PILOT_DIR / 'technical_smoke_attempt1_PRIVATE'
GATE = RUN / 'parent_PRIVATE' / 'parent_phase_PRIVATE' / 'gate_PRIVATE.json'
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-out', type=Path, required=True)
    p.add_argument('--safe-out', type=Path, required=True)
    a = p.parse_args()
    gate = json.loads(GATE.read_text())
    pinput = json.loads((RUN / 'pilot_input_PRIVATE.json').read_text())
    v2 = v1entry.load(v1entry.V2_SPEC); runtime = v1entry.load(v2['runtime_spec']); contract = v1entry.load(runtime['contract'])
    model = Path(runtime['model']); config = json.loads((model / 'config.json').read_text())
    processor, formatter = q.load_pinned_processor(model), q.load_formatter()
    tok = processor.tokenizer
    controls = {c['name']: c for c in v1.gate_controls(pinput['midline_train_PRIVATE'], pinput['border_controls_PRIVATE'])}
    rows, private = [], {'controls': []}
    for rec in gate['controls']:
        c = controls[rec['name']]
        prompt = q.render(formatter, processor, config, contract['system_instruction'], c['row']['extra_info']['question'],
                          enable_thinking=contract['enable_thinking'])
        text = rec['text']
        answer, fmt = _answer_and_format(text)
        reenc = tok.encode(text, add_special_tokens=False)
        rows.append({'name': rec['name'], 'kind': rec['kind'],
                     'prompt_sha256_matches_saved': hashlib.sha256(prompt.encode()).hexdigest() == rec['actual_PRIVATE']['prompt_sha256'],
                     'prompt_ends_with_assistant_prefix': prompt.endswith(contract['assistant_prefix']),
                     'saved_prompt_length': rec['actual_PRIVATE']['prompt_length'], 'system_in_prompt': contract['system_instruction'] in prompt,
                     'original_question_in_prompt': c['row']['extra_info']['original_question'] in prompt, 'v1_suffix_in_prompt': v1.MIDLINE_SUFFIX in prompt,
                     'saved_tokens_incl_EOS': rec['tokens'], 'reencoded_text_tokens': len(reenc), 'text_chars': len(text),
                     'think_tags': [text.count('<think>'), text.count('</think>')], 'answer_tags': [text.count('<answer>'), text.count('</answer>')],
                     'boxed_count': text.count('\\boxed'), 'parser_format': fmt, 'tag_answer_label': answer.strip().lower(), 'image_label': rec['image_label'],
                     'F32_LM_steps_saved': len(rec['actual_PRIVATE']['F32_evidence_PRIVATE'])})
        private['controls'].append({**rows[-1], 'prompt_PRIVATE': prompt, 'text_PRIVATE': text, 'reencoded_ids_PRIVATE': reenc})
    fp = {r['name']: r['actual_PRIVATE']['input_fingerprints_PRIVATE'] for r in gate['controls']}
    same = lambda x, y, k: json.dumps(fp[x][k], sort_keys=True) == json.dumps(fp[y][k], sort_keys=True)
    pixel = {'SWAP_0_pixels_eq_TRUE_1': same('SWAP_0', 'TRUE_1', 'pixel_values'), 'SWAP_1_pixels_eq_TRUE_0': same('SWAP_1', 'TRUE_0', 'pixel_values'),
             'BORDER_0_pixels_differ_TRUE_0': not same('BORDER_0', 'TRUE_0', 'pixel_values'), 'BORDER_1_pixels_differ_TRUE_1': not same('BORDER_1', 'TRUE_1', 'pixel_values'),
             'BORDER_0_input_ids_eq_TRUE_0': same('BORDER_0', 'TRUE_0', 'input_ids'), 'BORDER_1_input_ids_eq_TRUE_1': same('BORDER_1', 'TRUE_1', 'input_ids'),
             'TRUE_0_pixels_differ_TRUE_1': not same('TRUE_0', 'TRUE_1', 'pixel_values')}
    probe = '<think>The structures are midline.</think><answer>\\boxed{no}</answer>'
    ids = tok.encode(probe, add_special_tokens=False)
    counter = {'probe_roundtrip_identical': tok.decode(ids) == probe, 'probe_tokens': len(ids),
               'probe_parser_format': _answer_and_format(probe)[1], 'pinned_processor_has_clean_output': callable(getattr(processor, 'clean_output', None))}
    safe = {'schema': 'heterogeneous-mri-pilot-v1-format-diagnosis-v1', 'source': 'saved v1 artifacts + pinned processor/tokenizer files (CPU)', 'controls': [
                {k: v for k, v in r.items() if k not in ('tag_answer_label', 'image_label')} for r in rows],
            'greedy_swap_decision_changes': sum(rows[i]['tag_answer_label'] != rows[i + 2]['tag_answer_label'] for i in (0, 1)),
            'pixel_identity': pixel, 'cleaning_counterexample': counter,
            'contract': {'enable_thinking': contract['enable_thinking'], 'max_tokens': contract['generation']['max_tokens'],
                         'temperature': contract['generation']['temperature'], 'system_instruction_sha256': hashlib.sha256(contract['system_instruction'].encode()).hexdigest(),
                         'system_mentions_boxed': 'boxed' in contract['system_instruction'], 'v1_suffix_mentions_boxed': 'boxed' in v1.MIDLINE_SUFFIX},
            'saved_v1_evidence_gaps': ['response token IDs not persisted (count only)', 'finish_reason / EOS id not persisted',
                                       'no per-token logprobs or canonical yes/no margins persisted', 'pre-clean text/cleaner invocation not explicitly persisted'],
            'inputs_sha256': {'gate_record': H(GATE), 'pilot_input': H(RUN / 'pilot_input_PRIVATE.json'), 'contract': H(runtime['contract'])},
            'model_calls': 0, 'mlx_imported': 'mlx' in sys.modules, 'torch_imported': 'torch' in sys.modules}
    v1entry.require(not safe['mlx_imported'], 'MLX was imported')
    for path, value in ((a.private_out, {**safe, **private}), (a.safe_out, safe)):
        with Path(path).open('x') as f:
            json.dump(value, f, indent=1, sort_keys=True)
    print(json.dumps({'controls': [{k: r[k] for k in ('name', 'prompt_sha256_matches_saved', 'saved_tokens_incl_EOS', 'reencoded_text_tokens', 'boxed_count', 'parser_format')}
                                   for r in rows], 'pixel_identity': pixel, 'cleaning_counterexample': counter}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
