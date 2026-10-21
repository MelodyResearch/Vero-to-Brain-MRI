"""CPU-only measurement of the concise-prompt token delta (tokenizer file only; no MLX, no model weights, no generation).

For every one of the 32 frozen sequence rows (8 TRAIN schedule + 24 DEV24) the delta is measured twice: on the question
segment alone and inside the chat-turn special-token boundary. Both must agree and be identical across rows; the frozen
concise length is 213 + delta. The native path re-checks the ACTUAL expanded prompt length and stops on any mismatch.
Prints counts/hashes only; never prints row ids, questions or paths.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mri_rl import accuracy_pilot_v2_core as core
from mri_rl.shared_rollout_gspo_core_v1 import publish_once, file_sha256


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery-spec', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    from tokenizers import Tokenizer
    rspec = json.loads(args.recovery_spec.read_text())
    runtime, micro, dev = (json.loads(Path(rspec[k]).read_text()) for k in ('runtime_spec', 'micro_input', 'dev24'))
    stage = json.loads(Path(micro['source_stage_input_PRIVATE']).read_text())
    core.require(file_sha256(micro['source_stage_input_PRIVATE']) == micro['source_stage_input_sha256'], 'Frozen schedule changed')
    tokenizer_path = Path(runtime['model']) / 'tokenizer.json'
    tok = Tokenizer.from_file(str(tokenizer_path))
    n = lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    rows = [c['row'] for u in stage['units_PRIVATE'] if u['kind'] == 'SEQ_PAIR' for c in u['contexts']] + [r['row_PRIVATE'] for r in dev['records_PRIVATE']]
    core.require(len(rows) == 32 and all(r['ability'] == 'mri_sequence' for r in rows), 'Exactly 32 sequence rows required')
    deltas, wrapped = {}, {}
    for row in rows:
        q, c = row['extra_info']['question'], core.condition_row(row, 'concise')['extra_info']['question']
        deltas[row['id']] = n(c) - n(q)
        wrapped[row['id']] = n('<|im_start|>user\n' + c + '<|im_end|>') - n('<|im_start|>user\n' + q + '<|im_end|>')
    core.require(deltas == wrapped and len(set(deltas.values())) == 1, 'Segment/boundary deltas must agree and be uniform')
    delta = next(iter(deltas.values()))
    record = {'schema': 'accuracy-pilot-v2-concise-length-v1', 'instruction_sha256': core.CONCISE_INSTRUCTION_SHA256, 'row_ids': sorted(deltas),
              'deltas': deltas, 'boundary_deltas': wrapped, 'concise_sequence_prompt_length': core.ORIGINAL_PROMPT_LENGTHS['mri_sequence'] + delta,
              'tokenizer_sha256': file_sha256(tokenizer_path), 'method': 'tokenizers CPU only; native path re-verifies actual expanded length', 'model_calls': 0}
    publish_once(args.output, record)
    print(json.dumps({'rows': len(rows), 'delta': delta, 'concise_sequence_prompt_length': record['concise_sequence_prompt_length'], 'file_sha256': file_sha256(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
