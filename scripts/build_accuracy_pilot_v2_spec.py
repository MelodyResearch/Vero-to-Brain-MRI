"""Build the private Accuracy Pilot V2 spec from the executed V1 recovery spec (metadata/hashes only; no model, no approval flag).

Then run `scripts/run_accuracy_pilot_v2.py --spec SPEC --scope-output SCOPE` to obtain the exact PREPARED scope for independent review.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mri_rl import accuracy_pilot_v2_entry as entry
from mri_rl.shared_rollout_gspo_core_v1 import publish_once, file_sha256


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('recovery-spec', 'recovery-scope', 'concise-length-measurement', 'output', 'spec-output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args(argv)
    rspec = json.loads(args.recovery_spec.read_text()); micro = json.loads(Path(rspec['micro_input']).read_text())
    measure = json.loads(args.concise_length_measurement.read_text())
    spec = {'schema': entry.SPEC_SCHEMA, 'recovery_spec': str(args.recovery_spec.resolve()), 'recovery_scope': str(args.recovery_scope.resolve()),
            'recovery_output': rspec['output'], 'stage_input': micro['source_stage_input_PRIVATE'], 'runtime_spec': rspec['runtime_spec'], 'dev24': rspec['dev24'],
            'concise_length_measurement': str(args.concise_length_measurement.resolve()), 'concise_sequence_prompt_length': measure['concise_sequence_prompt_length'],
            'output': str(entry._output(args.output.resolve()))}
    spec['file_sha256'] = {k: file_sha256(spec[k]) for k in entry.SPEC_FILE_KEYS}
    publish_once(entry._output(args.spec_output.resolve()), spec)
    print(json.dumps({'spec_file_sha256': file_sha256(args.spec_output), 'file_sha256': spec['file_sha256'], 'approval_flags': 0, 'model_calls': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
