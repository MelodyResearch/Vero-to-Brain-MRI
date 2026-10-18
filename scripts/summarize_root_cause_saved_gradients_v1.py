"""Recount scalar gradient norms from authenticated TRAIN receipts, not tensors."""
import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-directory', type=Path, required=True)
    p.add_argument('--inventory-sha256', required=True)
    p.add_argument('--ledger-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    inv = a.replay_directory/'read_inventory_PRIVATE.json'
    ledger = a.replay_directory/'group_causes_PRIVATE.jsonl'
    assert sha(inv) == a.inventory_sha256 and sha(ledger) == a.ledger_sha256
    paths = {}
    for name, expected in json.loads(inv.read_text()).items():
        f = Path(name)
        if f.name == 'receipt.json' and f.parent.name.startswith('boundary_'):
            assert sha(f) == expected
            paths[expected] = f
    groups = collections.defaultdict(list)
    accepted = 0
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        if not row['accepted']:
            continue
        r = json.loads(paths[row['receipt_sha256']].read_text())
        assert r['updated'] is True
        value = r['update_evidence']['gradient_l2']
        assert type(value) in (int, float) and math.isfinite(value) and value > 0
        groups[(row['arm'], row['seed'], row['task'])].append(value)
        accepted += 1
    assert accepted == 362
    rows = [{'arm': arm, 'seed': seed, 'task': task, 'updates': len(v),
             'gradient_l2_min': min(v), 'gradient_l2_median': statistics.median(v),
             'gradient_l2_max': max(v)} for (arm, seed, task), v in sorted(groups.items())]
    result = {'schema': 'root-cause-saved-gradient-norms-v1', 'status': 'REPRODUCED_CPU_METADATA',
              'accepted_updates_with_positive_finite_saved_gradient_norm': accepted,
              'arm_seed_task_rows': rows, 'read_inventory_sha256': a.inventory_sha256,
              'group_ledger_sha256': a.ledger_sha256, 'source_sha256': sha(__file__),
              'limits': ['Saved scalar norms only; no vector gradient, parameter or Adam tensor reread.',
                         'Norms do not establish net per-task direction, visual dependence, causal task contribution or learning.',
                         'No new gradient, optimizer update, model or final100 read.'],
              'model_loads': 0, 'optimizer_updates_performed': 0}
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps({'updates': accepted, 'aggregate_sha256': sha(a.output)}))

if __name__ == '__main__':
    main()
