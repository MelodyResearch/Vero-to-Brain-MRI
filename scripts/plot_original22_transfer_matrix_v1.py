"""Plot authenticated aggregate cells only; no patient data or new statistics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--output-stem', type=Path, required=True)
    a = p.parse_args()
    assert hashlib.sha256(a.input.read_bytes()).hexdigest() == a.expected_sha256
    x = json.loads(a.input.read_text())
    assert x['status'] == 'ALL22_ACTUAL_OUTPUTS_AUTHENTICATED_AND_SCORED'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none'})
    order = [('B28', 7)] + [(arm, seed) for arm in (
        'S_sequence', 'S_correspondence', 'P_raw', 'U_raw',
        'P_calibrated', 'U_calibrated', 'same_C_SFT') for seed in (7, 11, 13)]
    tasks = ('mri_sequence', 'mri_visit_correspondence')
    cells = {(c['arm'], c['seed'], c['evaluation_task']): c for c in x['matrix']['cells']}
    assert len(cells) == 48 and len(order) == 22
    accuracy = np.array([[cells[(arm, seed, task)]['full_format_accuracy'] for task in tasks] for arm, seed in order])
    baseline = accuracy[0]
    delta = (accuracy - baseline) * 100
    labels = [('B28 (one common run)' if arm == 'B28' else f'{arm} | seed {seed}') for arm, seed in order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 11), sharey=True, gridspec_kw={'wspace': .32})
    im = axes[0].imshow(accuracy, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    dm = axes[1].imshow(delta, vmin=-25, vmax=25, cmap='RdBu', aspect='auto')
    for ax, values, is_delta in ((axes[0], accuracy, False), (axes[1], delta, True)):
        ax.set_xticks([0, 1], ['MRI sequence', 'Dallas identity'])
        ax.set_yticks(range(len(labels)), labels)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        for i in range(len(order)):
            for j in range(2):
                v = values[i, j]
                text = f'{v:+.2f}' if is_delta else f'{100*v:.2f}%'
                color = ('white' if abs(v)>14 else 'black') if is_delta else ('white' if v<.45 else 'black')
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
        for boundary in (0.5, 3.5, 6.5, 9.5, 12.5, 15.5, 18.5):
            ax.axhline(boundary, color='white', lw=1)
    axes[0].set_title('Strict full-format greedy accuracy', pad=35)
    axes[1].set_title('Change from common B28 (percentage points)', pad=35)
    fig.colorbar(im, ax=axes[0], fraction=.035, pad=.03)
    fig.colorbar(dm, ax=axes[1], fraction=.035, pad=.03)
    fig.suptitle('Frozen Vero-to-MRI final100: all 22 trained endpoints', y=.98, fontsize=14)
    fig.text(.02, .025, '84 sequence cases / 14 patients; 16 Dallas cases / 4 matched two-person cycles.\n'
             'Original prompt, greedy256, temperature0. One B28 run shown once; no adaptive/brief-prompt endpoints.\n'
             'Descriptive results; uncertainty and all 48 seed-indexed task cells are in the authenticated JSON.\n'
             'Dallas measures source identity, not clinical longitudinal change. No global cross-source independence claim.', fontsize=9)
    fig.subplots_adjust(left=.29, right=.94, top=.91, bottom=.115)
    a.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.svg', '.png'):
        dest = a.output_stem.with_suffix(suffix)
        assert not dest.exists(), 'Preserve existing figure rather than overwrite'
        fig.savefig(dest, dpi=180, metadata={'Description': 'Aggregate accuracy only; no MRI images or patient identifiers.'})
    plt.close(fig)
    print(json.dumps({'rows': 22, 'task_columns': 2, 'input_sha256': a.expected_sha256, 'new_statistics': False}))

if __name__ == '__main__':
    main()
