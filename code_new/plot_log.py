"""Plot a run's log.csv: accuracy/recall, dropout/penalty and model size per pass.

    python code_new/plot_log.py training-runs/surgical_gpu          # writes training-runs/surgical_gpu/log.png
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402


def col(rows, name):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(name, 'nan')))
        except ValueError:
            out.append(float('nan'))
    return out


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    rows = list(csv.DictReader(open(run / 'log.csv', encoding='utf-8')))
    if not rows:
        sys.exit('log.csv is empty')
    x = col(rows, 'pass')
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(x, col(rows, 'fold_acc'), label='accuracy (next fold)')
    ax[0].plot(x, col(rows, 'fold_recall'), label='recall (next fold)')
    ax[0].plot(x, col(rows, 'test_acc'), '--', label='accuracy (test patients)')
    ax[0].plot(x, col(rows, 'test_recall'), '--', label='recall (test patients)')
    ax[0].axhline(0.90, color='grey', lw=0.6); ax[0].axhline(0.95, color='grey', lw=0.6)
    ax[0].set_ylim(0, 1.02); ax[0].legend(loc='lower right', fontsize=8); ax[0].set_ylabel('metric')
    ax[1].plot(x, col(rows, 'dropout'), label='dropout')
    ax[1].plot(x, col(rows, 'penalty'), label='penalty λ')
    ax[1].plot(x, col(rows, 'sel_norm_ratio'), ':', label='|w| selected / kept')
    ax[1].legend(fontsize=8); ax[1].set_ylabel('schedule')
    ax[2].plot(x, [v * 100 for v in col(rows, 'param_ratio')], color='k')
    ax[2].set_ylabel('params, % of start'); ax[2].set_xlabel('pass')
    for r, xi in zip(rows, x):
        a = r.get('action')
        if a in ('cut', 'accept', 'fail'):
            c = {'cut': 'red', 'accept': 'green', 'fail': 'orange'}[a]
            for axi in ax:
                axi.axvline(xi, color=c, lw=0.7, alpha=0.6)
    fig.suptitle(f"{run.name}: green = accepted, red = cut, orange = recovery failed")
    fig.tight_layout()
    out = run / 'log.png'
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == '__main__':
    main()
