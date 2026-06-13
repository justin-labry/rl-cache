# RL-Cache: OHR vs BHR across cache sizes (grouped bar chart, paper Fig.4 style).
# Data: experiments/experiment0/sweep_results.json (from sweep_cache_size.py).
#
# Usage:
#   python experiments/experiment0/plots/plot_sweep.py

# Author: labry

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

from matplotlib import pyplot

JSON_PATH = os.path.join(EXP_DIR, 'sweep_results.json')
OUT_PATH = os.path.join(REPO_ROOT, 'results_ohr_bhr_sweep.png')

# Fixed policy order + colors (consistent across both panels).
POLICIES = ['AdmitAll', 'SecondHit', 'RL-Cache(OHR)', 'RL-Cache(BHR)']
COLORS = {'AdmitAll': '#7f7f7f', 'SecondHit': '#ff7f0e',
          'RL-Cache(OHR)': '#1f77b4', 'RL-Cache(BHR)': '#d62728'}


def main():
    if not os.path.exists(JSON_PATH):
        print(f'Error: {JSON_PATH} not found. Run sweep_cache_size.py first.')
        sys.exit(1)
    rows = json.load(open(JSON_PATH))

    cache_sizes = sorted({r['cache_size'] for r in rows})
    # index: metrics[policy][metric] -> list aligned to cache_sizes
    def val(policy, metric):
        out = []
        for cs in cache_sizes:
            hit = [r for r in rows if r['cache_size'] == cs and r['policy'] == policy]
            out.append(hit[0][metric] if hit else 0.0)
        return out

    fig, axes = pyplot.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(cache_sizes))
    width = 0.2

    for ax, metric, title in [(axes[0], 'ohr', 'Object Hit Rate (OHR)'),
                              (axes[1], 'bhr', 'Byte Hit Rate (BHR)')]:
        for i, pol in enumerate(POLICIES):
            ys = val(pol, metric)
            ax.bar(x + (i - 1.5) * width, ys, width, label=pol, color=COLORS[pol])
        ax.set_xticks(x)
        ax.set_xticklabels([f'{cs}\n(ratio {cs/1000:.2f})' for cs in cache_sizes])
        ax.set_xlabel('Cache size (objects)')
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)

    axes[0].legend(loc='upper left', fontsize=9)
    fig.suptitle('RL-Cache vs baselines on IcarusGym: OHR optimized vs BHR optimized', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
