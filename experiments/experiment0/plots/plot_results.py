# RL-Cache: Per-content hit probability plot
# Legend shows per-request hit rate (the practical metric, = OHR).
# Data source: an evaluation NPZ (e.g. results_rl_cache.npz) produced by
#   experiments/experiment0/eval_sim.py   (fast, via the validated simulator)
#   or by main.py --mode test             (IcarusGym-native)
#
# Usage:
#   python experiments/experiment0/plots/plot_results.py
#   python experiments/experiment0/plots/plot_results.py --input results_rl_cache_bhr --label "RL-Cache (BHR)"

# Author: labry

import os
import sys
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, '/home/labry/git/IcarusGym')
sys.path.insert(0, REPO_ROOT)

from matplotlib import pyplot


def plot_results_rl_cache(input_basename='results_rl_cache', output_file=None, label=None):
    """Plot per-content cache hit probabilities from an evaluation NPZ.

    X-axis: Content ID (log), Y-axis: cache hit probability (log).
    Legend shows the per-request hit rate (OHR).
    """
    input_file = os.path.join(REPO_ROOT, input_basename + '.npz')
    if output_file is None:
        output_file = os.path.join(REPO_ROOT, input_basename + '.png')

    if not os.path.exists(input_file):
        print(f'Error: {input_file} not found.')
        print('Generate it first, e.g.: python experiments/experiment0/eval_sim.py')
        sys.exit(1)

    data = np.load(input_file)
    print(f'Loaded: {input_file}')

    hit_probs = np.array(data['hit_probs'], dtype=np.float64)
    request_counts = np.array(data['request_counts'], dtype=np.int64)
    hit_counts = np.array(data['hit_counts'], dtype=np.int64)
    per_content_mean = float(data['mean_cache_hit_ratio'])

    total_req = request_counts.sum()
    total_hit = hit_counts.sum()
    per_request = float(total_hit / total_req) if total_req > 0 else 0.0

    n = len(hit_probs)
    xs = range(1, n + 1)
    print(f'  Contents: {n} | per-content mean {per_content_mean:.4f} | per-request hit rate {per_request:.4f}')

    hit_probs_plot = np.maximum(hit_probs, 1e-10)
    legend_label = label or f'RL-Cache (hit rate = {per_request:.4f})'

    pyplot.figure(figsize=(10, 6))
    line, = pyplot.plot(xs, hit_probs_plot, color='blue', marker='.', markersize=3, linewidth=1)
    pyplot.legend([line], [legend_label])
    pyplot.xlabel('Content ID')
    pyplot.ylabel('Cache Hit Probability')
    pyplot.title('RL-Cache: Per-Content Cache Hit Probability')
    pyplot.xscale('log')
    pyplot.yscale('log')
    y_min = hit_probs_plot.min() * 0.5
    y_max = hit_probs_plot.max() * 2.0
    pyplot.ylim(max(y_min, 1e-4), min(y_max, 2.0))
    pyplot.grid(True, which='both', alpha=0.3)
    pyplot.tight_layout()
    pyplot.savefig(output_file, dpi=150)
    print(f'Plot saved to {output_file}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Per-content cache hit probability plot')
    ap.add_argument('--input', default='results_rl_cache', help='NPZ basename in repo root (no extension)')
    ap.add_argument('--output', default=None, help='output PNG path (default: <input>.png in repo root)')
    ap.add_argument('--label', default=None, help='legend label')
    args = ap.parse_args()
    plot_results_rl_cache(args.input, args.output, args.label)
