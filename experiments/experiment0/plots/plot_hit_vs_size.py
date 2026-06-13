"""RL-Cache: learned admission probability and hit rate vs object size.

Directly visualizes the size threshold the OHR-reward model learns — admit
probability falls from ~1 to ~0 as object size grows, and hit rate follows.
Computed via the validated standalone simulator from a trained model.

Usage:
    python experiments/experiment0/plots/plot_hit_vs_size.py
    python experiments/experiment0/plots/plot_hit_vs_size.py --model model_bhr.pt --out results_size_bhr.png
"""

# Author: labry

import os
import sys
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, '/home/labry/git/IcarusGym')
sys.path.insert(0, REPO_ROOT)

import torch
from matplotlib import pyplot

import experiments.experiment0.config as conf
import experiments.experiment0.icarus_config as icarus_conf
from rl_cache import features as rlfeatures
from rl_cache.rl_cache_policy import RLCacheNetwork
from rl_cache.training import ttl_sim

TRACE_PATH = os.path.join(EXP_DIR, 'trace_seed42.npz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=conf.MODEL_PATH)
    ap.add_argument('--out', default='results_rl_cache_vs_size.png')
    ap.add_argument('--bins', type=int, default=18)
    args = ap.parse_args()

    data = np.load(TRACE_PATH)
    trace = list(zip(data['times'].tolist(), data['contents'].tolist(), data['sizes'].tolist()))
    sizes = np.array([s for _, _, s in trace])
    feats = rlfeatures.extract_all(trace, conf.N_CONTENTS)

    ckpt = torch.load(os.path.join(EXP_DIR, args.model), map_location='cpu', weights_only=False)
    net = RLCacheNetwork(rlfeatures.FEATURE_DIM, conf.HIDDEN_DIM, conf.NUM_LAYERS)
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    with torch.no_grad():
        p_admit = net(torch.from_numpy(feats)).squeeze(-1).numpy()   # per request
    greedy = (p_admit >= 0.5).astype(np.int64)

    size_aware = icarus_conf.SIZE_MIN < icarus_conf.SIZE_MAX
    res = ttl_sim.simulate(trace, decisions=greedy, cache_size=int(conf.B_0),
                           admit_ttl=conf.ADMIT_TTL, reject_ttl=conf.REJECT_TTL,
                           is_reset=icarus_conf.IS_RESET, size_aware=size_aware,
                           default_size=icarus_conf.DEFAULT_SIZE)
    hits = np.array(res['hits'])

    # Bin requests by object size (log-spaced).
    edges = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), args.bins + 1)
    centers, mean_padmit, hit_rate = [], [], []
    for i in range(args.bins):
        m = (sizes >= edges[i]) & (sizes < edges[i + 1] if i < args.bins - 1 else sizes <= edges[i + 1])
        if m.sum() == 0:
            continue
        centers.append(np.sqrt(edges[i] * edges[i + 1]))
        mean_padmit.append(p_admit[m].mean())
        hit_rate.append(hits[m].mean())

    pyplot.figure(figsize=(10, 6))
    pyplot.plot(centers, mean_padmit, 'o-', color='#1f77b4', label='mean P(admit)  [learned threshold]')
    pyplot.plot(centers, hit_rate, 's-', color='#d62728', label='hit rate (per request)')
    pyplot.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='admit threshold 0.5')
    pyplot.xscale('log')
    pyplot.xlabel('Object size (bytes)')
    pyplot.ylabel('Probability / rate')
    pyplot.ylim(-0.03, 1.03)
    pyplot.title(f'RL-Cache: admission and hit rate vs object size  '
                 f'(OHR={res["hit_rate"]:.3f}, BHR={res["byte_hit_rate"]:.3f})')
    pyplot.legend()
    pyplot.grid(True, which='both', alpha=0.3)
    pyplot.tight_layout()
    out = os.path.join(REPO_ROOT, args.out)
    pyplot.savefig(out, dpi=150)
    print(f'Saved {out}  (OHR={res["hit_rate"]:.4f}, BHR={res["byte_hit_rate"]:.4f})')


if __name__ == '__main__':
    main()
