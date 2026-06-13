"""Generate a per-content evaluation NPZ for a policy using the validated
standalone simulator, in the exact format plot_results.py expects.

The simulator reproduces IcarusGym's per-request hit/miss (validated by
validate_sim.py to ~0.003), so this is a fast, fully deterministic way to
produce evaluation figures for the trained model without a long IcarusGym
rollout. For IcarusGym-native numbers use `main.py --mode test` instead.

Writes <out>.npz to the repo root (where plot_results.py reads it) with the
same keys the RLCacheCallbacks produce: is_, hit_probs, request_counts,
hit_counts, mean_cache_hit_ratio.

Usage:
    source .venv/bin/activate
    python experiments/experiment0/eval_sim.py                 # RL-Cache (model.pt)
    python experiments/experiment0/eval_sim.py --policy admitall --out results_admitall
    python experiments/experiment0/eval_sim.py --policy secondhit --out results_secondhit
    python experiments/experiment0/eval_sim.py --model model_bhr.pt --out results_rl_cache_bhr
"""

# Author: labry

import sys
import os
import argparse

sys.path.insert(0, '/home/labry/git/IcarusGym')
sys.path.insert(0, '/home/labry/git/rl-cache')

import numpy as np
import torch

import experiments.experiment0.config as conf
import experiments.experiment0.icarus_config as icarus_conf
from rl_cache import features as rlfeatures
from rl_cache import baselines
from rl_cache.rl_cache_policy import RLCacheNetwork
from rl_cache.training import ttl_sim

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP_DIR, '..', '..'))
TRACE_PATH = os.path.join(EXP_DIR, 'trace_seed42.npz')


def greedy_from_model(model_path, feats):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    net = RLCacheNetwork(rlfeatures.FEATURE_DIM, conf.HIDDEN_DIM, conf.NUM_LAYERS)
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    with torch.no_grad():
        p = net(torch.from_numpy(feats)).squeeze(-1).numpy()
    return (p >= 0.5).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description='Per-content eval NPZ via the validated simulator')
    ap.add_argument('--policy', choices=['rlcache', 'admitall', 'secondhit'], default='rlcache')
    ap.add_argument('--model', default=conf.MODEL_PATH, help='model file (for --policy rlcache)')
    ap.add_argument('--out', default=conf.RESULT_OUTPUT_FILE_NAME, help='output NPZ basename (no extension)')
    ap.add_argument('--cache-ratio', type=float, default=conf.CACHE_RATIO)
    args = ap.parse_args()

    if not os.path.exists(TRACE_PATH):
        sys.exit(f'No trace at {TRACE_PATH}. Run: python experiments/experiment0/extract_trace.py')
    data = np.load(TRACE_PATH)
    trace = list(zip(data['times'].tolist(), data['contents'].tolist(), data['sizes'].tolist()))
    n_contents = conf.N_CONTENTS
    cache_size = max(1, int(n_contents * args.cache_ratio))
    feats = rlfeatures.extract_all(trace, n_contents)

    if args.policy == 'rlcache':
        model_path = os.path.join(EXP_DIR, args.model)
        if not os.path.exists(model_path):
            sys.exit(f'No model at {model_path}. Train first with train_mc.py')
        decisions = greedy_from_model(model_path, feats)
        label_src = f'model={args.model}'
    elif args.policy == 'admitall':
        decisions = None
        label_src = 'AdmitAll'
    else:  # secondhit
        decisions = baselines.secondhit_decisions(trace)
        label_src = 'SecondHit'

    size_aware = icarus_conf.SIZE_MIN < icarus_conf.SIZE_MAX
    res = ttl_sim.simulate(
        trace, decisions=decisions, cache_size=cache_size,
        admit_ttl=conf.ADMIT_TTL, reject_ttl=conf.REJECT_TTL,
        is_reset=icarus_conf.IS_RESET, size_aware=size_aware,
        default_size=icarus_conf.DEFAULT_SIZE,
    )

    # Build per-content arrays (1-indexed contents 1..N), matching RLCacheCallbacks.
    req = np.zeros(n_contents + 1, dtype=np.int64)
    hit = np.zeros(n_contents + 1, dtype=np.int64)
    for c, v in res['request_counts'].items():
        req[c] = v
    for c, v in res['hit_counts'].items():
        hit[c] = v
    is_ = np.arange(1, n_contents + 1, dtype=np.int32)
    rc = req[1:n_contents + 1]
    hc = hit[1:n_contents + 1]
    # Per-content hit ratio (callback convention: divide by req-1; first touch is a cold miss).
    with np.errstate(divide='ignore', invalid='ignore'):
        hit_probs = np.where(rc > 1, np.minimum(1.0, hc / np.maximum(rc - 1, 1)), 0.0).astype(np.float64)
    mean_cache_hit_ratio = float(hit_probs.mean())

    out_path = os.path.join(REPO_ROOT, args.out + '.npz')
    np.savez(out_path, is_=is_, hit_probs=hit_probs,
             request_counts=rc, hit_counts=hc, mean_cache_hit_ratio=mean_cache_hit_ratio)

    print(f'Policy: {args.policy} ({label_src}), cache_size={cache_size} (ratio {args.cache_ratio})')
    print(f'  OHR (per-request hit rate): {res["hit_rate"]:.4f}')
    print(f'  BHR (byte hit rate):        {res["byte_hit_rate"]:.4f}')
    print(f'  per-content mean hit ratio: {mean_cache_hit_ratio:.4f}')
    print(f'Saved {out_path}')
    print(f'Now plot with: python experiments/experiment0/plots/plot_results.py')


if __name__ == '__main__':
    main()
