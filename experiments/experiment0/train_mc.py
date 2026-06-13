"""Train RL-Cache with Monte Carlo elite-sampling on the standalone simulator.

Pipeline:
  1. load the deterministic trace (run extract_trace.py first to create it),
  2. precompute RL-Cache Table-1 features for the whole trace,
  3. run the MC elite-sampling trainer (rl_cache.training.mc_trainer),
  4. save model.pt in a format loadable by RLCachePolicy.load_model (for IcarusGym eval).

Smoke test (fast sanity run on a trace prefix):
    python experiments/experiment0/train_mc.py --smoke

Fuller run:
    python experiments/experiment0/train_mc.py --K 1000 --m 200 --L 5000 --passes 2
"""

# Author: labry

import sys
import os
import argparse
import time

sys.path.insert(0, '/home/labry/git/IcarusGym')
sys.path.insert(0, '/home/labry/git/rl-cache')

import logging
import numpy as np
import torch

import experiments.experiment0.config as conf
import experiments.experiment0.icarus_config as icarus_conf
from rl_cache import features as rlfeatures
from rl_cache.rl_cache_policy import RLCacheNetwork
from rl_cache.training.mc_trainer import MCTrainer

TRACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trace_seed42.npz')


def load_trace(limit=None):
    if not os.path.exists(TRACE_PATH):
        sys.exit(f'Trace not found: {TRACE_PATH}\nRun: python experiments/experiment0/extract_trace.py')
    data = np.load(TRACE_PATH)
    times, contents, sizes = data['times'], data['contents'], data['sizes']
    if limit:
        times, contents, sizes = times[:limit], contents[:limit], sizes[:limit]
    trace = list(zip(times.tolist(), contents.tolist(), sizes.tolist()))
    return trace


def save_model(net, optimizer, path, n_contents, b_0, admit_ttl, reject_ttl, total_steps,
               best_hr=float('nan')):
    """Save a checkpoint loadable by RLCachePolicy.load_model (IcarusGym eval)."""
    n = int(n_contents) + 1     # policy uses 1-indexed catalog (n = catalog + 1)
    checkpoint = {
        'best_hr': float(best_hr),
        'feature_dim': rlfeatures.FEATURE_DIM,
        'hidden_dim': conf.HIDDEN_DIM,
        'num_layers': conf.NUM_LAYERS,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Per-content stats are recomputed online at eval; store zeros for compatibility.
        'access_counts': np.zeros(n, dtype=np.int64),
        'last_access_time': np.zeros(n, dtype=np.float64),
        'inter_arrival': np.zeros(n, dtype=np.float64),
        'arrival_nums': np.zeros(n, dtype=np.int64),
        'tau_sums': np.zeros(n, dtype=np.float64),
        'episode_count': 0,
        'total_steps': int(total_steps),
        'n': n, 'b_0': b_0, 'lr': conf.LR, 'gamma': conf.GAMMA,
        'admit_ttl': admit_ttl, 'reject_ttl': reject_ttl, 'max_lambda': conf.MAX_LAMBDA,
    }
    torch.save(checkpoint, path)
    print(f'Model saved to {path}')


def main():
    ap = argparse.ArgumentParser(description='RL-Cache MC elite-sampling training')
    ap.add_argument('--smoke', action='store_true', help='fast sanity run on a trace prefix')
    ap.add_argument('--limit', type=int, default=None, help='use only the first N requests')
    ap.add_argument('--K', type=int, default=1000)
    ap.add_argument('--m', type=int, default=200)
    ap.add_argument('--L', type=int, default=5000)
    ap.add_argument('--p', type=float, default=20.0, help='elite percentile')
    ap.add_argument('--q', type=int, default=4, help='refill every q windows')
    ap.add_argument('--passes', type=int, default=1)
    ap.add_argument('--gamma', type=float, default=0.99997)
    ap.add_argument('--target-mode', type=str, default='advantage',
                    choices=['advantage', 'marginal'],
                    help='advantage = level-pinned selectivity (default); marginal = paper-literal BCE')
    ap.add_argument('--adv-gain', type=float, default=2.0, help='spread of advantage targets')
    ap.add_argument('--reward-mode', type=str, default='ohr', choices=['ohr', 'bhr'],
                    help='rank decision samples by object hit rate (ohr) or byte hit rate (bhr)')
    ap.add_argument('--seed', type=int, default=0, help='seed for net init + sampling (reproducibility)')
    ap.add_argument('--out', type=str, default=conf.MODEL_PATH)
    ap.add_argument('--force', action='store_true',
                    help='overwrite the saved model even if its best_hr is higher')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.smoke:
        args.limit = args.limit or 5000
        args.K, args.m, args.L, args.q, args.passes = 500, 40, 1000, 4, 1

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger('rl_cache.training.mc_trainer').setLevel(logging.INFO)

    trace = load_trace(args.limit)
    n_contents = conf.N_CONTENTS
    print(f'Trace: {len(trace)} requests, catalog {n_contents}')

    t0 = time.time()
    feats = rlfeatures.extract_all(trace, n_contents)
    print(f'Features: {feats.shape} computed in {time.time() - t0:.1f}s')

    # Cache config mirrors IcarusGym (validated by validate_sim.py).
    size_aware = icarus_conf.SIZE_MIN < icarus_conf.SIZE_MAX
    net = RLCacheNetwork(rlfeatures.FEATURE_DIM, conf.HIDDEN_DIM, conf.NUM_LAYERS)
    optimizer = torch.optim.Adam(net.parameters(), lr=conf.LR)

    trainer = MCTrainer(
        net, optimizer, trace, feats,
        cache_size=int(conf.B_0), admit_ttl=conf.ADMIT_TTL, reject_ttl=conf.REJECT_TTL,
        is_reset=icarus_conf.IS_RESET, size_aware=size_aware,
        default_size=icarus_conf.DEFAULT_SIZE,
        K=args.K, m=args.m, p_percentile=args.p, L=args.L, gamma=args.gamma, q=args.q,
        target_mode=args.target_mode, adv_gain=args.adv_gain, seed=args.seed,
        reward_mode=args.reward_mode,
    )

    print(f'\nMC training: K={args.K} m={args.m} L={args.L} p={args.p}% q={args.q} '
          f'passes={args.passes} target={args.target_mode} gain={args.adv_gain}')
    print('-' * 70)
    t0 = time.time()
    history = trainer.train(num_passes=args.passes)
    train_time = time.time() - t0

    if history:
        first = np.mean([h['committed_hit_rate'] for h in history[:max(1, len(history) // 4)]])
        last = np.mean([h['committed_hit_rate'] for h in history[-max(1, len(history) // 4):]])
        elite_last = np.mean([h['elite_score_per_req'] for h in history[-max(1, len(history) // 4):]])
        print('-' * 70)
        print(f'Windows trained: {len(history)}  in {train_time:.1f}s')
        print(f'Committed hit rate: first quarter {first:.4f} -> last quarter {last:.4f}')
        print(f'Elite score/req (last quarter): {elite_last:.4f}')
        print(f'Best full-trace greedy hit rate: {trainer.best_hr:.4f}')

    # Save the BEST checkpoint (highest full-trace greedy hit rate), not the last.
    if trainer.best_state is not None:
        net.load_state_dict(trainer.best_state)
        print(f'Loaded best checkpoint (hit rate {trainer.best_hr:.4f}) for saving.')

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)

    # Save-if-better: keep model.pt holding the global best across seeds/runs.
    if os.path.exists(out_path) and not args.force:
        try:
            prev = torch.load(out_path, map_location='cpu', weights_only=False)
            prev_hr = float(prev.get('best_hr', float('nan')))
        except Exception:
            prev_hr = float('nan')
        if prev_hr == prev_hr and prev_hr >= trainer.best_hr:   # not NaN and higher
            print(f'Kept existing model: saved best_hr {prev_hr:.4f} >= this run {trainer.best_hr:.4f} '
                  f'(use --force to overwrite).')
            return

    total_steps = len(history) * args.K
    save_model(net, optimizer, out_path, n_contents, conf.B_0,
               conf.ADMIT_TTL, conf.REJECT_TTL, total_steps, best_hr=trainer.best_hr)


if __name__ == '__main__':
    main()
