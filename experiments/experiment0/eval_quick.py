"""End-to-end check: does the MC-trained selective policy reproduce its hit rate
in IcarusGym, and do IcarusGym and the standalone simulator agree on a non-trivial
(trained, greedy) policy?

  1. Load model.pt, run N greedy eval episodes in IcarusGym -> IcarusGym hit rate.
     This exercises the ONLINE FeatureExtractor in RLCachePolicy._extract_features.
  2. Standalone: load the same net, compute greedy decisions on the SEED=42 trace
     using the BATCH feature path, and replay through ttl_sim -> simulator hit rate.
  3. Compare. A match proves (a) eval features == training features, and
     (b) the learned selectivity carries over from the trainer to IcarusGym.

Also reports AdmitAll (force-admit) for reference.

Usage:
    source .venv/bin/activate
    python experiments/experiment0/eval_quick.py [--episodes 2]
"""

# Author: labry

import sys
import os
import argparse

sys.path.insert(0, '/home/labry/git/IcarusGym')
sys.path.insert(0, '/home/labry/git/rl-cache')

import logging
import numpy as np
import torch
import ray

import experiments.experiment0.config as conf
import experiments.experiment0.icarus_config as icarus_conf
from experiments.experiment0.main import create_agent, env_creator
from rl_cache import features as rlfeatures
from rl_cache.rl_cache_policy import RLCacheNetwork
from rl_cache.training import ttl_sim
from ray.tune.registry import register_env

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), conf.MODEL_PATH)
TRACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trace_seed42.npz')


def icarusgym_eval(episodes):
    """Run `episodes` greedy eval episodes in IcarusGym; return per-episode hit rates."""
    logging.basicConfig(level=logging.WARNING)
    for name in ('icarusgym', 'icarus', 'rl_cache', 'experiments', 'root',
                 'icarusgym_actual_env'):
        logging.getLogger(name).setLevel(logging.WARNING)

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    register_env('TtlCache-v0', env_creator)
    agent = create_agent(conf, episode_measurement_begin=0,
                         result_output_file_name='results_eval_quick')
    policy = agent.get_policy('default_policy')
    policy.load_model(MODEL_PATH)
    policy.set_eval_mode(True)          # greedy, no exploration, no learning

    hit_rates = []
    for ep in range(episodes):
        agent.train()                   # drives one eval episode
        hit_rates.append(policy._last_hit_rate)
        print(f'  [IcarusGym] episode {ep + 1}: hit_rate={policy._last_hit_rate:.6f}')

    try:
        agent.stop()
    except Exception:
        pass
    if ray.is_initialized():
        ray.shutdown()
    # Clean up stray results file.
    stray = os.path.join(os.path.dirname(MODEL_PATH), 'results_eval_quick.npz')
    if os.path.exists(stray):
        os.remove(stray)
    return hit_rates


def standalone_eval():
    """Replay the trained greedy policy and AdmitAll through ttl_sim."""
    data = np.load(TRACE_PATH)
    trace = list(zip(data['times'].tolist(), data['contents'].tolist(), data['sizes'].tolist()))

    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    net = RLCacheNetwork(rlfeatures.FEATURE_DIM, conf.HIDDEN_DIM, conf.NUM_LAYERS)
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()

    feats = rlfeatures.extract_all(trace, conf.N_CONTENTS)
    with torch.no_grad():
        p = net(torch.from_numpy(feats)).squeeze(-1).numpy()
    greedy = (p >= 0.5).astype(np.int64)

    size_aware = icarus_conf.SIZE_MIN < icarus_conf.SIZE_MAX
    common = dict(cache_size=int(conf.B_0), admit_ttl=conf.ADMIT_TTL, reject_ttl=conf.REJECT_TTL,
                  is_reset=icarus_conf.IS_RESET, size_aware=size_aware,
                  default_size=icarus_conf.DEFAULT_SIZE)

    trained = ttl_sim.simulate(trace, decisions=greedy, **common)
    admitall = ttl_sim.simulate(trace, decisions=None, **common)
    return trained, admitall, float(greedy.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=2)
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        sys.exit(f'No model at {MODEL_PATH}. Train first: python experiments/experiment0/train_mc.py ...')

    print('=== IcarusGym greedy eval (online features) ===')
    ig_rates = icarusgym_eval(args.episodes)
    ig_hit = float(np.mean(ig_rates))

    print('\n=== Standalone simulator (batch features) ===')
    trained, admitall, admit_frac = standalone_eval()
    print(f'  [sim] trained greedy hit_rate = {trained["hit_rate"]:.6f}  (admit fraction {admit_frac:.3f})')
    print(f'  [sim] AdmitAll      hit_rate = {admitall["hit_rate"]:.6f}')

    print('\n=== end-to-end comparison ===')
    print(f'  IcarusGym trained hit rate : {ig_hit:.6f}')
    print(f'  simulator trained hit rate : {trained["hit_rate"]:.6f}')
    diff = abs(ig_hit - trained['hit_rate'])
    print(f'  |difference|               : {diff:.6f}')
    print(f'  AdmitAll baseline (sim)    : {admitall["hit_rate"]:.6f}')
    lift = (ig_hit - admitall['hit_rate']) / admitall['hit_rate'] * 100 if admitall['hit_rate'] else 0.0
    print(f'  trained vs AdmitAll lift   : {lift:+.1f}%')

    print('\n' + '=' * 60)
    if diff <= 0.005:
        print('PASS: IcarusGym reproduces the simulator hit rate (features consistent,')
        print('      selectivity carries over end-to-end).')
        if ig_hit > admitall['hit_rate'] + 0.01:
            print(f'      Trained policy beats AdmitAll by {lift:+.1f}% in IcarusGym.')
    else:
        print(f'MISMATCH (|diff|={diff:.4f}): online eval features likely differ from')
        print('      training features. Inspect FeatureExtractor usage in compute_actions.')
    print('=' * 60)


if __name__ == '__main__':
    main()
