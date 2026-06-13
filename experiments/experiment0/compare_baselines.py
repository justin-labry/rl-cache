"""Compare RL-Cache against the AdmitAll and SecondHit baselines.

Each policy is run natively in IcarusGym (the common evaluation substrate) and
cross-checked against the validated standalone simulator (ttl_sim):

  * AdmitAll  - policy force-admit mode      (sim: decisions=all-ones)
  * SecondHit - policy SecondHit mode        (sim: baselines.secondhit_decisions)
  * RL-Cache  - trained NN, greedy eval      (sim: net greedy on batch features)

Prints a comparison table of hit rates (IcarusGym vs sim), admit fractions, and
lift over AdmitAll. IcarusGym/sim agreement re-confirms the simulator fidelity.

Usage:
    source .venv/bin/activate
    python experiments/experiment0/compare_baselines.py [--episodes 1]
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
from rl_cache import baselines
from rl_cache.rl_cache_policy import RLCacheNetwork
from rl_cache.training import ttl_sim
from ray.tune.registry import register_env

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), conf.MODEL_PATH)
TRACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trace_seed42.npz')


def _avg_episodes(agent, policy, episodes, label):
    rates = []
    for ep in range(episodes):
        agent.train()
        rates.append(policy._last_hit_rate)
    hr = float(np.mean(rates))
    print(f'  [IcarusGym] {label:10s}: hit_rate={hr:.6f}')
    return hr


def icarusgym_all(episodes):
    """Run AdmitAll, SecondHit, and RL-Cache in IcarusGym; return {name: hit_rate}."""
    logging.basicConfig(level=logging.WARNING)
    for name in ('icarusgym', 'icarus', 'rl_cache', 'experiments', 'root',
                 'icarusgym_actual_env'):
        logging.getLogger(name).setLevel(logging.WARNING)

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    register_env('TtlCache-v0', env_creator)
    agent = create_agent(conf, episode_measurement_begin=0,
                         result_output_file_name='results_compare')
    policy = agent.get_policy('default_policy')
    policy.load_model(MODEL_PATH)          # needed for the RL-Cache phase
    policy.set_eval_mode(True)             # no exploration, no learning for all phases

    results = {}

    # AdmitAll
    policy._force_admit, policy._secondhit = True, False
    results['AdmitAll'] = _avg_episodes(agent, policy, episodes, 'AdmitAll')

    # SecondHit
    policy._force_admit, policy._secondhit = False, True
    policy._secondhit_seen.clear()
    results['SecondHit'] = _avg_episodes(agent, policy, episodes, 'SecondHit')

    # RL-Cache (trained NN, greedy)
    policy._force_admit, policy._secondhit = False, False
    results['RL-Cache'] = _avg_episodes(agent, policy, episodes, 'RL-Cache')

    try:
        agent.stop()
    except Exception:
        pass
    if ray.is_initialized():
        ray.shutdown()
    stray = os.path.join(os.path.dirname(MODEL_PATH), 'results_compare.npz')
    if os.path.exists(stray):
        os.remove(stray)
    return results


def simulator_all():
    """Run the same three policies through ttl_sim; return {name: (hit_rate, admit_frac)}."""
    data = np.load(TRACE_PATH)
    trace = list(zip(data['times'].tolist(), data['contents'].tolist(), data['sizes'].tolist()))

    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    net = RLCacheNetwork(rlfeatures.FEATURE_DIM, conf.HIDDEN_DIM, conf.NUM_LAYERS)
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    feats = rlfeatures.extract_all(trace, conf.N_CONTENTS)
    with torch.no_grad():
        p = net(torch.from_numpy(feats)).squeeze(-1).numpy()

    decisions = {
        'AdmitAll': baselines.admitall_decisions(trace),
        'SecondHit': baselines.secondhit_decisions(trace),
        'RL-Cache': (p >= 0.5).astype(np.int64),
    }
    size_aware = icarus_conf.SIZE_MIN < icarus_conf.SIZE_MAX
    common = dict(cache_size=int(conf.B_0), admit_ttl=conf.ADMIT_TTL, reject_ttl=conf.REJECT_TTL,
                  is_reset=icarus_conf.IS_RESET, size_aware=size_aware,
                  default_size=icarus_conf.DEFAULT_SIZE)
    out = {}
    for name, dec in decisions.items():
        r = ttl_sim.simulate(trace, decisions=dec, **common)
        out[name] = (r['hit_rate'], float(np.mean(dec)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=1)
    args = ap.parse_args()
    if not os.path.exists(MODEL_PATH):
        sys.exit(f'No model at {MODEL_PATH}. Train first with train_mc.py')

    print('=== IcarusGym (native, common substrate) ===')
    ig = icarusgym_all(args.episodes)
    print('\n=== standalone simulator (cross-check) ===')
    sim = simulator_all()

    base = ig.get('AdmitAll', 0.0)
    print('\n' + '=' * 72)
    print(f'{"policy":12s} {"IcarusGym HR":>13s} {"sim HR":>10s} {"admit%":>8s} '
          f'{"|diff|":>8s} {"vs AdmitAll":>12s}')
    print('-' * 72)
    max_diff = 0.0
    for name in ('AdmitAll', 'SecondHit', 'RL-Cache'):
        ig_hr = ig[name]
        sim_hr, admit_frac = sim[name]
        diff = abs(ig_hr - sim_hr)
        max_diff = max(max_diff, diff)
        lift = (ig_hr - base) / base * 100 if base else 0.0
        print(f'{name:12s} {ig_hr:13.6f} {sim_hr:10.6f} {admit_frac * 100:7.1f}% '
              f'{diff:8.4f} {lift:+11.1f}%')
    print('=' * 72)
    print(f'max |IcarusGym - sim| = {max_diff:.4f} '
          f'({"OK, simulator faithful" if max_diff <= 0.005 else "CHECK: divergence"})')


if __name__ == '__main__':
    main()
