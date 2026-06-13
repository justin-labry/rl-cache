"""Cache-size sweep: compare RL-Cache vs AdmitAll/SecondHit across cache sizes,
reporting both Object Hit Rate (OHR) and Byte Hit Rate (BHR).

For each cache ratio the cache size is the admission "aggressiveness knob"
(RL-Cache paper, §4): a fresh RL-Cache model is trained at that size (seed sweep,
best checkpoint by greedy hit rate), then AdmitAll, SecondHit and RL-Cache are
evaluated on the standalone simulator -- which reproduces IcarusGym hit/miss
exactly (validate_sim.py) -- so the whole sweep runs offline and fast.

Output: a table of OHR and BHR per (cache ratio, policy), the data behind a
"hit rate vs cache size" figure.

Usage:
    source .venv/bin/activate
    python experiments/experiment0/sweep_cache_size.py [--ratios 0.05 0.1 0.2] [--seeds 0 1] [--passes 3]
"""

# Author: labry

import sys
import os
import argparse
import copy
import time

sys.path.insert(0, '/home/labry/git/IcarusGym')
sys.path.insert(0, '/home/labry/git/rl-cache')

import numpy as np
import torch

import experiments.experiment0.config as conf
import experiments.experiment0.icarus_config as icarus_conf
from rl_cache import features as rlfeatures
from rl_cache import baselines
from rl_cache.rl_cache_policy import RLCacheNetwork
from rl_cache.training.mc_trainer import MCTrainer
from rl_cache.training import ttl_sim

TRACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trace_seed42.npz')


def train_best(trace, feats, cache_size, seeds, passes, common, reward_mode='ohr'):
    """Seed-sweep MC training at a fixed cache size for a given reward metric.

    Returns the best net state and its best metric value (OHR if reward_mode=='ohr',
    BHR if 'bhr'). Only the sample-ranking reward differs between modes.
    """
    best_state, best_hr = None, -1.0
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = RLCacheNetwork(rlfeatures.FEATURE_DIM, conf.HIDDEN_DIM, conf.NUM_LAYERS)
        opt = torch.optim.Adam(net.parameters(), lr=conf.LR)
        trainer = MCTrainer(net, opt, trace, feats, cache_size=cache_size,
                            admit_ttl=common['admit_ttl'], reject_ttl=common['reject_ttl'],
                            is_reset=common['is_reset'], size_aware=common['size_aware'],
                            default_size=common['default_size'],
                            K=1000, m=400, p_percentile=20.0, L=3000, gamma=0.99997, q=4,
                            target_mode='advantage', adv_gain=2.0, seed=seed,
                            reward_mode=reward_mode)
        trainer.train(num_passes=passes, verbose=False)
        print(f'    [{reward_mode}] seed {seed}: best {reward_mode} {trainer.best_hr:.4f}')
        if trainer.best_hr > best_hr and trainer.best_state is not None:
            best_hr = trainer.best_hr
            best_state = copy.deepcopy(trainer.best_state)
    return best_state, best_hr


def greedy_decisions(state, feats):
    """Greedy admit decisions for a saved net state."""
    net = RLCacheNetwork(rlfeatures.FEATURE_DIM, conf.HIDDEN_DIM, conf.NUM_LAYERS)
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        p = net(torch.from_numpy(feats)).squeeze(-1).numpy()
    return (p >= 0.5).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.05, 0.1, 0.2])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--passes', type=int, default=3)
    args = ap.parse_args()

    if not os.path.exists(TRACE_PATH):
        sys.exit(f'No trace at {TRACE_PATH}. Run extract_trace.py first.')
    data = np.load(TRACE_PATH)
    trace = list(zip(data['times'].tolist(), data['contents'].tolist(), data['sizes'].tolist()))
    feats = rlfeatures.extract_all(trace, conf.N_CONTENTS)

    size_aware = icarus_conf.SIZE_MIN < icarus_conf.SIZE_MAX
    common = dict(admit_ttl=conf.ADMIT_TTL, reject_ttl=conf.REJECT_TTL,
                  is_reset=icarus_conf.IS_RESET, size_aware=size_aware,
                  default_size=icarus_conf.DEFAULT_SIZE)
    sh_dec = baselines.secondhit_decisions(trace)   # cache-size independent

    rows = []
    for ratio in args.ratios:
        cache_size = max(1, int(conf.N_CONTENTS * ratio))
        print(f'\n=== cache ratio {ratio} (cache_size={cache_size}) ===')
        t0 = time.time()
        # Train RL-Cache twice: optimizing OHR, then optimizing BHR (reward choice only).
        ohr_state, _ = train_best(trace, feats, cache_size, args.seeds, args.passes, common, 'ohr')
        bhr_state, _ = train_best(trace, feats, cache_size, args.seeds, args.passes, common, 'bhr')
        greedy_ohr = greedy_decisions(ohr_state, feats)
        greedy_bhr = greedy_decisions(bhr_state, feats)

        sim = dict(cache_size=cache_size, **common)
        res = {
            'AdmitAll': ttl_sim.simulate(trace, None, **sim),
            'SecondHit': ttl_sim.simulate(trace, sh_dec, **sim),
            'RL-Cache(OHR)': ttl_sim.simulate(trace, greedy_ohr, **sim),
            'RL-Cache(BHR)': ttl_sim.simulate(trace, greedy_bhr, **sim),
        }
        for name, r in res.items():
            rows.append((ratio, cache_size, name, r['hit_rate'], r['byte_hit_rate'],
                         float(np.mean(greedy_ohr if name == 'RL-Cache(OHR)'
                               else greedy_bhr if name == 'RL-Cache(BHR)'
                               else (sh_dec if name == 'SecondHit' else np.ones(len(trace)))))))
        print(f'    trained+evaluated in {time.time() - t0:.0f}s')

    print('\n' + '=' * 78)
    print(f'{"ratio":>6s} {"cache":>6s} {"policy":15s} {"OHR":>9s} {"BHR":>9s} {"admit%":>8s}')
    print('-' * 78)
    for ratio, csize, name, ohr, bhr, admit in rows:
        print(f'{ratio:6.2f} {csize:6d} {name:15s} {ohr:9.4f} {bhr:9.4f} {admit * 100:7.1f}%')
    print('=' * 78)

    # Save structured results for plotting (plot_sweep.py).
    import json
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_results.json')
    with open(out_json, 'w') as f:
        json.dump([{'ratio': r, 'cache_size': c, 'policy': n, 'ohr': o, 'bhr': b, 'admit_frac': a}
                   for r, c, n, o, b, a in rows], f, indent=2)
    print(f'Saved {out_json}')


if __name__ == '__main__':
    main()
