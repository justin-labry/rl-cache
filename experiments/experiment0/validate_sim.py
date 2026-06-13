"""Step-0 de-risk spike: validate the standalone TTL simulator against the real
IcarusGym pipeline.

Procedure:
  1. Run IcarusGym in force-admit (AdmitAll / LRU-equivalent) for ONE episode,
     capturing the exact per-request trace IcarusGym processed:
       (env_time, content_id, size, remaining_ttl, hit)
  2. Replay the captured (time, content, size) trace through rl_cache.training
     .ttl_sim with force-admit and the SAME cache configuration.
  3. Compare per-request hit (and remaining TTL) between IcarusGym and the sim.

PASS criterion: the simulator reproduces IcarusGym's hit/miss decision for every
request. If it does, the Monte Carlo trainer can be built on the sim with
confidence that training and IcarusGym evaluation share identical cache behaviour.

Usage:
    source .venv/bin/activate
    python experiments/experiment0/validate_sim.py
"""

# Author: labry

import sys
import os

sys.path.insert(0, '/home/labry/git/IcarusGym')
sys.path.insert(0, '/home/labry/git/rl-cache')

import logging
import numpy as np
import ray

import experiments.experiment0.config as conf
import experiments.experiment0.icarus_config as icarus_conf
from experiments.experiment0.main import create_agent, env_creator
from rl_cache.training import ttl_sim
from ray.tune.registry import register_env


def run_icarusgym_capture():
    """Run one force-admit episode in IcarusGym and return the captured trace."""
    logging.basicConfig(level=logging.WARNING)
    for name in ('icarusgym', 'icarus', 'rl_cache', 'experiments', 'root',
                 'icarusgym_actual_env'):
        logging.getLogger(name).setLevel(logging.WARNING)

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    register_env('TtlCache-v0', env_creator)

    # measurement_begin=0 keeps the callback happy; we only use the capture buffer.
    agent = create_agent(conf, episode_measurement_begin=0,
                         result_output_file_name='results_spike_baseline')
    policy = agent.get_policy('default_policy')

    # Force-admit (AdmitAll), no exploration, no learning, and turn on trace capture.
    policy.set_eval_mode(True)
    policy._force_admit = True
    policy._capture_buffer = []

    print('Running 1 force-admit episode in IcarusGym (capturing trace)...')
    agent.train()
    trace = list(policy._capture_buffer)

    try:
        agent.stop()
    except Exception:
        pass
    if ray.is_initialized():
        ray.shutdown()

    return trace


def main():
    trace = run_icarusgym_capture()
    n = len(trace)
    if n == 0:
        print('FAIL: no requests captured from IcarusGym.')
        sys.exit(1)

    # Captured columns: (env_time, content_id, size, remaining_ttl, hit)
    times = [t[0] for t in trace]
    contents = [t[1] for t in trace]
    sizes = [t[2] for t in trace]
    ig_rttl = [t[3] for t in trace]
    ig_hits = [int(t[4]) for t in trace]

    # Cache configuration, read from the experiment configs (no magic numbers).
    size_aware = icarus_conf.SIZE_MIN < icarus_conf.SIZE_MAX
    default_size = icarus_conf.DEFAULT_SIZE
    is_reset = icarus_conf.IS_RESET
    cache_size = int(conf.B_0)
    admit_ttl = conf.ADMIT_TTL
    reject_ttl = conf.REJECT_TTL

    print('\n--- captured trace ---')
    print(f'  requests captured : {n}')
    print(f'  unique contents   : {len(set(contents))}')
    print(f'  IcarusGym hit rate: {np.mean(ig_hits):.6f}')
    print('--- cache config (from experiment files) ---')
    print(f'  cache_size={cache_size}  is_reset={is_reset}  size_aware={size_aware}  '
          f'default_size={default_size}  admit_ttl={admit_ttl}')

    requests = list(zip(times, contents, sizes))
    result = ttl_sim.simulate(
        requests, decisions=None,          # None -> force-admit (AdmitAll)
        cache_size=cache_size, admit_ttl=admit_ttl, reject_ttl=reject_ttl,
        is_reset=is_reset, size_aware=size_aware, default_size=default_size,
    )
    sim_hits = result['hits']

    # --- per-request hit comparison (the primary criterion) ---
    matches = sum(1 for a, b in zip(ig_hits, sim_hits) if a == b)
    mismatches = [k for k in range(n) if ig_hits[k] != sim_hits[k]]
    match_pct = 100.0 * matches / n

    print('\n--- comparison ---')
    print(f'  IcarusGym hit rate : {np.mean(ig_hits):.6f}')
    print(f'  simulator hit rate : {result["hit_rate"]:.6f}')
    print(f'  per-request match  : {matches}/{n}  ({match_pct:.4f}%)')

    # remaining-TTL secondary check (soft): mean abs diff on hits IcarusGym saw
    diffs = [abs(ig_rttl[k] - result['remaining_ttls'][k])
             for k in range(n) if ig_hits[k] == 1 and sim_hits[k] == 1]
    if diffs:
        print(f'  remaining-ttl |diff| on common hits: '
              f'mean={np.mean(diffs):.4g}  max={np.max(diffs):.4g}')

    if mismatches:
        print(f'\n  first mismatches (k: content | IcarusGym hit / sim hit):')
        for k in mismatches[:15]:
            print(f'    {k:6d}: content={contents[k]:5d} size={sizes[k]:.0f} | '
                  f'IG={ig_hits[k]} sim={sim_hits[k]}')

    print('\n' + '=' * 60)
    if match_pct == 100.0:
        print('PASS: simulator reproduces IcarusGym hit/miss for every request.')
        print('The Monte Carlo trainer can be built on the standalone simulator.')
    elif match_pct >= 99.5:
        print(f'NEAR-PASS ({match_pct:.4f}%): a few mismatches remain '
              f'(likely expiration tie-breaking). Inspect the list above.')
    else:
        print(f'FAIL ({match_pct:.4f}%): simulator diverges from IcarusGym. '
              f'Cache logic replication needs fixing before building MC training.')
    print('=' * 60)


if __name__ == '__main__':
    main()
