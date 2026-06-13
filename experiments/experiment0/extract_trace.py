"""Extract the deterministic (SEED=42) request trace from IcarusGym and save it
to ``trace_seed42.npz`` for Monte Carlo training.

The workload is deterministic, so this is a one-time extraction: the saved trace
is exactly what IcarusGym replays each episode, and the standalone simulator was
validated (validate_sim.py) to reproduce IcarusGym's cache behaviour on it.

Saved arrays: times (float64), contents (int64), sizes (float64).

Usage:
    source .venv/bin/activate
    python experiments/experiment0/extract_trace.py
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
from experiments.experiment0.main import create_agent, env_creator
from ray.tune.registry import register_env

TRACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trace_seed42.npz')


def main():
    logging.basicConfig(level=logging.WARNING)
    for name in ('icarusgym', 'icarus', 'rl_cache', 'experiments', 'root',
                 'icarusgym_actual_env'):
        logging.getLogger(name).setLevel(logging.WARNING)

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    register_env('TtlCache-v0', env_creator)

    agent = create_agent(conf, episode_measurement_begin=0,
                         result_output_file_name='results_extract_trace')
    policy = agent.get_policy('default_policy')
    policy.set_eval_mode(True)
    policy._force_admit = True
    policy._capture_buffer = []

    print('Capturing one episode of the SEED=42 trace from IcarusGym...')
    agent.train()
    trace = list(policy._capture_buffer)

    try:
        agent.stop()
    except Exception:
        pass
    if ray.is_initialized():
        ray.shutdown()

    times = np.array([t[0] for t in trace], dtype=np.float64)
    contents = np.array([t[1] for t in trace], dtype=np.int64)
    sizes = np.array([t[2] for t in trace], dtype=np.float64)

    np.savez(TRACE_PATH, times=times, contents=contents, sizes=sizes)
    print(f'Saved {len(times)} requests to {TRACE_PATH}')
    print(f'  unique contents: {len(np.unique(contents))}')
    print(f'  size range: [{sizes.min():.0f}, {sizes.max():.0f}]')
    # Clean up the stray results file the callback may have written.
    stray = os.path.join(os.path.dirname(TRACE_PATH), 'results_extract_trace.npz')
    if os.path.exists(stray):
        os.remove(stray)


if __name__ == '__main__':
    main()
