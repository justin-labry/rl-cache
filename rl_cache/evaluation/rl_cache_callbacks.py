"""
Callbacks for RL-Cache experiments.

Tracks per-content cache hit ratios and saves results to NPZ files,
following the same pattern as DehghanCacheCallbacks / Experiment0Callbacks.
"""

# Author: labry

import logging
import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict


logger = logging.getLogger(__name__)


class RLCacheCallbacks(DefaultCallbacks):
    """Callbacks for tracking RL-Cache performance metrics.

    Tracks per-content hit ratios and neural network training metrics.
    Saves results to NPZ files at the end of each episode (after measurement begins).
    """

    DEFAULT_CONFIG = {
        'episode_measurement_begin': 1,         # Episode number to begin performance measurement
        'result_output_file_name': 'results',   # Result output file name (without .npz extension)
        'workload_n_warm_up': 0,                # Skip this many steps per episode (warm-up)
    }

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        self._policy = None
        self._episode_seq_num = 0
        self._request_nums = None
        self._hit_nums = None
        self._episode_measurement_begin = None
        self._result_output_file_name = None

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
                         episode: Episode, **kwargs):
        self._policy = policies['default_policy']
        # n+1 slots for content IDs 0..n (1-based catalog). Reallocate if policy.n
        # changes (e.g. load_model after train; train→test continues at episode 300).
        need = self._policy.n + 1
        if self._request_nums is None or self._request_nums.shape[0] != need:
            self._request_nums = np.zeros(shape=need, dtype=np.int32)
            self._hit_nums = np.zeros(shape=need, dtype=np.int32)
        self._episode_measurement_begin = self._policy.config['callbacks_config']['episode_measurement_begin']
        self._result_output_file_name = self._policy.config['callbacks_config']['result_output_file_name']
        self._workload_n_warm_up = self._policy.config['callbacks_config'].get('workload_n_warm_up', 0)
        # Reset stats at measurement begin
        if self._episode_seq_num == self._episode_measurement_begin and self._request_nums is not None:
            self._request_nums.fill(0)
            self._hit_nums.fill(0)
            logger.info(f'Reset request/hit counters at episode {self._episode_seq_num} (measurement begin).')
        self._steps_this_episode = 0
        logger.info(f'Episode {self._episode_seq_num} started.')

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv, episode: Episode, **kwargs):
        if self._episode_seq_num >= self._episode_measurement_begin:
            self._steps_this_episode = getattr(self, '_steps_this_episode', 0) + 1
            # Skip warm-up steps
            if self._steps_this_episode <= self._workload_n_warm_up:
                return

            try:
                # Try to get last observation (handle both old and new RLlib API)
                if hasattr(episode, '_agent_collectors'):
                    agent_ids = episode.get_agents()
                    if len(agent_ids) == 0:
                        return
                    agent_id = agent_ids[0]
                    if agent_id in episode._agent_collectors:
                        collector = episode._agent_collectors[agent_id]
                        if hasattr(collector, 'buffers') and 'obs' in collector.buffers:
                            obs_buffer = collector.buffers['obs']
                            if len(obs_buffer) > 0 and len(obs_buffer[0]) > 0:
                                last_obs = obs_buffer[0][-1]
                            else:
                                return
                        else:
                            return
                    else:
                        return
                elif hasattr(episode, 'last_observation_for'):
                    last_obs = episode.last_observation_for()
                else:
                    return

                # Extract content ID and hit status
                i = int(last_obs[1])
                if i < 0 or i > self._policy.n:
                    return

                # Hit information at index 5 for TtlCache-v0
                if len(last_obs) > 5:
                    hit = last_obs[5]
                else:
                    hit = last_obs[2]

                self._request_nums[i] += 1
                if hit:
                    self._hit_nums[i] += 1

            except (AttributeError, TypeError, IndexError) as e:
                if not hasattr(self, '_warned_no_obs'):
                    logger.warning(f'Could not get last observation: {type(e).__name__}: {e}')
                    self._warned_no_obs = True

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
                       episode: Episode, **kwargs):
        if self._episode_seq_num >= self._episode_measurement_begin:
            # Use policy.n - 1, not policy.config['n']: RLlib may merge keys so
            # config['n'] can disagree with RLCachePolicy (e.g. cache_size vs catalog size).
            n_contents = self._policy.n - 1
            if n_contents <= 0:
                n_contents = self._policy.n

            is_ = []
            hit_ratios = []

            for i in range(1, n_contents + 1):
                req = self._request_nums[i]
                hits = self._hit_nums[i]
                if req <= 1:
                    hit_ratio = 0.0
                else:
                    hit_ratio = min(1.0, float(hits) / float(req - 1))
                is_.append(i)
                hit_ratios.append(hit_ratio)

            is_ = np.array(is_, dtype=np.int32)
            hit_ratios = np.array(hit_ratios, dtype=np.float64)
            mean_hit_ratio = np.mean(hit_ratios) if len(hit_ratios) > 0 else 0.0

            logger.info(f'Episode {self._episode_seq_num}: '
                        f'Contents tracked: {len(is_)}/{n_contents}, '
                        f'Mean hit ratio: {mean_hit_ratio:.4f}')

            np.savez(self._result_output_file_name,
                     is_=is_,
                     hit_probs=hit_ratios,
                     request_counts=self._request_nums[1:n_contents + 1],
                     hit_counts=self._hit_nums[1:n_contents + 1],
                     mean_cache_hit_ratio=mean_hit_ratio)

            logger.info(f'Saved results to {self._result_output_file_name}.npz')

        self._episode_seq_num += 1
