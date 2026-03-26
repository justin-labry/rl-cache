# RL-Cache Experiment 0 Main Script
#
# Two-phase execution:
#   Phase 1 (Training):  Episodes 0 ~ NUM_TRAIN_EPISODES-1
#       - Neural network learns via REINFORCE policy gradient
#       - Exploration (epsilon-greedy) is active
#       - Callbacks do NOT save results
#
#   Phase 2 (Evaluation): Episodes NUM_TRAIN_EPISODES ~ NUM_EPISODES-1
#       - Neural network is FROZEN (no learning, no exploration)
#       - Greedy actions only (action = argmax P(admit))
#       - Callbacks save per-content hit ratios to NPZ

# Author: labry

import sys
import os
import time

# Add IcarusGym and project root to Python path
sys.path.insert(0, '/home/labry/git/IcarusGym')
sys.path.insert(0, '/home/labry/git/rl-cache')

import experiments.experiment0.config as conf
import ray
import gymnasium as gym

from rl_cache.rl_cache_agent import RLCacheAgent
from icarusgym.envs import *
from ray.tune.registry import register_env


def env_creator(env_config):
    """Custom environment creator function for Ray."""
    return gym.make(id='TtlCache-v0', config=env_config)


def main():
    """Main function to set up and run the RL-Cache training + evaluation."""
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('icarusgym').setLevel(logging.WARNING)
    logging.getLogger('icarus').setLevel(logging.WARNING)
    logging.getLogger('rl_cache').setLevel(logging.INFO)
    logging.getLogger('experiments').setLevel(logging.INFO)
    logging.getLogger('root').setLevel(logging.WARNING)

    # Icarus simulator config is a SEPARATE file (icarus_config.py) from agent config (config.py)
    # to avoid import issues when IcarusGym's settings.read_from() loads it
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    env_config = {
        'config_path': os.path.join(experiment_dir, conf.TTLSIM_CONFIG_PATH),
        'output_path': conf.ICARUS_OUTPUT,
        'ttl_max': conf.TTL_MAX,
        'content_max': conf.CONTENT_MAX,
        'cache_size_max': conf.CACHE_SIZE_MAX
    }

    # Initialize Ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Register the custom environment
    register_env('TtlCache-v0', env_creator)

    # Create algorithm config
    config = RLCacheAgent.get_default_config().copy()
    config.update({
        # Callbacks
        'callbacks': conf.CALLBACKS,
        'callbacks_config': {
            'episode_measurement_begin': conf.EPISODE_MEASUREMENT_BEGIN,
            'result_output_file_name': conf.RESULT_OUTPUT_FILE_NAME
        },

        # Cache / environment parameters
        'n': conf.N,
        'b_0': conf.B_0,

        # Neural network architecture
        'feature_dim': conf.FEATURE_DIM,
        'hidden_dim': conf.HIDDEN_DIM,
        'num_layers': conf.NUM_LAYERS,

        # Training hyperparameters
        'lr': conf.LR,
        'gamma': conf.GAMMA,
        'entropy_coeff': conf.ENTROPY_COEFF,

        # Feature normalization
        'max_lambda': conf.MAX_LAMBDA,

        # TTL mapping
        'admit_ttl': conf.ADMIT_TTL,
        'reject_ttl': conf.REJECT_TTL,

        # Exploration schedule
        'explore_start': conf.EXPLORE_START,
        'explore_end': conf.EXPLORE_END,
        'explore_decay': conf.EXPLORE_DECAY,

        # Environment
        'env': 'TtlCache-v0',
        'env_config': env_config,
        'disable_env_checking': True,
        'rollout_fragment_length': conf.WORKLOAD_N_MEASURED,
        'batch_mode': 'complete_episodes',
    })

    # Create the agent
    agent = RLCacheAgent(config=config)

    try:
        print('=' * 70)
        print(f'RL-Cache Experiment 0')
        print(f'  Neural network: {conf.NUM_LAYERS} layers, {conf.HIDDEN_DIM} hidden dim')
        print(f'  Learning rate: {conf.LR}, Gamma: {conf.GAMMA}')
        print(f'  Admit TTL: {conf.ADMIT_TTL}, Reject TTL: {conf.REJECT_TTL}')
        print(f'  Contents: {conf.N}, Cache size: {conf.B_0}')
        print(f'  Training episodes:   {conf.NUM_TRAIN_EPISODES}')
        print(f'  Evaluation episodes: {conf.NUM_EVAL_EPISODES}')
        print('=' * 70)

        start_time = time.time()
        iteration_count = 0

        # ================================================================
        # Phase 1: TRAINING
        # ================================================================
        print(f'\n[Phase 1] Training ({conf.NUM_TRAIN_EPISODES} episodes)...')
        print('-' * 70)

        for ep in range(conf.NUM_TRAIN_EPISODES):
            iteration_count += 1
            results = agent.train()

            hit_rate = results.get('hit_rate', float('nan'))
            explore_rate = results.get('explore_rate', float('nan'))
            policy_loss = results.get('policy_loss', float('nan'))

            print(f'  [TRAIN] Episode {ep + 1:3d}/{conf.NUM_TRAIN_EPISODES} | '
                  f'Hit Rate: {hit_rate:.4f} | '
                  f'Loss: {policy_loss:.4f} | '
                  f'Explore: {explore_rate:.4f}')

        train_time = time.time() - start_time
        print(f'\nTraining complete. Time: {train_time:.1f}s')

        # ================================================================
        # Phase 2: EVALUATION (freeze model)
        # ================================================================
        print(f'\n[Phase 2] Evaluation ({conf.NUM_EVAL_EPISODES} episodes)...')
        print('-' * 70)

        # Switch policy to eval mode: no exploration, no learning
        policy = agent.get_policy('default_policy')
        policy.set_eval_mode(True)

        eval_hit_rates = []
        for ep in range(conf.NUM_EVAL_EPISODES):
            iteration_count += 1
            results = agent.train()  # Still calls train() but policy ignores learning

            hit_rate = results.get('hit_rate', float('nan'))
            eval_hit_rates.append(hit_rate)

            print(f'  [EVAL]  Episode {ep + 1:3d}/{conf.NUM_EVAL_EPISODES} | '
                  f'Hit Rate: {hit_rate:.4f}')

        # ================================================================
        # Summary
        # ================================================================
        total_time = time.time() - start_time
        mean_eval_hit_rate = sum(eval_hit_rates) / len(eval_hit_rates) if eval_hit_rates else 0.0

        print('\n' + '=' * 70)
        print(f'Experiment Complete')
        print(f'  Total time:          {total_time:.1f}s')
        print(f'  Training episodes:   {conf.NUM_TRAIN_EPISODES}')
        print(f'  Evaluation episodes: {conf.NUM_EVAL_EPISODES}')
        print(f'  Mean eval hit rate:  {mean_eval_hit_rate:.4f}')
        print(f'  Results saved to:    {conf.RESULT_OUTPUT_FILE_NAME}.npz')
        print('=' * 70)

    except KeyboardInterrupt:
        print('\nTraining interrupted by user.')
    except Exception as e:
        print(f'\nAn error occurred: {e}')
        import traceback
        traceback.print_exc()
    finally:
        print('Cleaning up...')
        if 'agent' in locals() and agent:
            try:
                agent.stop()
                print('Agent stopped.')
            except Exception as e:
                print(f'Error stopping agent: {e}')

        if ray.is_initialized():
            try:
                ray.shutdown()
                print('Ray shut down.')
            except Exception as e:
                print(f'Error shutting down Ray: {e}')

        print('Done.')
        sys.exit(0)


if __name__ == '__main__':
    main()
