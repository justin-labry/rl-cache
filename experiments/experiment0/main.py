# RL-Cache Experiment 0 Main Script
#
# Usage:
#   python main.py --mode train   # Train NN and save to model.pt
#   python main.py --mode test    # Load model.pt and evaluate
#   python main.py --mode both    # Train → save → evaluate (default)
#
# The --mode flag controls the execution flow:
#   train: Runs NUM_TRAIN_EPISODES, saves model to MODEL_PATH, exits
#   test:  Loads model from MODEL_PATH, runs NUM_EVAL_EPISODES, saves results to NPZ
#   both:  Train phase → save model → eval phase → save results (legacy behavior)

# Author: labry

import sys
import os
import time
import argparse

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


def create_agent(conf, episode_measurement_begin=None, result_output_file_name=None):
    """Create and configure the RLCacheAgent.

    :param conf: Configuration module (experiments.experiment0.config)
    :param episode_measurement_begin: Override when to start recording (default: conf).
        Use 0 for --mode test so eval episodes 0..N-1 save results (train uses conf value).
    :return: Configured RLCacheAgent instance
    """
    measurement_begin = (
        conf.EPISODE_MEASUREMENT_BEGIN
        if episode_measurement_begin is None
        else episode_measurement_begin
    )
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    env_config = {
        'config_path': os.path.join(experiment_dir, conf.TTLSIM_CONFIG_PATH),
        'output_path': conf.ICARUS_OUTPUT,
        'ttl_max': conf.TTL_MAX,
        'content_max': conf.CONTENT_MAX,
        'cache_size_max': conf.CACHE_SIZE_MAX
    }

    config = RLCacheAgent.get_default_config().copy()
    config.update({
        # Callbacks
        'callbacks': conf.CALLBACKS,
        'callbacks_config': {
            'episode_measurement_begin': measurement_begin,
            'result_output_file_name': result_output_file_name or conf.RESULT_OUTPUT_FILE_NAME
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

    return RLCacheAgent(config=config)


def run_train(agent, conf):
    """Run training phase: train NN and save model.

    :param agent: RLCacheAgent instance
    :param conf: Configuration module
    """
    print(f'\n[TRAIN] Training ({conf.NUM_TRAIN_EPISODES} episodes)...')
    print('-' * 70)

    policy = agent.get_policy('default_policy')
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), conf.MODEL_PATH)

    for ep in range(conf.NUM_TRAIN_EPISODES):
        agent.train()

        hit_rate = policy._last_hit_rate
        policy_loss = policy._last_policy_loss
        explore_rate = policy._get_explore_rate()

        print(f'  [TRAIN] Episode {ep + 1:3d}/{conf.NUM_TRAIN_EPISODES} | '
              f'Hit Rate: {hit_rate:.4f} | '
              f'Loss: {policy_loss:.4f} | '
              f'Explore: {explore_rate:.4f}')

    # Save the trained model
    policy = agent.get_policy('default_policy')
    policy.save_model(model_path)
    print(f'\nModel saved to: {model_path}')


def run_test(agent, conf):
    """Run test phase: load model and evaluate.

    :param agent: RLCacheAgent instance
    :param conf: Configuration module
    """
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), conf.MODEL_PATH)

    # Load trained model
    policy = agent.get_policy('default_policy')
    policy.load_model(model_path)
    print(f'Model loaded from: {model_path}')

    # Switch to eval mode: no exploration, no learning
    policy.set_eval_mode(True)

    print(f'\n[TEST] Evaluation ({conf.NUM_EVAL_EPISODES} episodes)...')
    print('-' * 70)

    eval_hit_rates = []
    for ep in range(conf.NUM_EVAL_EPISODES):
        agent.train()  # drives the episode loop; policy ignores learning in eval mode

        hit_rate = policy._last_hit_rate
        eval_hit_rates.append(hit_rate)

        print(f'  [TEST]  Episode {ep + 1:3d}/{conf.NUM_EVAL_EPISODES} | '
              f'Hit Rate: {hit_rate:.4f}')

    mean_eval_hit_rate = sum(eval_hit_rates) / len(eval_hit_rates) if eval_hit_rates else 0.0
    print(f'\n  Mean eval hit rate: {mean_eval_hit_rate:.4f}')
    print(f'  Results saved to:  {conf.RESULT_OUTPUT_FILE_NAME}.npz')
    return mean_eval_hit_rate


def run_baseline(agent, conf):
    """Run LRU-equivalent baseline: always admit, no learning.

    Sets the force_admit flag on the policy so every request receives
    ADMIT_TTL, making the size-aware TTL cache behave like LRU.
    This provides a fair baseline comparison on the same cache and workload.

    :param agent: RLCacheAgent instance
    :param conf: Configuration module
    """
    policy = agent.get_policy('default_policy')
    policy.set_eval_mode(True)
    policy._force_admit = True

    print(f'\n[BASELINE] LRU-equivalent evaluation ({conf.NUM_EVAL_EPISODES} episodes)...')
    print('-' * 70)

    eval_hit_rates = []
    for ep in range(conf.NUM_EVAL_EPISODES):
        agent.train()
        hit_rate = policy._last_hit_rate
        eval_hit_rates.append(hit_rate)
        print(f'  [BASELINE] Episode {ep + 1:3d}/{conf.NUM_EVAL_EPISODES} | '
              f'Hit Rate: {hit_rate:.4f}')

    valid = [h for h in eval_hit_rates if h == h]  # filter NaN
    mean_hit = sum(valid) / len(valid) if valid else 0.0
    print(f'\n  Mean baseline hit rate: {mean_hit:.4f}')
    return mean_hit


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='RL-Cache Experiment')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['train', 'test', 'both', 'baseline'],
                        help='Execution mode: train, test, both, or baseline (LRU-equivalent always-admit)')
    args = parser.parse_args()

    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('icarusgym').setLevel(logging.WARNING)
    logging.getLogger('icarus').setLevel(logging.WARNING)
    logging.getLogger('rl_cache').setLevel(logging.INFO)
    logging.getLogger('experiments').setLevel(logging.INFO)
    logging.getLogger('root').setLevel(logging.WARNING)

    # Initialize Ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Register the custom environment
    register_env('TtlCache-v0', env_creator)

    # Create the agent (baseline uses a separate results file to avoid overwriting RL-Cache results)
    result_file = 'results_lru_baseline' if args.mode == 'baseline' else None
    agent = create_agent(
        conf,
        episode_measurement_begin=0 if args.mode in ('test', 'baseline') else None,
        result_output_file_name=result_file,
    )

    try:
        print('=' * 70)
        print(f'RL-Cache Experiment 0  [mode: {args.mode}]')
        print(f'  Neural network: {conf.NUM_LAYERS} layers, {conf.HIDDEN_DIM} hidden dim')
        print(f'  Learning rate: {conf.LR}, Gamma: {conf.GAMMA}')
        print(f'  Admit TTL: {conf.ADMIT_TTL}, Reject TTL: {conf.REJECT_TTL}')
        print(f'  Contents: {conf.N}, Cache size: {conf.B_0}')
        print(f'  Model path: {conf.MODEL_PATH}')
        print('=' * 70)

        start_time = time.time()

        if args.mode == 'train':
            # Train only: train NN → save model → exit
            run_train(agent, conf)

        elif args.mode == 'test':
            # Test only: load model → evaluate → save results
            run_test(agent, conf)

        elif args.mode == 'both':
            # Full pipeline: train → save → eval
            run_train(agent, conf)
            run_test(agent, conf)

        elif args.mode == 'baseline':
            # LRU-equivalent baseline: always admit, no learning
            run_baseline(agent, conf)

        total_time = time.time() - start_time
        print('\n' + '=' * 70)
        print(f'Done. Total time: {total_time:.1f}s')
        print('=' * 70)

    except KeyboardInterrupt:
        print('\nInterrupted by user.')
    except Exception as e:
        print(f'\nError: {e}')
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
