"""
RL-Cache algorithm implementation for RLlib.

This module wraps RLCachePolicy as an RLlib Algorithm, following the same pattern as
DehghanCacheAlgorithm in the dehghan-cache project. It provides:
    - Default configuration for RL-Cache
    - Configuration validation
    - Training step that collects samples and trains the neural network policy

Reference:
    N. Beckmann et al., "RL-Cache: Learning-Based Cache Admission for Content Delivery"
"""

# Author: labry

import logging

from rl_cache.rl_cache_policy import RLCachePolicy
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.annotations import override


logger = logging.getLogger(__name__)


# Default configuration for RL-Cache
DEFAULT_CONFIG = {
    # RLlib framework settings
    'num_workers': 0,
    'rollout_fragment_length': 1,
    'train_batch_size': 1,
    'framework': 'torch',
    'clip_actions': False,
    'clip_rewards': False,
    'callbacks': None,              # Callbacks class for logging custom performance metrics
    'callbacks_config': None,       # Configs for callbacks
    'checkpoint_json_indent': 4,    # Number of indentation of JSON document for checkpoint

    # Neural network architecture
    'feature_dim': 6,       # Dimension of input feature vector
    'hidden_dim': 64,       # Dimension of each hidden layer
    'num_layers': 5,        # Number of hidden layers (RL-Cache paper uses 5)

    # Training hyperparameters
    'lr': 1e-3,             # Learning rate for Adam optimizer
    'gamma': 0.99,          # Discount factor for REINFORCE
    'entropy_coeff': 0.01,  # Entropy regularization coefficient

    # TTL mapping for admit/reject decisions
    'admit_ttl': 100.0,     # TTL value when admitting content to cache
    'reject_ttl': 0.1,      # TTL value when rejecting content (effectively not caching)

    # Cache / environment parameters
    'n': 16,                # Number of contents
    'b_0': 5,               # Cache size

    # Feature normalization
    'max_lambda': 10.0,         # Max arrival rate for feature clipping/normalization

    # Exploration schedule (epsilon-greedy)
    'explore_start': 1.0,       # Initial exploration rate
    'explore_end': 0.05,        # Final exploration rate
    'explore_decay': 10000,     # Steps over which exploration decays

    # Numerical stability
    'epsilon': 1e-6,
}


def validate_config(config: dict):
    """Validate and normalize RL-Cache configuration.

    :param config: Configuration dictionary to validate (modified in-place)
    """
    config['rollout_fragment_length'] = 1
    config['min_iter_time_s'] = 1
    config['train_batch_size'] = 1
    config['framework'] = 'torch'
    config['clip_actions'] = False

    # Validate neural network parameters
    if config.get('feature_dim', 6) <= 0:
        logger.info('Feature dimension is invalid. Setting default: 6.')
        config['feature_dim'] = 6
    if config.get('hidden_dim', 64) <= 0:
        logger.info('Hidden dimension is invalid. Setting default: 64.')
        config['hidden_dim'] = 64
    if config.get('num_layers', 5) <= 0:
        logger.info('Number of layers is invalid. Setting default: 5.')
        config['num_layers'] = 5

    # Validate training parameters
    if config.get('lr', 1e-3) <= 0:
        logger.info('Learning rate is invalid. Setting default: 1e-3.')
        config['lr'] = 1e-3
    if not (0 <= config.get('gamma', 0.99) <= 1):
        logger.info('Discount factor is invalid. Setting default: 0.99.')
        config['gamma'] = 0.99

    # Validate TTL values
    if config.get('admit_ttl', 100.0) <= 0:
        logger.info('Admit TTL is invalid. Setting default: 100.0.')
        config['admit_ttl'] = 100.0
    if config.get('reject_ttl', 0.1) <= 0:
        logger.info('Reject TTL is invalid. Setting default: 0.1.')
        config['reject_ttl'] = 0.1

    # Validate exploration parameters
    if not (0 <= config.get('explore_start', 1.0) <= 1):
        config['explore_start'] = 1.0
    if not (0 <= config.get('explore_end', 0.05) <= 1):
        config['explore_end'] = 0.05


class RLCacheAlgorithm(Algorithm):
    """RL-Cache algorithm implementation for RLlib.

    This algorithm uses a neural network policy (RLCachePolicy) to make cache admission
    decisions. Training follows the REINFORCE algorithm with entropy regularization.
    """

    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        """Return default configuration for RL-Cache.

        :return: Default configuration dictionary
        """
        logger.info('RLCacheAlgorithm.get_default_config()')
        config = DEFAULT_CONFIG.copy()
        # Set the default policy class using PolicySpec for RLlib 2.8.1+
        from ray.rllib.policy.policy import PolicySpec
        config['policies'] = {'default_policy': PolicySpec(policy_class=RLCachePolicy)}
        config['policy_mapping_fn'] = lambda agent_id, episode, worker, **kwargs: 'default_policy'
        return config

    @override(Algorithm)
    def setup(self, config):
        """Setup the algorithm."""
        super().setup(config)

    @override(Algorithm)
    def training_step(self):
        """Perform one training step.

        1. Collect a sample batch from the environment (one episode)
        2. Train the neural network policy using REINFORCE
        3. Return training metrics
        """
        logger.info('RLCacheAlgorithm.training_step()')

        # Get sample batch from local worker
        local_worker = self.workers.local_worker()
        sample_batch = local_worker.sample()

        # Train on the collected sample
        train_results = {}
        if sample_batch is not None:
            policy = self.get_policy('default_policy')
            learn_info = policy.learn_on_batch(sample_batch)
            if learn_info:
                train_results.update(learn_info)

        return train_results


# For backward compatibility
RLCacheAgent = RLCacheAlgorithm
