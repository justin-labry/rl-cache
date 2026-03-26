"""
RL-Cache: Learning-Based Cache Admission for Content Delivery.

A neural network-based cache admission policy implemented as an RLlib policy for IcarusGym.
The policy uses a feedforward neural network to decide whether to admit content into the cache
by outputting a TTL value (high TTL = admit, low TTL = reject).

This implementation is based on:
    N. Beckmann et al., "RL-Cache: Learning-Based Cache Admission for Content Delivery,"
    (Earlier ideas from the RL-based caching literature)

Training strategy:
    - REINFORCE (Monte Carlo policy gradient) with episode-level hit rate as reward
    - Inference (compute_actions): torch.no_grad() → stores (features, action) pairs only
    - Learning (learn_on_batch): re-runs forward pass on stored features → computes loss → backprop
    - This avoids holding 2000+ computation graphs in memory during an episode

Feature vector (6-dim, all normalized):
    [0] size           - content size (as-is from IcarusGym, typically ~1.0)
    [1] log_frequency  - log(1 + access_count) to prevent unbounded growth
    [2] arrival_rate   - estimated λ = 1/τ, clipped to [0, max_lambda]
    [3] recency        - 1/(1 + time_since_last_access), bounded in [0, 1]
    [4] remaining_ttl  - remaining TTL normalized by admit_ttl
    [5] hit            - cache hit indicator (0 or 1)
"""

# Author: labry

import gymnasium as gym
import h5py
import json
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI, override


logger = logging.getLogger(__name__)


class RLCacheNetwork(nn.Module):
    """Feedforward neural network for cache admission decision.

    Architecture: input → [hidden_dim x num_layers with ELU] → sigmoid output
    Output: P(admit) in [0, 1]
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        """Constructor of RL-Cache neural network.

        :param input_dim: Dimension of input feature vector
        :param hidden_dim: Dimension of each hidden layer
        :param num_layers: Number of hidden layers
        """
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ELU())
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ELU())
        # Output layer: single sigmoid output for P(admit)
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Feature tensor of shape (batch_size, input_dim)
        :return: Admission probability tensor of shape (batch_size, 1)
        """
        return self.net(x)


class RLCachePolicy(Policy):
    """RL-Cache admission policy implemented as an RLlib Policy.

    This policy uses a neural network to decide cache admission for each content request.
    It maps IcarusGym observations to TTL actions:
        - High P(admit) -> large TTL (content stays in cache)
        - Low P(admit) -> minimal TTL (content not cached / quickly evicted)

    Training architecture (memory-efficient):
        - compute_actions(): inference only (no_grad), stores features + actions
        - learn_on_batch(): re-forward on stored features, compute REINFORCE loss, backprop
        - This avoids holding N computation graphs in memory during long episodes

    Reward design:
        - Per-step reward: cache hit = 1, miss = 0
        - Episode return: discounted sum of per-step rewards
        - Maximizing expected return = maximizing cache hit rate
    """

    @DeveloperAPI
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: dict):
        """Constructor of RL-Cache policy.

        :param observation_space: Observation space of IcarusGym environment
        :param action_space: Action space of IcarusGym environment (Tuple of TTL and cache_size)
        :param config: Configuration dictionary
        """

        # Neural network parameters
        self._feature_dim = config.get('feature_dim', 6)
        self._hidden_dim = config.get('hidden_dim', 64)
        self._num_layers = config.get('num_layers', 5)
        self._lr = config.get('lr', 1e-3)
        self._gamma = config.get('gamma', 0.99)             # Discount factor for future rewards
        self._entropy_coeff = config.get('entropy_coeff', 0.01)  # Entropy regularization coefficient
        self._epsilon = config.get('epsilon', 1e-6)          # Small value for numerical stability

        # TTL values for admit/reject decisions
        self._admit_ttl = config.get('admit_ttl', 100.0)     # TTL when admitting content
        self._reject_ttl = config.get('reject_ttl', 0.1)     # TTL when rejecting content

        # Feature normalization constants
        self._max_lambda = config.get('max_lambda', 10.0)    # Max arrival rate for clipping

        # Cache configuration
        # A trick: In the IcarusGym environment, index begins from 1 instead of 0. Thus, we should use index n for n
        # contents
        self._n = config['n'] + 1       # Number of contents (IcarusGym uses 1-indexed content IDs)
        self._b_0 = config['b_0']       # Cache size

        # Build neural network
        self._device = torch.device('cpu')
        self._net = RLCacheNetwork(self._feature_dim, self._hidden_dim, self._num_layers).to(self._device)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)

        # Per-content statistics (maintained across episodes for feature extraction)
        self._access_counts = np.zeros(self._n, dtype=np.int64)        # Total access count per content
        self._last_access_time = np.zeros(self._n, dtype=np.float64)   # Last access time per content
        self._inter_arrival = np.zeros(self._n, dtype=np.float64)      # Estimated inter-arrival time
        self._arrival_nums = np.zeros(self._n, dtype=np.int64)         # Arrival count for lambda estimation
        self._tau_sums = np.zeros(self._n, dtype=np.float64)           # Sum of inter-arrival times

        # Episode buffers for REINFORCE (memory-efficient: store features + actions, not computation graphs)
        self._episode_features = []     # List of feature vectors (np.ndarray)
        self._episode_actions = []      # List of actions taken (0=reject, 1=admit)
        self._episode_rewards = []      # List of rewards (hit=1, miss=0)

        # Tracking
        self._prev_env_time = 0.0
        self._episode_count = 0
        self._total_steps = 0

        # Exploration: epsilon-greedy schedule
        self._explore_start = config.get('explore_start', 1.0)    # Initial exploration rate
        self._explore_end = config.get('explore_end', 0.05)       # Final exploration rate
        self._explore_decay = config.get('explore_decay', 10000)  # Steps to decay exploration

        # Eval mode: when True, no exploration and no learning
        self._eval_mode = False

        super().__init__(observation_space, action_space, config)

    def set_eval_mode(self, eval_mode: bool):
        """Switch between training and evaluation mode.

        In eval mode:
            - Exploration is OFF (always greedy: action = argmax P(admit))
            - learn_on_batch() is a no-op (neural network weights frozen)
            - Episode buffers are still collected for metrics only

        :param eval_mode: True for evaluation, False for training
        """
        self._eval_mode = eval_mode
        if eval_mode:
            self._net.eval()    # Switch PyTorch layers to eval mode (affects dropout, batchnorm)
            logger.info(f'Policy switched to EVAL mode (episode {self._episode_count}). '
                        f'No exploration, no learning.')
        else:
            self._net.train()   # Switch PyTorch layers to train mode
            logger.info(f'Policy switched to TRAIN mode (episode {self._episode_count}).')

    def _get_explore_rate(self) -> float:
        """Compute current exploration rate (epsilon) using exponential decay.

        :return: Current exploration rate
        """
        return self._explore_end + (self._explore_start - self._explore_end) * \
            math.exp(-self._total_steps / max(self._explore_decay, 1))

    def _extract_features(self, obs: np.ndarray) -> tuple:
        """Extract normalized feature vector from IcarusGym observation.

        IcarusGym observation format:
            obs[0] = env_time     (simulation time)
            obs[1] = content_id   (1-indexed)
            obs[2] = weight       (content weight/importance)
            obs[3] = size         (content size)
            obs[4] = remaining_ttl (remaining TTL if cached, else negative)
            obs[5] = hit          (1.0 if cache hit, 0.0 if miss)

        All features are normalized to reasonable ranges to ensure stable neural network training:
            - size: as-is (typically ~1.0 in IcarusGym)
            - frequency: log(1 + count) to prevent unbounded growth
            - arrival_rate: clipped lambda, bounded in [0, max_lambda]
            - recency: 1/(1 + delta_t), naturally bounded in [0, 1]
            - remaining_ttl: normalized by admit_ttl, bounded in [0, 1]
            - hit: binary indicator, already in {0, 1}

        :param obs: Raw observation from IcarusGym
        :return: Tuple of (content_id, normalized_feature_vector)
        """
        env_time = obs[0]
        i = int(obs[1])         # Content ID (1-indexed)
        w = obs[2]              # Weight
        s = obs[3]              # Size
        r = obs[4]              # Remaining TTL
        hit = obs[5]            # Cache hit indicator

        # Handle episode reset (env_time goes backward)
        if env_time < self._prev_env_time:
            self._prev_env_time = 0.0
            self._last_access_time = np.zeros(self._n, dtype=np.float64)

        # Update per-content statistics
        self._access_counts[i] += 1
        self._arrival_nums[i] += 1

        # Compute inter-arrival time (tau)
        if self._last_access_time[i] > 0:
            tau = env_time - self._last_access_time[i]
            if tau > 0:
                self._tau_sums[i] += tau
                self._inter_arrival[i] = self._tau_sums[i] / float(self._arrival_nums[i])

        # Recency: time since last access (computed BEFORE updating last_access_time)
        time_since_last = env_time - self._last_access_time[i] if self._last_access_time[i] > 0 else 0.0
        self._last_access_time[i] = env_time

        # Estimate arrival rate (lambda = 1/tau), clipped to prevent extreme values
        if self._inter_arrival[i] > self._epsilon:
            lambda_est = min(1.0 / self._inter_arrival[i], self._max_lambda)
        else:
            lambda_est = 0.0

        # Build NORMALIZED feature vector (all features in reasonable ranges for NN training)
        features = np.array([
            s,                                                          # Size (~1.0, stable)
            math.log1p(float(self._access_counts[i])),                  # log(1+count), grows slowly
            lambda_est / self._max_lambda,                              # Normalized arrival rate [0, 1]
            1.0 / (1.0 + time_since_last),                             # Recency [0, 1]
            min(max(r, 0.0) / max(self._admit_ttl, 1.0), 1.0),        # Normalized remaining TTL [0, 1]
            float(hit),                                                 # Hit indicator {0, 1}
        ], dtype=np.float32)

        self._prev_env_time = env_time
        return i, features

    @override(Policy)
    def compute_actions(self, obs_batch: list or np.ndarray,
                        state_batches: list or np.ndarray = None,
                        prev_action_batch: list or np.ndarray = None,
                        prev_reward_batch: list or np.ndarray = None,
                        info_batch: list = None,
                        episodes: list = None,
                        explore: bool = None,
                        timestep: int = None,
                        **kwargs) -> tuple:
        """Compute cache admission actions for a batch of observations.

        IMPORTANT: This method runs in INFERENCE mode (torch.no_grad).
        We do NOT store computation graphs here. Instead, we store (features, action) pairs.
        The actual gradient computation happens in learn_on_batch() via re-forward pass.

        For each observation:
        1. Extract normalized features from IcarusGym observation
        2. Feed features into neural network (no_grad) to get P(admit)
        3. Sample admit/reject decision (epsilon-greedy + Bernoulli)
        4. Store (features, action, reward) for later training
        5. Map decision to TTL action for IcarusGym

        :param obs_batch: Batch of observations from IcarusGym
        :return: Tuple of (actions, state_outs, info)
        """
        actions_ = []
        if info_batch is None:
            info_batch = [{}] * len(obs_batch)

        for obs, info in zip(obs_batch, info_batch):
            i, features = self._extract_features(obs)
            hit = obs[5]

            # Neural network forward pass (INFERENCE ONLY - no gradient tracking)
            feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(self._device)
            with torch.no_grad():
                p_admit = self._net(feat_tensor).squeeze().item()

            # Clamp probability to avoid log(0)
            p_admit = max(self._epsilon, min(1.0 - self._epsilon, p_admit))

            if self._eval_mode:
                # EVAL MODE: greedy action (no exploration, no randomness)
                action = 1 if p_admit >= 0.5 else 0
            else:
                # TRAIN MODE: epsilon-greedy exploration
                eps = self._get_explore_rate()
                if np.random.random() < eps:
                    # Random action for exploration
                    action = np.random.randint(0, 2)
                else:
                    # Sample from Bernoulli distribution
                    action = 1 if np.random.random() < p_admit else 0

            # Store (features, action, reward) for REINFORCE training later
            # NOTE: No computation graph stored here - memory efficient!
            self._episode_features.append(features.copy())
            self._episode_actions.append(action)
            self._episode_rewards.append(float(hit))

            # Map admit/reject to TTL action for IcarusGym
            if action == 1:  # Admit
                ttl = self._admit_ttl
            else:  # Reject
                ttl = self._reject_ttl

            actions_.append((ttl, self._b_0))
            self._total_steps += 1

        # Return actions in batched format (same as DehghanCachePolicy)
        if len(actions_) > 0:
            ttl_values = np.array([a[0] for a in actions_], dtype=np.float64)
            cache_size_values = np.array([a[1] for a in actions_], dtype=np.int64)
            actions = (ttl_values, cache_size_values)
        else:
            actions = (np.array([], dtype=np.float64), np.array([], dtype=np.int64))
        return actions, [], {}

    @override(Policy)
    def learn_on_batch(self, sample_batch: list):
        """Train the neural network using REINFORCE on the collected episode data.

        Training strategy (memory-efficient re-forward approach):
            1. Batch all stored features into a single tensor
            2. Single forward pass through the network (WITH gradient tracking)
            3. Compute log probabilities for the actions actually taken
            4. Compute discounted returns G_t with baseline subtraction
            5. REINFORCE loss: L = -mean(log pi(a_t|s_t) * G_t)
            6. Entropy bonus: L -= entropy_coeff * mean(H(pi))
            7. Backpropagate and update network weights

        This is much more memory-efficient than storing 2000+ computation graphs
        during compute_actions(). Only ONE forward pass graph exists during training.

        :param sample_batch: SampleBatch from RLlib rollout worker
        :return: Dictionary with training metrics
        """
        n_steps = len(self._episode_features)
        if n_steps == 0:
            return {}

        # In eval mode: compute metrics only, do NOT update network weights
        if self._eval_mode:
            hit_rate = np.mean(self._episode_rewards) if n_steps > 0 else 0.0
            self._episode_features = []
            self._episode_actions = []
            self._episode_rewards = []
            self._episode_count += 1
            logger.info(f'[EVAL] Episode {self._episode_count}: hit_rate={hit_rate:.4f} (no learning)')
            return {
                'hit_rate': hit_rate,
                'policy_loss': 0.0,
                'entropy': 0.0,
                'mean_p_admit': 0.0,
                'explore_rate': 0.0,
                'episode': self._episode_count,
                'eval_mode': True
            }

        # ---- Step 1: Compute discounted returns ----
        returns = np.zeros(n_steps, dtype=np.float32)
        G = 0.0
        for t in range(n_steps - 1, -1, -1):
            G = self._episode_rewards[t] + self._gamma * G
            returns[t] = G

        # Normalize returns (baseline subtraction for variance reduction)
        returns_tensor = torch.FloatTensor(returns).to(self._device)
        if n_steps > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + self._epsilon)

        # ---- Step 2: Batch forward pass (WITH gradient tracking) ----
        features_batch = torch.FloatTensor(np.array(self._episode_features)).to(self._device)  # (N, feature_dim)
        actions_batch = torch.FloatTensor(self._episode_actions).to(self._device)               # (N,)

        # Single forward pass through the network - only ONE computation graph!
        p_admit_batch = self._net(features_batch).squeeze(-1)   # (N,)
        p_admit_batch = torch.clamp(p_admit_batch, self._epsilon, 1.0 - self._epsilon)

        # ---- Step 3: Compute log probabilities for actions taken ----
        # log pi(a=1|s) = log(p),  log pi(a=0|s) = log(1-p)
        log_probs = actions_batch * torch.log(p_admit_batch) + \
                    (1.0 - actions_batch) * torch.log(1.0 - p_admit_batch)   # (N,)

        # ---- Step 4: REINFORCE policy gradient loss ----
        # L = -E[log pi(a|s) * G]  (negative because we minimize loss)
        policy_loss = -(log_probs * returns_tensor).mean()

        # ---- Step 5: Entropy regularization (encourages exploration) ----
        # H(Bernoulli(p)) = -p*log(p) - (1-p)*log(1-p)
        entropy = -(p_admit_batch * torch.log(p_admit_batch) +
                     (1.0 - p_admit_batch) * torch.log(1.0 - p_admit_batch))  # (N,)
        entropy_bonus = entropy.mean()

        # Total loss = policy gradient loss - entropy bonus
        total_loss = policy_loss - self._entropy_coeff * entropy_bonus

        # ---- Step 6: Backpropagation ----
        self._optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
        self._optimizer.step()

        # ---- Step 7: Compute metrics ----
        hit_rate = np.mean(self._episode_rewards)
        mean_p_admit = p_admit_batch.mean().item()

        # Clear episode buffers
        self._episode_features = []
        self._episode_actions = []
        self._episode_rewards = []
        self._episode_count += 1

        logger.info(f'Episode {self._episode_count}: hit_rate={hit_rate:.4f}, '
                    f'loss={total_loss.item():.4f}, entropy={entropy_bonus.item():.4f}, '
                    f'mean_p_admit={mean_p_admit:.4f}, explore={self._get_explore_rate():.4f}')

        return {
            'hit_rate': hit_rate,
            'policy_loss': policy_loss.item(),
            'entropy': entropy_bonus.item(),
            'mean_p_admit': mean_p_admit,
            'explore_rate': self._get_explore_rate(),
            'episode': self._episode_count
        }

    @override(Policy)
    def get_weights(self) -> dict:
        """Returns neural network weights and policy state.

        :return: Dictionary containing network state_dict and statistics
        """
        return {
            'net_state_dict': {k: v.cpu().numpy() for k, v in self._net.state_dict().items()},
            'optimizer_state': None,  # Optimizer state is complex; skip for simplicity
            'access_counts': self._access_counts.copy(),
            'inter_arrival': self._inter_arrival.copy(),
            'episode_count': self._episode_count,
            'total_steps': self._total_steps
        }

    @override(Policy)
    def set_weights(self, weights: dict):
        """Sets neural network weights and policy state.

        :param weights: Dictionary from get_weights()
        """
        state_dict = {k: torch.FloatTensor(v) for k, v in weights['net_state_dict'].items()}
        self._net.load_state_dict(state_dict)
        self._access_counts = weights['access_counts']
        self._inter_arrival = weights['inter_arrival']
        self._episode_count = weights['episode_count']
        self._total_steps = weights['total_steps']

    def save_model(self, path: str):
        """Save trained neural network model to a .pt file.

        Saves everything needed to resume training or run inference:
            - Neural network weights (state_dict)
            - Optimizer state (for resuming training)
            - Per-content statistics (access counts, inter-arrival times)
            - Training progress (episode count, total steps)
            - Architecture config (to reconstruct the network)

        :param path: File path to save the model (e.g., 'model.pt')
        """
        checkpoint = {
            # Architecture config (needed to reconstruct network for loading)
            'feature_dim': self._feature_dim,
            'hidden_dim': self._hidden_dim,
            'num_layers': self._num_layers,

            # Neural network weights
            'net_state_dict': self._net.state_dict(),

            # Optimizer state (for resuming training)
            'optimizer_state_dict': self._optimizer.state_dict(),

            # Per-content statistics
            'access_counts': self._access_counts,
            'last_access_time': self._last_access_time,
            'inter_arrival': self._inter_arrival,
            'arrival_nums': self._arrival_nums,
            'tau_sums': self._tau_sums,

            # Training progress
            'episode_count': self._episode_count,
            'total_steps': self._total_steps,

            # Config values needed for inference
            'n': self._n,
            'b_0': self._b_0,
            'lr': self._lr,
            'gamma': self._gamma,
            'admit_ttl': self._admit_ttl,
            'reject_ttl': self._reject_ttl,
            'max_lambda': self._max_lambda,
        }
        torch.save(checkpoint, path)
        logger.info(f'Model saved to {path} (episode {self._episode_count}, '
                    f'{self._total_steps} total steps)')

    def load_model(self, path: str):
        """Load a trained neural network model from a .pt file.

        Restores neural network weights, optimizer state, and per-content statistics.
        After loading, the policy can immediately run inference (test mode) or
        resume training (train mode).

        :param path: File path to load the model from (e.g., 'model.pt')
        """
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        # Restore neural network weights
        self._net.load_state_dict(checkpoint['net_state_dict'])

        # Restore optimizer state (for resuming training)
        if 'optimizer_state_dict' in checkpoint:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore per-content statistics
        self._access_counts = checkpoint['access_counts']
        self._last_access_time = checkpoint.get('last_access_time',
                                                 np.zeros(self._n, dtype=np.float64))
        self._inter_arrival = checkpoint['inter_arrival']
        self._arrival_nums = checkpoint.get('arrival_nums',
                                             np.zeros(self._n, dtype=np.int64))
        self._tau_sums = checkpoint.get('tau_sums',
                                         np.zeros(self._n, dtype=np.float64))

        # Restore training progress
        self._episode_count = checkpoint['episode_count']
        self._total_steps = checkpoint['total_steps']

        logger.info(f'Model loaded from {path} (episode {self._episode_count}, '
                    f'{self._total_steps} total steps)')

    @override(Policy)
    def export_model(self, export_dir: str):
        """Exports neural network model to HDF5 file.

        :param export_dir: Path to save HDF5 file
        """
        with h5py.File(export_dir, 'w') as f:
            grp = f.create_group('rl_cache_policy')
            for k, v in self._net.state_dict().items():
                grp.create_dataset(f'net/{k}', data=v.cpu().numpy())
            grp.create_dataset('access_counts', data=self._access_counts)
            grp.create_dataset('inter_arrival', data=self._inter_arrival)
            grp.attrs['feature_dim'] = self._feature_dim
            grp.attrs['hidden_dim'] = self._hidden_dim
            grp.attrs['num_layers'] = self._num_layers
            grp.attrs['lr'] = self._lr
            grp.attrs['n'] = self._n
            grp.attrs['b_0'] = self._b_0
            grp.attrs['admit_ttl'] = self._admit_ttl
            grp.attrs['reject_ttl'] = self._reject_ttl
            grp.attrs['episode_count'] = self._episode_count
            grp.attrs['total_steps'] = self._total_steps

    @override(Policy)
    def export_checkpoint(self, export_dir: str):
        """Exports policy checkpoint as JSON.

        :param export_dir: Path to save JSON file
        """
        checkpoint = {
            'feature_dim': self._feature_dim,
            'hidden_dim': self._hidden_dim,
            'num_layers': self._num_layers,
            'lr': self._lr,
            'n': self._n,
            'b_0': self._b_0,
            'admit_ttl': self._admit_ttl,
            'reject_ttl': self._reject_ttl,
            'episode_count': self._episode_count,
            'total_steps': self._total_steps
        }
        with open(export_dir, 'w') as f:
            json.dump(checkpoint, f, indent=4)

    @override(Policy)
    def import_model_from_h5(self, import_file: str):
        """Imports neural network model from HDF5 file.

        :param import_file: Path to HDF5 file
        """
        with h5py.File(import_file, 'r') as f:
            grp = f['rl_cache_policy']
            state_dict = {}
            for k in grp['net'].keys():
                state_dict[k] = torch.FloatTensor(grp[f'net/{k}'][:])
            self._net.load_state_dict(state_dict)
            self._access_counts = grp['access_counts'][:]
            self._inter_arrival = grp['inter_arrival'][:]
            self._episode_count = int(grp.attrs['episode_count'])
            self._total_steps = int(grp.attrs['total_steps'])

    @override(Policy)
    def compute_gradients(self, postprocessed_batch):
        pass

    @override(Policy)
    def apply_gradients(self, gradients):
        pass

    @override(Policy)
    def compute_log_likelihoods(self, actions, obs_batch, state_batches=None, prev_action_batch=None,
                                prev_reward_batch=None):
        pass

    @property
    def n(self):
        return self._n

    @property
    def net(self):
        return self._net
