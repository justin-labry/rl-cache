# RL-Cache Experiment 0 Configuration
# Agent-level configuration for RL-Cache training and evaluation.
# Icarus simulator configuration is in a separate file (icarus_config.py)
# to avoid import issues when IcarusGym loads it via settings.read_from().

# Author: labry

import sys
import numpy as np

# Add IcarusGym and project root to Python path
sys.path.insert(0, '/home/labry/git/IcarusGym')
sys.path.insert(0, '/home/labry/git/rl-cache')

from rl_cache.evaluation.rl_cache_callbacks import RLCacheCallbacks


# ============================== EXECUTION MODE ==============================
# main.py --mode train : Train the neural network and save model to MODEL_PATH
# main.py --mode test  : Load model from MODEL_PATH and run evaluation only
# main.py --mode both  : Train → save model → evaluate (default, legacy behavior)
# ============================================================================

# ============================== TRAINING / EVALUATION SPLIT ==============================
NUM_TRAIN_EPISODES = 80         # Episodes for training (model learning)
NUM_EVAL_EPISODES = 20          # Episodes for evaluation (performance measurement)
NUM_EPISODES = NUM_TRAIN_EPISODES + NUM_EVAL_EPISODES   # Total episodes (= 100)
EPISODE_MEASUREMENT_BEGIN = NUM_TRAIN_EPISODES           # Callbacks start recording here
# =========================================================================================

# ============================== MODEL SAVE / LOAD ==============================
MODEL_PATH = 'model.pt'         # Path to save/load trained neural network model
# ===============================================================================

# ============================== ICARUSGYM ENVIRONMENT ==============================
N_CONTENTS = 100                                    # Number of contents (IDs: 1 to N_CONTENTS)
WORKLOAD_N_MEASURED = N_CONTENTS * 20               # Requests per episode (2000)
CACHE_RATIO = 0.1                                   # Cache size ratio

TTLSIM_CONFIG_PATH = 'icarus_config.py'             # Icarus simulator config (SEPARATE file)
ICARUS_OUTPUT = 'result.pickle'                     # Icarus simulation output
TTL_MAX = np.inf                                    # Maximum TTL value
CONTENT_MAX = N_CONTENTS                            # Maximum content ID (1-based)
CACHE_SIZE_MAX = int(N_CONTENTS * CACHE_RATIO)      # Maximum cache size (= 10)
# ===================================================================================

# ============================== RL-CACHE AGENT ==============================
N = N_CONTENTS                              # Number of contents (for policy)
B_0 = N * CACHE_RATIO                       # Cache size (= 10.0)

# Callbacks for performance measurement
CALLBACKS = RLCacheCallbacks
RESULT_OUTPUT_FILE_NAME = 'results_rl_cache'

# Neural network architecture (RL-Cache paper: 5 hidden layers)
FEATURE_DIM = 6         # Input feature dimension [size, logFreq, lambda, recency, ttl, hit]
HIDDEN_DIM = 64         # Hidden layer dimension
NUM_LAYERS = 5          # Number of hidden layers

# Training hyperparameters
LR = 1e-3               # Learning rate (Adam optimizer)
GAMMA = 0.99            # Discount factor for REINFORCE
ENTROPY_COEFF = 0.01    # Entropy regularization coefficient

# Feature normalization
MAX_LAMBDA = 10.0       # Max arrival rate for feature clipping

# TTL mapping for admit/reject decisions
ADMIT_TTL = 100.0       # TTL when admitting content to cache
REJECT_TTL = 0.1        # TTL when rejecting content (effectively not caching)

# Exploration schedule (epsilon-greedy with exponential decay)
# NOTE: During eval phase, exploration is forced to 0 regardless of these values
EXPLORE_START = 1.0     # Initial exploration rate (100% random at start)
EXPLORE_END = 0.05      # Final exploration rate (5% random)
EXPLORE_DECAY = 10000   # Decay half-life in steps (~5 episodes)
# ============================================================================
