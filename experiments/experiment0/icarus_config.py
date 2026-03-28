# Icarus Simulator Configuration for RL-Cache Experiment 0
#
# This file is read by IcarusGym's settings.read_from().
# It must be self-contained (no project-specific imports) because it is loaded
# by the Icarus simulator process independently.
#
# Topology: PATH (3 nodes: source - router - receiver)
# Workload: STATIONARY Zipf (alpha=0.8, 100 contents, rate=1.0)
# Cache: TTL-based (reset mode), 10% of contents (cache_size=10)

# Author: labry

import sys
sys.path.insert(0, '/home/labry/git/IcarusGym')

from collections import deque
from icarus.util import Tree

# ============================== SIMULATION PARAMETERS ==============================
N_CONTENTS = 1000                       # Number of contents (IDs: 1 to N_CONTENTS)
WORKLOAD_N_WARM_UP = 0                  # Warm-up requests (0 = no warm-up)
WORKLOAD_N_MEASURED = N_CONTENTS * 20   # Measured requests per episode (2000)
WORKLOAD_NAME = 'STATIONARY'            # Stationary Zipf workload
ALPHA = 0.8                             # Zipf skewness parameter
RATE = 1.                               # Request arrival rate
CACHE_RATIO = 0.1                       # Cache size / number of contents
TAU = 1.                                # Sliding time window for hit probability
DEFAULT_WEIGHT = 1.                     # Default content weight
DEFAULT_SIZE = 1.                       # Default content size
IS_RESET = True                         # Reset TTL on cache hit (True = LRU-like)
# ===================================================================================

# ============================== GENERAL SETTINGS ==============================
LOG_LEVEL = 'WARNING'                   # Icarus log level (WARNING to reduce noise)
PARALLEL_EXECUTION = False              # Must be False for IcarusGym
RESULTS_FORMAT = 'PICKLE'               # Output format
N_REPLICATIONS = 1                      # Number of replications per experiment
CACHING_GRANULARITY = 'OBJECT'          # Cache at object level
DATA_COLLECTORS = ['CACHE_HIT_RATIO']   # Collect cache hit ratio data
# ==============================================================================

# ============================== EXPERIMENT DEFINITION ==============================
EXPERIMENT_QUEUE = deque()

experiment = Tree()

# Network topology: PATH with 3 nodes (source → router → receiver)
experiment['topology']['name'] = 'PATH'
experiment['topology']['n'] = 3
experiment['topology']['delay'] = 10.

# Workload: Stationary Zipf distribution
experiment['workload']['name'] = WORKLOAD_NAME
experiment['workload']['n_contents'] = N_CONTENTS
experiment['workload']['n_warmup'] = WORKLOAD_N_WARM_UP
experiment['workload']['n_measured'] = WORKLOAD_N_MEASURED
experiment['workload']['rate'] = RATE
experiment['workload']['alpha'] = ALPHA

# Content placement: uniform across sources
experiment['content_placement']['name'] = 'UNIFORM'

# Cache placement: uniform across routers, 10% of content catalog
experiment['cache_placement']['name'] = 'UNIFORM'
experiment['cache_placement']['network_cache'] = CACHE_RATIO

# Strategy: IcarusGym LCE (Leave Copy Everywhere) with TTL control
experiment['strategy']['name'] = 'ICARUSGYM_LCE'
experiment['strategy']['tau'] = TAU
experiment['strategy']['default_weight'] = DEFAULT_WEIGHT
experiment['strategy']['default_size'] = DEFAULT_SIZE
experiment['strategy']['content_max'] = N_CONTENTS

# Cache policy: IcarusGym TTL (controlled by RL agent)
experiment['cache_policy']['name'] = 'ICARUSGYM_TTL'
experiment['cache_policy']['is_reset'] = IS_RESET
experiment['cache_policy']['epsilon_b'] = 0.2
experiment['cache_policy']['default_weight'] = DEFAULT_WEIGHT
experiment['cache_policy']['default_size'] = DEFAULT_SIZE

experiment['desc'] = ('RL-Cache experiment 0 - '
                      'strategy: %s / cache policy: %s' %
                      (str(experiment['strategy']['name']),
                       str(experiment['cache_policy']['name'])))

EXPERIMENT_QUEUE.append(experiment)
# ==================================================================================
