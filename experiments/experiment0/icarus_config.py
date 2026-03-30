# Icarus Simulator Configuration for RL-Cache Experiment 0
#
# This file is read by IcarusGym's settings.read_from().
# It must be self-contained (no project-specific imports) because it is loaded
# by the Icarus simulator process independently.
#
# Topology: PATH (3 nodes: source - router - receiver)
# Workload: NONSTATIONARY Zipf (alpha=0.8, 1000 contents, n_phases=4)
# Cache: TTL-based (reset mode), 10% of contents (cache_size=100)

# Author: labry

import sys
sys.path.insert(0, '/home/labry/git/IcarusGym')

from collections import deque
from icarus.registry import register_workload
from icarus.util import Tree


# ============================== NONSTATIONARY WORKLOAD ==============================
@register_workload('NONSTATIONARY')
class NonstationaryWorkload:
    """Non-stationary Zipf workload with popularity shifts.

    The episode is divided into n_phases equal segments. At each phase boundary,
    the mapping from Zipf rank to content ID is randomly permuted, simulating
    a sudden change in content popularity (e.g., trending content rotation).

    Within each phase, requests follow a standard Zipf distribution. Between
    phases, the set of popular content changes, forcing cache policies to adapt.
    """

    def __init__(self, topology, n_contents, alpha, n_phases=4,
                 rate=1.0, n_warmup=0, n_measured=10000, seed=None,
                 size_min=1.0, size_max=1.0, **kwargs):
        """Constructor.

        :param topology: Network topology.
        :param n_contents: Number of contents (IDs: 1 to n_contents).
        :param alpha: Zipf skewness parameter.
        :param n_phases: Number of popularity phases per episode.
        :param rate: Request arrival rate.
        :param n_warmup: Warm-up requests.
        :param n_measured: Measured requests.
        :param seed: Random seed.
        :param size_min: Minimum content size (default 1.0 for backward compat).
        :param size_max: Maximum content size. When size_min == size_max, all
            contents have equal size (backward compatible). When different,
            per-content sizes are drawn from a log-uniform distribution
            [size_min, size_max] to model the heavy-tailed size distribution
            observed in real CDN traces (RL-Cache paper: 10^2 to 10^8 bytes).
        """
        # Import inside methods to survive settings.read_from() cleanup
        # (exec() + delete non-uppercase names removes module-level references)
        from icarus.tools.stats import TruncatedZipfDist
        import numpy as np

        if alpha < 0:
            raise ValueError('alpha must be positive')
        if n_phases < 1:
            raise ValueError('n_phases must be >= 1')
        self.receivers = [
            v for v in topology.nodes() if topology.node[v]['stack'][0] == 'receiver'
        ]
        self.zipf = TruncatedZipfDist(alpha, n_contents)
        self.n_contents = n_contents
        self.contents = range(1, n_contents + 1)
        self.n_phases = n_phases
        self.rate = rate
        self.n_warmup = n_warmup
        self.n_measured = n_measured
        self.seed = seed

        # Generate per-content sizes (fixed for the lifetime of the workload).
        # Log-uniform distribution produces heavy-tailed sizes similar to
        # real CDN traces where object sizes span multiple orders of magnitude.
        self.size_min = size_min
        self.size_max = size_max
        rng_size = np.random.RandomState(seed)
        if size_min < size_max:
            log_sizes = rng_size.uniform(np.log(size_min), np.log(size_max),
                                         size=n_contents)
            self.content_sizes = np.exp(log_sizes)  # 0-indexed: content_sizes[i] = size of content (i+1)
        else:
            self.content_sizes = np.full(n_contents, size_min)

    def __iter__(self):
        import random
        import numpy as np

        rng = random.Random(self.seed)
        np_rng = np.random.RandomState(self.seed)

        total_requests = self.n_warmup + self.n_measured
        phase_len = total_requests // self.n_phases

        # Generate a random permutation for each phase
        # permutation[rank] = content_id (1-indexed)
        permutations = []
        for _ in range(self.n_phases):
            perm = np_rng.permutation(self.n_contents) + 1  # 1-indexed
            permutations.append(perm)

        req_counter = 0
        t_event = 0.0
        while req_counter < total_requests:
            t_event += rng.expovariate(self.rate)
            receiver = rng.choice(self.receivers)

            # Determine current phase
            phase = min(req_counter // phase_len, self.n_phases - 1)

            # Sample Zipf rank (1-indexed), map to content ID via phase permutation
            rank = int(self.zipf.rv()) - 1  # 0-indexed rank
            content = int(permutations[phase][rank])

            log = req_counter >= self.n_warmup
            content_size = float(self.content_sizes[content - 1])  # 1-indexed → 0-indexed
            event = {'receiver': receiver, 'content': content, 'log': log,
                     'size': content_size}
            yield (t_event, event)
            req_counter += 1
# ===================================================================================


# ============================== SIMULATION PARAMETERS ==============================
N_CONTENTS = 1000                       # Number of contents (IDs: 1 to N_CONTENTS)
WORKLOAD_N_WARM_UP = 0                  # Warm-up requests (0 = no warm-up)
WORKLOAD_N_MEASURED = N_CONTENTS * 20   # Measured requests per episode (2000)
WORKLOAD_NAME = 'NONSTATIONARY'         # Non-stationary Zipf workload
ALPHA = 0.8                             # Zipf skewness parameter
N_PHASES = 4                            # Number of popularity phases per episode
RATE = 1.                               # Request arrival rate
CACHE_RATIO = 0.1                       # Cache size / number of contents
TAU = 1.                                # Sliding time window for hit probability
DEFAULT_WEIGHT = 1.                     # Default content weight
IS_RESET = True                         # Reset TTL on cache hit (True = LRU-like)
# Content size distribution (log-uniform).
# Set SIZE_MIN == SIZE_MAX == 1.0 for equal-size (backward compatible).
# Set SIZE_MIN < SIZE_MAX for variable sizes (e.g., 100 to 1e6 bytes).
SIZE_MIN = 100.0                        # Minimum content size (bytes)
SIZE_MAX = 1000000.0                    # Maximum content size (bytes, 10^6)
# DEFAULT_SIZE: geometric mean of size range.  In size-aware mode, cache
# capacity in bytes = cache_slots * DEFAULT_SIZE, so using the geometric
# mean ensures the cache can hold ~CACHE_RATIO fraction of catalog by bytes.
import math as _math
DEFAULT_SIZE = _math.sqrt(SIZE_MIN * SIZE_MAX) if SIZE_MIN < SIZE_MAX else SIZE_MIN
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

# Workload: Non-stationary Zipf distribution (popularity shifts every phase)
experiment['workload']['name'] = WORKLOAD_NAME
experiment['workload']['n_contents'] = N_CONTENTS
experiment['workload']['n_warmup'] = WORKLOAD_N_WARM_UP
experiment['workload']['n_measured'] = WORKLOAD_N_MEASURED
experiment['workload']['rate'] = RATE
experiment['workload']['alpha'] = ALPHA
experiment['workload']['n_phases'] = N_PHASES
experiment['workload']['size_min'] = SIZE_MIN
experiment['workload']['size_max'] = SIZE_MAX

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
# Size-aware caching: when True, cache capacity is tracked in bytes
# (max_bytes = cache_slots * default_size) instead of item count.
# Enable only when SIZE_MIN < SIZE_MAX (variable content sizes).
experiment['cache_policy']['size_aware'] = (SIZE_MIN < SIZE_MAX)

experiment['desc'] = ('RL-Cache experiment 0 - '
                      'strategy: %s / cache policy: %s' %
                      (str(experiment['strategy']['name']),
                       str(experiment['cache_policy']['name'])))

EXPERIMENT_QUEUE.append(experiment)
# ==================================================================================
