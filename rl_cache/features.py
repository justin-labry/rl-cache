"""RL-Cache object features (Kirilin et al., NetAI 2019, Table 1).

The eight features describe a requested object purely from the REQUEST STREAM
(size, frequency, recency in time and in request count, plus their composites).
Crucially, none of them depend on the cache state or on admission decisions, so:

  * during Monte Carlo training the whole trace's features are precomputed ONCE
    and reused across all m decision samples (the network's admit probabilities
    are fixed per window), and
  * the same extractor runs online during IcarusGym evaluation.

A single ``FeatureExtractor`` implementation is shared by both paths so the
network sees identical inputs at train and eval time.

Feature vector (8-dim, all normalized to ~[0, 1]):
    [0] log_size      log(size) / log(running max size)
    [1] frequency     access_count(j) / total_requests_so_far
    [2] recency       1 / (1 + temporal recency r_j)            (r_j in seconds)
    [3] recency_ema   1 / (1 + rho_j)   rho_j = EMA of r_j
    [4] ord_recency   1 / (1 + ordinal recency d_j)             (d_j in #requests)
    [5] ord_rec_ema   1 / (1 + delta_j) delta_j = EMA of d_j
    [6] freq_per_size frequency / max(log_size, eps)            (clipped to [0,1])
    [7] freq_x_size   frequency * log_size

Maps to Table 1: s_j, f_j, r_j, rho_j, d_j, delta_j, f_j/s_j, f_j*s_j.
"""

# Author: labry

import math
import numpy as np

FEATURE_DIM = 8


class FeatureExtractor:
    """Incremental, order-sensitive extractor for RL-Cache object features.

    Call :meth:`step` once per request, in arrival order. State (counts, last
    access time/index, EMAs, running max size) carries across calls. Use
    :meth:`reset` between independent episodes.

    :param n_contents: Number of distinct content IDs (1-indexed; arrays sized n+1).
    :param ema_alpha: Smoothing factor for the recency EMAs (rho, delta).
    :param eps: Numerical-stability floor.
    """

    def __init__(self, n_contents, ema_alpha=0.5, eps=1e-6):
        self._n = int(n_contents) + 1      # 1-indexed content IDs
        self._alpha = float(ema_alpha)
        self._eps = float(eps)
        self.reset()

    def reset(self):
        self._count = np.zeros(self._n, dtype=np.int64)
        self._last_time = np.full(self._n, -1.0, dtype=np.float64)
        self._last_ord = np.full(self._n, -1, dtype=np.int64)
        self._rho = np.zeros(self._n, dtype=np.float64)     # EMA of temporal recency
        self._delta = np.zeros(self._n, dtype=np.float64)   # EMA of ordinal recency
        self._total = 0                                      # total requests so far
        self._max_size = 1.0                                 # running max size

    def step(self, time, content, size):
        """Update state with one request and return its 8-dim feature vector.

        :param time: Arrival time (seconds).
        :param content: Content ID (1-indexed).
        :param size: Content size (bytes).
        :return: np.ndarray shape (FEATURE_DIM,), dtype float32.
        """
        i = int(content)
        s = max(float(size), self._eps)
        if s > self._max_size:
            self._max_size = s

        self._total += 1
        self._ord_now = self._total                          # 1-based request index

        seen = self._last_time[i] >= 0.0
        r = (time - self._last_time[i]) if seen else 0.0     # temporal recency
        d = (self._total - self._last_ord[i]) if seen else 0  # ordinal recency

        # Exponential smoothing of recency (initialize to the first observation).
        if seen:
            self._rho[i] = self._alpha * r + (1.0 - self._alpha) * self._rho[i]
            self._delta[i] = self._alpha * d + (1.0 - self._alpha) * self._delta[i]
        else:
            self._rho[i] = r
            self._delta[i] = d

        self._count[i] += 1
        self._last_time[i] = time
        self._last_ord[i] = self._total

        freq = self._count[i] / self._total                  # fraction of requests so far
        log_size = math.log(s) / max(math.log(self._max_size), self._eps)  # [0,1]

        freq_per_size = min(freq / max(log_size, self._eps), 1.0)
        freq_x_size = freq * log_size

        return np.array([
            log_size,
            freq,
            1.0 / (1.0 + r),
            1.0 / (1.0 + self._rho[i]),
            1.0 / (1.0 + d),
            1.0 / (1.0 + self._delta[i]),
            freq_per_size,
            freq_x_size,
        ], dtype=np.float32)


def extract_all(trace, n_contents, ema_alpha=0.5, eps=1e-6):
    """Precompute features for an entire trace (batch helper for training).

    :param trace: iterable of (time, content, size) in arrival order.
    :param n_contents: Number of distinct content IDs.
    :return: np.ndarray shape (len(trace), FEATURE_DIM), dtype float32.
    """
    fx = FeatureExtractor(n_contents, ema_alpha=ema_alpha, eps=eps)
    feats = np.empty((len(trace), FEATURE_DIM), dtype=np.float32)
    for k, (time, content, size) in enumerate(trace):
        feats[k] = fx.step(time, content, size)
    return feats
