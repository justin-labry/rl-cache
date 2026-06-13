"""Standalone TTL-cache simulator that faithfully replicates IcarusGym's
``TtlCache`` (registered as ``ICARUSGYM_TTL``) driven by the ``ICARUSGYM_LCE``
strategy on a PATH topology with a single router cache.

Why this exists
---------------
RL-Cache's Monte Carlo training (Kirilin et al., NetAI 2019) must replay the
SAME window of requests many times with DIFFERENT admit/reject decisions and
measure the resulting hit rate. IcarusGym drives requests through an inverted,
blocking event loop (the simulator pulls actions per request via
``IcarusActualEnv.get_action``) and exposes no rewind/replay hook, so the MC
method cannot be implemented by patching that loop.

This module re-implements the cache's decision-relevant behaviour as a plain,
fully-controllable Python loop. It is validated against the real IcarusGym
pipeline by ``experiments/experiment0/validate_sim.py`` (force-admit must
reproduce IcarusGym's per-request hit sequence exactly). Training then runs
here; evaluation still runs in IcarusGym.

Replicated semantics (see icarusgym/models/cache/policies.py::TtlCache and
icarusgym/models/strategy/onpath.py::IcarusGymLce):

Per request (time, content, size), at the single router cache:
  1. update(time): evict every cached item whose expiration_time < time.
  2. hit = content currently cached (after step 1).
  3. agent supplies (ttl, cache_size); ttl is stored for this content.
  4. set_cache_size(int(cache_size * (1 + extra_cache_size_ratio))): trim by
     ITEM COUNT (evict soonest-to-expire) while len > size. Applies in both modes.
  5. if hit and is_reset: expiration_time = time + ttl  (reset-on-hit).
     if miss: put(content) on the return path:
       - item-count mode: if full, only admit when the new item's expiration is
         later than the soonest incumbent's; otherwise drop the new item.
       - size-aware mode: evict soonest-to-expire until the new item fits in
         bytes; if it cannot fit even when empty, drop it.

``admit`` maps to ttl=admit_ttl, ``reject`` maps to ttl=reject_ttl. With a tiny
reject_ttl the rejected object is evicted on the very next update(), i.e. it is
effectively not cached -- the TTL-cache rendering of a binary admission gate.
"""

# Author: labry

import math


class _Item:
    """Minimal cache entry mirroring TtlCache.CacheInfo (decision-relevant fields)."""
    __slots__ = ('content', 'expiration_time', 'ttl', 'size', 'seq')

    def __init__(self, content, expiration_time, ttl, size, seq):
        self.content = content
        self.expiration_time = expiration_time
        self.ttl = ttl
        self.size = size
        self.seq = seq          # insertion order; tie-breaker for equal expirations


class TtlReplayCache:
    """Single-cache replica of IcarusGym TtlCache + IcarusGymLce request flow.

    :param size: Item-count capacity (the ``set_cache_size`` target).
    :param is_reset: Reset (extend) expiration on every hit.
    :param size_aware: Track capacity in bytes (max_bytes = size * default_size)
        instead of item count for the put() eviction. The item-count trim in
        set_cache_size still applies in both modes.
    :param default_size: Per-content size fallback; also sets the byte capacity.
    :param extra_cache_size_ratio: Slack ratio applied in set_cache_size, exactly
        as TtlCache.get() does: set_cache_size(int(cache_size * (1 + ratio))).
    """

    def __init__(self, size, is_reset=True, size_aware=False, default_size=1.0,
                 extra_cache_size_ratio=0.001):
        self._size = int(size)
        self._is_reset = is_reset
        self._size_aware = size_aware
        self._default_size = float(default_size)
        self._extra_cache_size_ratio = extra_cache_size_ratio
        self._max_bytes = float(self._size) * self._default_size if size_aware else math.inf

        self._items = {}            # content_id -> _Item
        self._current_bytes = 0.0
        self._current_time = 0.0
        self._seq = 0               # monotonically increasing insertion counter

    def __len__(self):
        return len(self._items)

    # -- min remaining-TTL victim (soonest to expire); ties broken by insertion order --
    def _pop_min_expiration(self):
        victim = min(self._items.values(),
                     key=lambda it: (it.expiration_time, it.seq))
        self._items.pop(victim.content)
        if self._size_aware:
            self._current_bytes -= victim.size
        return victim

    def update(self, time):
        """Advance time and evict expired items (expiration_time < time)."""
        self._current_time = time
        expired = [c for c, it in self._items.items() if it.expiration_time < time]
        for c in expired:
            it = self._items.pop(c)
            if self._size_aware:
                self._current_bytes -= it.size

    def set_cache_size(self, raw_cache_size):
        """Replicate TtlCache.set_cache_size: trim by item count (evict soonest)."""
        self._size = int(raw_cache_size * (1.0 + self._extra_cache_size_ratio))
        while len(self._items) > self._size:
            self._pop_min_expiration()

    def _put(self, content, size, ttl):
        """Insert on a miss; replicate TtlCache.put eviction for the active mode."""
        expiration_time = self._current_time + ttl
        if self._size_aware:
            while (self._current_bytes + size > self._max_bytes and len(self._items) > 0):
                self._pop_min_expiration()
            if self._current_bytes + size > self._max_bytes:
                return  # single item larger than whole cache: not inserted
            self._seq += 1
            self._items[content] = _Item(content, expiration_time, ttl, size, self._seq)
            self._current_bytes += size
        else:
            if len(self._items) == self._size:
                # Admit only if new item outlives the soonest-to-expire incumbent.
                min_exp = min((it.expiration_time for it in self._items.values()),
                              default=math.inf)
                if min_exp > expiration_time:
                    return  # drop the new item
                self._pop_min_expiration()
            self._seq += 1
            self._items[content] = _Item(content, expiration_time, ttl, size, self._seq)

    def step(self, time, content, size, ttl, cache_size):
        """Process one request; returns (hit, remaining_ttl) as IcarusGym would.

        Mirrors the order inside IcarusGymLce.process_event for one router cache:
        update -> (hit determined) -> set_cache_size -> reset-on-hit / put-on-miss.
        """
        self.update(time)

        it = self._items.get(content)
        hit = it is not None
        remaining_ttl = (it.expiration_time - time) if hit else 0.0

        self.set_cache_size(cache_size)

        if hit:
            if self._is_reset:
                cur = self._items.get(content)
                if cur is not None:                 # may have been trimmed by set_cache_size
                    cur.expiration_time = time + ttl
                    cur.ttl = ttl
        else:
            self._put(content, size, ttl)

        return hit, remaining_ttl


def simulate(requests, decisions, cache_size, admit_ttl, reject_ttl,
             is_reset=True, size_aware=False, default_size=1.0,
             extra_cache_size_ratio=0.001):
    """Replay a request trace under a sequence of admit/reject decisions.

    :param requests: list of (time, content, size) in arrival order.
    :param decisions: iterable of 1 (admit) / 0 (reject) per request, OR None to
        force-admit every request (the AdmitAll / LRU-equivalent baseline).
    :param cache_size: item-count capacity passed to set_cache_size each request
        (the agent's cache_size action; constant b_0 in experiment0).
    :return: dict with per-request 'hits' (0/1) and 'remaining_ttls', plus
        aggregate 'hit_rate' and per-content 'request_counts'/'hit_counts'.
    """
    cache = TtlReplayCache(cache_size, is_reset=is_reset, size_aware=size_aware,
                           default_size=default_size,
                           extra_cache_size_ratio=extra_cache_size_ratio)
    n = len(requests)
    hits = [0] * n
    rttls = [0.0] * n
    req_counts = {}
    hit_counts = {}

    for k, (time, content, size) in enumerate(requests):
        admit = 1 if decisions is None else int(decisions[k])
        ttl = admit_ttl if admit == 1 else reject_ttl
        hit, rttl = cache.step(time, content, size, ttl, cache_size)
        hits[k] = 1 if hit else 0
        rttls[k] = rttl
        req_counts[content] = req_counts.get(content, 0) + 1
        if hit:
            hit_counts[content] = hit_counts.get(content, 0) + 1

    total_hits = sum(hits)
    return {
        'hits': hits,
        'remaining_ttls': rttls,
        'hit_rate': total_hits / n if n else 0.0,
        'n_requests': n,
        'n_hits': total_hits,
        'request_counts': req_counts,
        'hit_counts': hit_counts,
    }
