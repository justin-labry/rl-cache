"""Admission baselines for comparison against RL-Cache (Kirilin et al., NetAI 2019, §4).

The paper compares RL-Cache with AdmitAll, SecondHit, and AdaptSize. For the
IcarusGym adaptation we use **AdmitAll** and **SecondHit** (AdaptSize, a Markov
size-threshold model, is out of scope and noted as such in the paper).

  * AdmitAll  - admit every requested object (the default for eviction-focused
                caches). In the simulator this is simply ``decisions=None``; in
                IcarusGym it is the policy's force-admit mode.
  * SecondHit - admit an object only on a REPEATED request (Maggs & Sitaraman,
                "Algorithmic Nuggets in Content Delivery", 2015): a Bloom-filter
                front end records first sightings and admits on the second hit.
                ``window`` bounds how long a prior sighting still counts (the
                paper's "fixed time interval"); default is unbounded.

These helpers produce decision arrays for the standalone simulator (ttl_sim).
The policy implements the same SecondHit rule online for IcarusGym evaluation;
the two are cross-checked in experiments/experiment0/compare_baselines.py.
"""

# Author: labry

import numpy as np


def admitall_decisions(trace):
    """All-ones decision array (admit everything)."""
    return np.ones(len(trace), dtype=np.int64)


def secondhit_decisions(trace, window=float('inf')):
    """SecondHit decisions: admit object j iff it was requested before, within `window`.

    :param trace: iterable of (time, content, size) in arrival order.
    :param window: max seconds since the previous sighting for it to still count
        (the Bloom-filter retention interval); default unbounded.
    :return: np.ndarray (len(trace),) of 0/1 admission decisions.
    """
    last_seen = {}
    decisions = np.empty(len(trace), dtype=np.int64)
    for k, (t, c, s) in enumerate(trace):
        prev = last_seen.get(c)
        decisions[k] = 1 if (prev is not None and (t - prev) <= window) else 0
        last_seen[c] = t
    return decisions
