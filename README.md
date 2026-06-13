# RL-Cache: Learning-Based Cache Admission on RLlib

A faithful re-implementation of **RL-Cache** adapted to the
[IcarusGym](https://github.com/justin-labry/IcarusGym) environment (via
[GymProxy](https://github.com/justin-labry/GymProxy)), for the IcarusGym TCCN paper.

Based on:

> V. Kirilin, A. Sundarrajan, S. Gorinsky, and R. K. Sitaraman,
> *"RL-Cache: Learning-Based Cache Admission for Content Delivery,"*
> NetAI 2019 (ACM SIGCOMM Workshop on Network Meets AI & ML), Beijing, China.
> Official implementation: https://github.com/quovadim/RL-Cache (formerly `WVadim/RL-Cache`).

> **Framing.** This is an *adaptation to IcarusGym*, not a reproduction of the
> paper's Akamai-trace study. Evaluation runs on IcarusGym's common workload so
> that RL-Cache can be compared apples-to-apples with the other algorithms in the
> platform. Algorithmic fidelity is preserved: the agent learns with RL-Cache's
> Monte Carlo elite-sampling method (not a generic policy gradient).


## Algorithm

RL-Cache is a **cache-admission** front end. A feedforward neural network maps an
object's features to an admission probability `A(u, w) ∈ [0, 1]`; the decision is
rounded to admit (1) or reject (0). Eviction is left to the cache (LRU in the
paper). RL-Cache optimizes the cache hit rate directly.

**Training — Monte Carlo elite-sampling (paper §3.2):**
slide a window of `K` requests; for each window draw `m` admit/reject decision
samples from `A(u, w)`; score each sample by the hit rate over an extended window
`K + L` (future requests discounted by `γ`); keep the top `p`-percentile samples
and train the network to imitate them with binary cross-entropy; refill the cache
every `q` windows; stop when `|Δw| < ε`.

This is **not** REINFORCE — the network learns to reproduce the highest-hit-rate
decision sequences, which is the defining mechanism of RL-Cache.


## Architecture: train on a standalone simulator, evaluate in IcarusGym

IcarusGym drives requests through an inverted, blocking event loop (the simulator
pulls actions per request via `IcarusActualEnv.get_action`) and exposes no
rewind/replay hook. RL-Cache's MC method needs to replay the same window many
times with different decisions, so training cannot run inside that loop.

```
[TRAIN]  deterministic trace (SEED=42) ─→ standalone TTL sim ─→ MC elite-sampling ─→ model.pt
                                                │  (rl_cache/training/ttl_sim.py)
                                                │  replicates IcarusGym TtlCache exactly
                                                ▼
                                          validate_sim.py: per-request hit/miss
                                          matches IcarusGym 20001/20001 (100%)
                                                ▲
[EVAL]   deterministic trace (SEED=42) ─→ IcarusGym (TtlCache, unchanged) ←──────── model.pt
                                                │
                              all reported hit-rate numbers come from here
```

This matches the paper's own design (offline training, online serving). The
standalone simulator reuses IcarusGym's cache logic and request stream, and a
validation spike (`experiments/experiment0/validate_sim.py`) proves the two
produce identical hit/miss decisions under force-admit.

**Cache substrate.** Like dehghan-cache, RL-Cache runs on the IcarusGym TTL cache
(`ICARUSGYM_TTL`, reset-on-hit). The binary admission decision is rendered as a
TTL action: **admit → large TTL**, **reject → tiny TTL** (evicted on the next
sweep). With reset-on-hit, this is equivalent to an LRU admission front end while
keeping a common substrate for platform comparison.

**Features (8-dim, RL-Cache Table 1).** size, frequency, temporal recency,
exponentially-smoothed recency, ordinal recency, exponentially-smoothed ordinal
recency, frequency/size, frequency·size.


## Requirements

- Python 3.11
- [GymProxy](https://github.com/justin-labry/GymProxy) (`pip install -e`)
- [Icarus](https://github.com/justin-labry/icarus) (`pip install -e`)
- [IcarusGym](https://github.com/justin-labry/IcarusGym) (via `sys.path`)


## Setup

```bash
cd /home/labry/git/rl-cache
python3.11 -m venv .venv
source .venv/bin/activate

pip install -e /home/labry/git/GymProxy
pip install -e /home/labry/git/icarus
pip install torch h5py
```


## Run

```bash
source .venv/bin/activate
cd experiments/experiment0

# Validate the standalone simulator against IcarusGym (de-risk spike)
python validate_sim.py

# Train (MC elite-sampling on the standalone sim) → save model.pt
python main.py --mode train

# Evaluate the trained model in IcarusGym → results_rl_cache.npz
python main.py --mode test

# AdmitAll / LRU-equivalent baseline (force-admit, no learning)
python main.py --mode baseline
```


## Project structure

```
rl-cache/
├── rl_cache/                        # Core module
│   ├── rl_cache_policy.py           # NN admission policy (RLlib Policy); trace-capture hook
│   ├── rl_cache_agent.py            # RLlib algorithm wrapper
│   ├── training/
│   │   └── ttl_sim.py               # Standalone TtlCache replica for MC training
│   └── evaluation/
│       └── rl_cache_callbacks.py    # Metrics logging callbacks
├── experiments/
│   └── experiment0/
│       ├── main.py                  # Train / test / baseline entry point
│       ├── validate_sim.py          # Spike: sim vs IcarusGym hit-rate equivalence
│       ├── config.py                # Agent / experiment configuration
│       └── icarus_config.py         # Icarus simulator configuration
└── README.md
```


## Status / roadmap

Migration from the initial REINFORCE prototype to a faithful RL-Cache:

- [x] **0.** De-risk spike: standalone TTL sim reproduces IcarusGym 100% (force-admit)
- [ ] **1.** MC elite-sampling trainer (replaces REINFORCE) — *in progress*
- [ ] **2.** Align features to RL-Cache Table 1 (add ρ, d, δ)
- [ ] **3.** Baselines: SecondHit + AdmitAll
- [x] **4.** Cache substrate decision: keep TTL-reset (common with dehghan-cache)
- [ ] **5.** Byte hit rate (BHR) + cache-size sweep
- [x] **6.** Fix citations (Kirilin et al., NetAI 2019)


## Key differences from Dehghan-Cache

| | Dehghan-Cache | RL-Cache |
|:--|:--|:--|
| Decision | Utility gradient → TTL | Neural network → P(admit) → TTL |
| State | Per-content vectors (`_ws`, `_hs`, `_ts`, `_alpha`) | Neural network weights |
| Learning | Gradient ascent on utility function | Monte Carlo elite-sampling (imitate top-p% hit-rate samples) |
| Output | Continuous TTL value | Binary admit/reject mapped to TTL |
| Substrate | IcarusGym TTL cache | IcarusGym TTL cache (shared, for comparison) |
