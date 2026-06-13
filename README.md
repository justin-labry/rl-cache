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

Run from the repo root with the venv active. (The MC trainer is `train_mc.py`;
`main.py --mode train` is the legacy REINFORCE prototype, kept for reference.)

```bash
source .venv/bin/activate
cd /home/labry/git/rl-cache
E=experiments/experiment0

# 0. One-time: extract the deterministic SEED=42 request trace -> trace_seed42.npz
python $E/extract_trace.py

# 1. (optional) Validate the standalone simulator reproduces IcarusGym
python $E/validate_sim.py

# 2. Train RL-Cache with Monte Carlo elite-sampling -> model.pt
#    (seed sweep + best-checkpoint; object-hit reward. Use --reward-mode bhr for byte hit rate.)
python $E/train_mc.py --K 1000 --m 400 --L 3000 --q 4 --passes 6 --seed 0

# 3. Compare RL-Cache vs AdmitAll/SecondHit, and sweep cache sizes (OHR + BHR)
python $E/compare_baselines.py
python $E/sweep_cache_size.py

# IcarusGym-native eval (slower; full gym rollout instead of the validated sim)
python $E/main.py --mode test
```

### Plotting the per-content hit-probability figure

```bash
# 1. Evaluate a policy -> per-content NPZ in the repo root (via the validated simulator)
python $E/eval_sim.py                                  # RL-Cache (model.pt) -> results_rl_cache.npz
python $E/eval_sim.py --policy admitall  --out results_admitall
python $E/eval_sim.py --model model_bhr.pt --out results_rl_cache_bhr   # a BHR-trained model

# 2. Plot it -> PNG in the repo root (x = Content ID, y = hit probability, both log)
python $E/plots/plot_results.py                                        # results_rl_cache.npz -> .png
python $E/plots/plot_results.py --input results_rl_cache_bhr --label "RL-Cache (BHR)"
```

The legend reports the per-request hit rate (OHR). Note: with the size-aware,
non-stationary workload a faithful RL-Cache admits by object size rather than by
popularity, so hit probability is scattered across Content IDs (small objects hit,
large objects are rejected) rather than decreasing smoothly with ID.

### Other figures

```bash
# Learned size threshold: mean P(admit) and hit rate vs object size
python $E/plots/plot_hit_vs_size.py                    # model.pt -> results_rl_cache_vs_size.png
# (a BHR-trained model first needs: python $E/train_mc.py --reward-mode bhr --out model_bhr.pt --force)
python $E/plots/plot_hit_vs_size.py --model model_bhr.pt --out results_size_bhr.png

# OHR vs BHR across cache sizes (grouped bars; reads sweep_results.json from sweep_cache_size.py)
python $E/plots/plot_sweep.py                          # -> results_ohr_bhr_sweep.png
```

`plot_hit_vs_size.py` shows the admission probability falling from ~1 to ~0 as
object size grows (the learned threshold) with hit rate following. `plot_sweep.py`
shows the OHR/BHR mirror image: RL-Cache (OHR-reward) dominates object hit rate
while RL-Cache (BHR-reward) recovers byte hit rate toward AdmitAll.


## Project structure

```
rl-cache/
├── rl_cache/                        # Core module
│   ├── rl_cache_policy.py           # NN admission policy (RLlib Policy); SecondHit mode; trace capture
│   ├── rl_cache_agent.py            # RLlib algorithm wrapper
│   ├── features.py                  # RL-Cache Table-1 features (shared by train + eval)
│   ├── baselines.py                 # AdmitAll, SecondHit decision generators
│   ├── training/
│   │   ├── ttl_sim.py               # Standalone TtlCache replica (OHR + BHR)
│   │   └── mc_trainer.py            # Monte Carlo elite-sampling trainer (reward_mode ohr/bhr)
│   └── evaluation/
│       └── rl_cache_callbacks.py    # Metrics logging callbacks
├── experiments/
│   └── experiment0/
│       ├── main.py                  # RLlib entry point (legacy REINFORCE train / eval / baseline)
│       ├── train_mc.py              # MC elite-sampling training -> model.pt
│       ├── extract_trace.py         # Save the deterministic SEED=42 trace
│       ├── validate_sim.py          # Spike: sim vs IcarusGym hit-rate equivalence
│       ├── eval_quick.py            # Eval trained model in IcarusGym + sim cross-check
│       ├── compare_baselines.py     # AdmitAll / SecondHit / RL-Cache comparison
│       ├── sweep_cache_size.py      # OHR/BHR across cache sizes (OHR- and BHR-trained)
│       ├── config.py                # Agent / experiment configuration
│       └── icarus_config.py         # Icarus simulator configuration
└── README.md
```


## Results

Experiment0 workload: non-stationary Zipf (α=0.8), 1000 contents, sizes log-uniform
100 B–1 MB, size-aware byte-capacity TTL cache. Hit rates from the standalone
simulator (validated to match IcarusGym within ~0.003); RL-Cache trained per cache
size (seed sweep, best checkpoint). OHR = object hit rate, BHR = byte hit rate.

| cache | policy | OHR | BHR |
|:--|:--|--:|--:|
| 10% | AdmitAll | 0.088 | **0.047** |
| 10% | SecondHit | 0.091 | 0.048 |
| 10% | **RL-Cache (OHR-reward)** | **0.165** | 0.005 |
| 10% | RL-Cache (BHR-reward) | 0.041 | 0.041 |
| 20% | AdmitAll | 0.145 | **0.097** |
| 20% | **RL-Cache (OHR-reward)** | **0.210** | 0.021 |
| 20% | RL-Cache (BHR-reward) | 0.070 | 0.089 |

**Object vs byte hit rate is a reward choice.** With the object-hit reward (as in the
original paper), RL-Cache learns a *size threshold* — admit ~all small objects, reject
large ones (admit fraction by size quartile: 1.00 / 1.00 / 0.22 / 0.06) — maximizing
OHR (+45–87% over AdmitAll) but serving almost no bytes from cache. Switching the
reward to byte-hits flips the policy toward admitting large objects (Q4 admit 0.06 →
0.90), recovering BHR ~9× while ceding OHR.

On this workload object size and popularity are independent (corr ≈ −0.02) and 90.7%
of requested bytes are in the largest size quartile, so for BHR there is no "harmful
large" subset to reject and AdmitAll is near-optimal — RL-Cache (BHR) approaches but
does not beat it. The original RL-Cache reports only object hit rate; surfacing this
OHR/BHR tradeoff (and that the rich feature set collapses to a learned AdaptSize-style
size threshold here) is what the IcarusGym evaluation adds.


## Status / roadmap

Migration from the initial REINFORCE prototype to a faithful RL-Cache — complete:

- [x] **0.** De-risk spike: standalone TTL sim reproduces IcarusGym 100% (force-admit)
- [x] **1.** MC elite-sampling trainer (replaces REINFORCE; advantage target, best-checkpoint)
- [x] **2.** Align features to RL-Cache Table 1 (size, freq, recency r/ρ, ordinal d/δ, f/s, f·s)
- [x] **3.** Baselines: SecondHit + AdmitAll (native in IcarusGym + simulator)
- [x] **4.** Cache substrate decision: keep TTL-reset (common with dehghan-cache)
- [x] **5.** Byte hit rate (BHR) + cache-size sweep; OHR- and BHR-reward variants
- [x] **6.** Fix citations (Kirilin et al., NetAI 2019)


## Key differences from Dehghan-Cache

| | Dehghan-Cache | RL-Cache |
|:--|:--|:--|
| Decision | Utility gradient → TTL | Neural network → P(admit) → TTL |
| State | Per-content vectors (`_ws`, `_hs`, `_ts`, `_alpha`) | Neural network weights |
| Learning | Gradient ascent on utility function | Monte Carlo elite-sampling (imitate top-p% hit-rate samples) |
| Output | Continuous TTL value | Binary admit/reject mapped to TTL |
| Substrate | IcarusGym TTL cache | IcarusGym TTL cache (shared, for comparison) |
