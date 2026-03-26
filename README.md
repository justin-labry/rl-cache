# RL-Cache: Learning-Based Cache Admission on RLlib

RLlib agent for RL-Cache neural network-based cache admission control, running on
[IcarusGym](https://github.com/justin-labry/IcarusGym) environment with
[GymProxy](https://github.com/justin-labry/GymProxy).

Based on: *N. Narayanan et al., "RL-Cache: Learning-Based Cache Admission for Content Delivery," IEEE JSAC, 2018.*


## Architecture

```
IcarusGym Environment
  obs = [time, content_id, weight, size, remaining_ttl, hit]
  action = (ttl_value, cache_size)
        │
        ▼
  RLCachePolicy (Neural Network)
    obs → feature extraction (size, frequency, recency, ...)
        → 5-layer feedforward (ELU) → P(admit)
        → admit: large TTL / reject: small TTL
    training: REINFORCE policy gradient
```


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

# Install dependencies (editable mode)
pip install -e /home/labry/git/GymProxy
pip install -e /home/labry/git/icarus
pip install torch h5py
```


## Run

```bash
source .venv/bin/activate
cd experiments/experiment0
python main.py
```

Training and measurement are controlled by `config.py`:
- `NUM_EPISODES`: total iterations
- `EPISODE_MEASUREMENT_BEGIN`: episodes before this are training-only; after this, measurement begins


## Project Structure

```
rl-cache/
├── rl_cache/                        # Core module
│   ├── rl_cache_policy.py           # Neural network policy (REINFORCE)
│   ├── rl_cache_agent.py            # RLlib agent wrapper
│   └── evaluation/
│       └── rl_cache_callbacks.py    # Metrics logging callbacks
├── experiments/
│   └── experiment0/
│       ├── main.py                  # Entry point
│       ├── config.py                # Experiment configuration
│       └── icarus_config.py         # Icarus simulator configuration
└── .gitignore
```


## Key Differences from Dehghan-Cache

| | Dehghan-Cache | RL-Cache |
|:--|:--|:--|
| Decision | Utility gradient → TTL | Neural network → P(admit) → TTL |
| State | Per-content vectors (`_ws`, `_hs`, `_ts`, `_alpha`) | Neural network weights |
| Learning | Gradient ascent on utility function | REINFORCE policy gradient |
| Output | Continuous TTL value | Binary admit/reject mapped to TTL |
