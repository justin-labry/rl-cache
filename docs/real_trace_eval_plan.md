# RL-Cache — Real-Trace Evaluation Hardening (handoff)

**Last updated:** 2026-06-15 · machine: 10.254.73.47 (`ssh ncof` → .212 / L40S also available)

## Why this matters
The RL-Cache reimplementation is functionally complete, but the evaluation uses a
**synthetic non-stationary Zipf workload** (~20k requests, 1000 contents). For the
**IEEE TNSM** submission, real traces + larger scale + a train/test split are the
**binding constraint** for acceptance (reviewers expect real CDN/web traces, not
synthetic-only). Goal: re-run the OHR/BHR + cache-size sweep on a **real trace**.

## Feasibility: HIGH
The pipeline is already trace-based — it ingests an ordered list of
`(timestamp, object_id, size_bytes)`. So "introduce real traces" = swap the
*generator*, not the architecture. Training runs in the standalone `ttl_sim`
(trace in directly); IcarusGym needs a trace-replay workload (a ~50-line clone of
the current `NonstationaryWorkload`). `ttl_sim`/`features` key content by dict so
arbitrary object IDs are fine; only IcarusGym's cache + policy arrays need a
catalog bound (= #unique objects in the segment).

## DONE so far
- RL-Cache reimplementation complete + committed/pushed on `main`:
  - `7e691ea` MC elite-sampling (replaces REINFORCE), Table-1 features, train-on-sim/eval-in-IcarusGym (validated 100%), citations fixed.
  - `c2a1872` BHR + cache-size sweep + BHR-reward variant (`reward_mode={ohr,bhr}`).
  - `ea844c1` figure/eval scripts (per-content, hit-vs-size, OHR/BHR sweep).
- Key finding (synthetic): RL-Cache(OHR) beats AdmitAll OHR +86% but BHR is 4-10× worse; reward switch (BHR) flips policy to admit large objects. Verified.
- Venue decision: **IEEE TNSM** (#1), backups Computer Networks / Performance Evaluation. Framing = platform + comparative + findings.
- Real-trace source vetting: workflow `wxa1xcsr7` (Wikipedia/LRB, Twitter OSDI'20, Akamai Tragen, SNIA) — **was still running at handoff; re-check / re-run for the final dataset pick.**

## Recommended trace (verified 2026-06-17)
> The automated trace-vetting workflow (`wxa1xcsr7`) **stalled** (survey agents hung mid web-search, no structured result); replaced by direct verification below.

**#1 — open CDN cache traces via the cacheMon / libCacheSim hub** (Juncheng Yang et al.):
- Repos: `github.com/cacheMon/cache_dataset`, `github.com/1a1a11a/libCacheSim`. Aggregates many open cache traces (incl. **Wikipedia/Wikimedia CDN**, Tencent Photo, etc.) in a **uniform format carrying `(time, obj_id, obj_size)`**, with libCacheSim to parse / replay / convert.
- Has **object sizes** (BHR ✓) + timestamps; CDN framing matches RL-Cache; standard in modern caching papers (LRB NSDI'20). Cite **Song et al., LRB, NSDI 2020** / Yang et al.
- Pipeline fit: convert to `timestamp object_id size` → drop-in (also the `quovadim/RL-Cache` CSV format).

**#2 — Twitter production cache traces** (Yang et al., OSDI'20; `github.com/twitter/cache-trace`):
- Plain-text CSV columns: `timestamp, anon_key, key_size, value_size, client_id, operation, TTL`. **Sizes ✓** (use value size, or key+value, for BHR). zstd-compressed; very large (use a representative segment).
- Key-value (memcached) workload — not CDN. Best as a **second workload class** to show generality.

**#3 — Akamai Tragen** (IMC'21; footprint descriptors, Sundarrajan et al. CoNEXT'17): realistic CDN trace **generator** calibrated to Akamai footprints — **same research lineage as RL-Cache**. Emits `(time, id, size)`; avoids privacy/access issues. Good fallback if real downloads are awkward.

⚠️ All three carry object sizes (needed for BHR). Avoid block/storage traces lacking per-object sizes.

## TODO (next session)
1. **Obtain the trace.** Download on an internet-capable machine (this .47 sandbox blocks outbound downloads; use a normal shell or `.212`). Normalize to `timestamp object_id object_size`.
2. **`experiments/experiment0/preprocess_trace.py`** — parse → sorted `(time, content_id, size)` → `trace_real.npz`; remap object IDs to dense indices; record catalog size; print stats (n_requests, n_unique, size distribution, active bytes).
3. **Train/test split** — front segment for training, held-out tail for eval (paper methodology + TNSM "no train/test overlap"). Cap training segment to ~100k–1M (pure-Python MC limit; see constraints).
4. **`TraceWorkload`** in IcarusGym (clone `NonstationaryWorkload` in `icarus_config.py`) that replays `trace_real.npz`; set `content_max` = catalog size; register it.
5. **Byte cache-size sweep** — set capacities as fractions of the trace's active/unique bytes (paper's 2/16/128 GB style). Extend `sweep_cache_size.py`.
6. **Evaluate** on held-out via validated `ttl_sim` (fast): OHR/BHR + sweep for AdmitAll / SecondHit / RL-Cache(OHR) / RL-Cache(BHR); confirm one point in IcarusGym (equivalence already validated).
7. **Multi-seed + confidence intervals** (MC training is high-variance; best-checkpoint already implemented).
8. Update README Results + figures with real-trace numbers.

## Honest constraints
- **Scale:** pure-Python MC elite-sampling cannot do the paper's 10M requests. Train on a representative segment (100k–1M), eval on held-out. State this in the paper.
- **Download:** trace files are multi-GB; the .47 sandbox may block fetching — download elsewhere (.212) or locally.
- **Object-ID space:** large/sparse → remap to dense indices (done in preprocess).

## Key files (rl-cache)
- `rl_cache/training/ttl_sim.py` — standalone TTL cache sim (OHR+BHR), validated == IcarusGym.
- `rl_cache/training/mc_trainer.py` — MC elite-sampling; `reward_mode={ohr,bhr}`; advantage target; best-checkpoint.
- `rl_cache/features.py` — RL-Cache Table-1 features (shared train/eval).
- `rl_cache/baselines.py` — AdmitAll, SecondHit.
- `experiments/experiment0/`: `train_mc.py`, `extract_trace.py` (current synthetic capture → adapt for real trace), `sweep_cache_size.py`, `compare_baselines.py`, `eval_quick.py`, `validate_sim.py`, `icarus_config.py` (add `TraceWorkload`), `config.py`.
- Artifacts (gitignored): `trace_seed42.npz` (current synthetic trace), `model.pt`, `sweep_results.json`.

## Resume commands
```sh
cd /home/labry/git/rl-cache && source .venv/bin/activate
git log --oneline -3                      # confirm at ea844c1
python experiments/experiment0/train_mc.py --smoke    # sanity: MC trainer runs
# then implement TODO #1-2 (get trace + preprocess) to start the real-trace path
```
