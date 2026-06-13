"""Monte Carlo elite-sampling trainer for RL-Cache (Kirilin et al., NetAI 2019, §3.2).

This replaces the earlier REINFORCE prototype with RL-Cache's actual training
mechanism. It runs on the validated standalone TTL simulator (ttl_sim) so that
the same window of requests can be replayed many times with different admission
decisions -- something IcarusGym's blocking event loop cannot do.

Algorithm (per sliding window of K requests, advancing by K each time):
  1. Sampling   - the network gives fixed admit probabilities p_i = A(u_i) for the
                  window (features are cache-state-independent, so p_i is constant
                  across samples). Draw m decision samples d ~ Bernoulli(p) over the
                  K window requests.
  2. Selection  - score each sample by the hit rate over the extended window K + L
                  (future hits discounted by gamma^(i-K)), simulating from the
                  current committed cache state; keep the top p-percentile samples.
  3. Learning   - train the network with binary cross-entropy to imitate the elite
                  samples: target_i = fraction of elite samples that admitted i.
  4. Refill     - every q windows, rebuild the committed cache from the trace start
                  under the current greedy policy to correct accumulated drift.
Termination: stop when the parameter change ||Δw|| falls below eps (or after a
fixed number of passes for smoke tests).

The extended-window tail [K, K+L) is simulated GREEDILY (admit iff p >= 0.5) so a
sample's score reflects its own K sampled decisions rather than tail randomness.
"""

# Author: labry

import copy
import logging
import numpy as np
import torch

from rl_cache.training.ttl_sim import TtlReplayCache

logger = logging.getLogger(__name__)


class MCTrainer:
    """Monte Carlo elite-sampling trainer.

    :param net: RLCacheNetwork (features -> P(admit) in [0,1]).
    :param optimizer: torch optimizer over net.parameters().
    :param trace: list of (time, content, size) in arrival order.
    :param features: np.ndarray (len(trace), feature_dim), precomputed once.
    :param cache_size: item-count capacity (b_0).
    :param admit_ttl / reject_ttl: TTL applied for admit / reject decisions.
    :param is_reset / size_aware / default_size: cache config (match IcarusGym).
    :param K: window size (requests sampled per window).
    :param m: decision samples per window.
    :param p_percentile: elite fraction in percent (top p% by score are selected).
    :param L: extended-window length for scoring.
    :param gamma: discount for the L tail.
    :param q: refill the committed cache every q windows.
    :param grad_clip: max grad norm.
    """

    def __init__(self, net, optimizer, trace, features, cache_size,
                 admit_ttl, reject_ttl, is_reset=True, size_aware=False,
                 default_size=1.0, extra_cache_size_ratio=0.001,
                 K=1000, m=100, p_percentile=20.0, L=2000, gamma=0.99997,
                 q=4, grad_clip=1.0, device='cpu', seed=0,
                 target_mode='advantage', adv_gain=2.0):
        self.net = net
        self.opt = optimizer
        self.trace = trace
        self.features = features
        self.cache_size = int(cache_size)
        self.admit_ttl = admit_ttl
        self.reject_ttl = reject_ttl
        self.is_reset = is_reset
        self.size_aware = size_aware
        self.default_size = default_size
        self.extra_ratio = extra_cache_size_ratio
        self.K = int(K)
        self.m = int(m)
        self.p_percentile = float(p_percentile)
        self.L = int(L)
        self.gamma = float(gamma)
        self.q = int(q)
        self.grad_clip = grad_clip
        self.device = torch.device(device)
        self.rng = np.random.RandomState(seed)
        self.n = len(trace)
        # target_mode: 'advantage' (mean-centered, level-pinned -> learns selectivity)
        #              'marginal'  (raw elite admit rate -> the paper's literal BCE; drifts
        #                           toward admit-all at small scale)
        self.target_mode = target_mode
        self.adv_gain = float(adv_gain)

    # ---- cache helpers -------------------------------------------------------
    def _new_cache(self):
        return TtlReplayCache(self.cache_size, is_reset=self.is_reset,
                              size_aware=self.size_aware,
                              default_size=self.default_size,
                              extra_cache_size_ratio=self.extra_ratio)

    def _ttl_for(self, admit):
        return self.admit_ttl if admit else self.reject_ttl

    def _admit_probs(self, lo, hi):
        """Network admit probabilities for trace[lo:hi] (no grad)."""
        feats = torch.from_numpy(self.features[lo:hi]).to(self.device)
        with torch.no_grad():
            p = self.net(feats).squeeze(-1).cpu().numpy()
        return np.clip(p, 1e-6, 1.0 - 1e-6)

    def _apply_window(self, cache, lo, hi, decisions):
        """Advance `cache` through trace[lo:hi] applying `decisions` (0/1). Returns #hits."""
        hits = 0
        for j in range(lo, hi):
            t, c, s = self.trace[j]
            hit, _ = cache.step(t, c, s, self._ttl_for(decisions[j - lo]), self.cache_size)
            hits += 1 if hit else 0
        return hits

    def _score_sample(self, base_cache, start, k_decisions, p_tail):
        """Discounted hit score over the extended window K+L from a cache snapshot.

        k_decisions: sampled 0/1 for the K window requests.
        p_tail: greedy decisions for the L tail are derived from probs >= 0.5.
        """
        cache = copy.deepcopy(base_cache)
        score = 0.0
        end_k = min(start + self.K, self.n)
        # K window: full weight.
        for j in range(start, end_k):
            t, c, s = self.trace[j]
            hit, _ = cache.step(t, c, s, self._ttl_for(k_decisions[j - start]), self.cache_size)
            if hit:
                score += 1.0
        # L tail: greedy decisions, discounted by gamma^(i-K).
        end_l = min(end_k + self.L, self.n)
        for idx, j in enumerate(range(end_k, end_l), start=1):
            t, c, s = self.trace[j]
            admit = 1 if p_tail[j - end_k] >= 0.5 else 0
            hit, _ = cache.step(t, c, s, self._ttl_for(admit), self.cache_size)
            if hit:
                score += (self.gamma ** idx)
        return score

    def _greedy_eval_full(self):
        """Greedy hit rate over the whole trace from a cold cache (clean eval proxy).

        Matches IcarusGym eval to ~0.002 (validated by eval_quick.py) and is cheap
        (~one ttl_sim pass), so it is used to pick the best checkpoint -- which is
        what RL-Cache's paper does ("choose the model that gives the highest hit rate").
        """
        p = self._admit_probs(0, self.n)
        greedy = (p >= 0.5).astype(np.int64)
        cache = self._new_cache()
        hits = 0
        for k in range(self.n):
            t, c, s = self.trace[k]
            hit, _ = cache.step(t, c, s, self._ttl_for(greedy[k]), self.cache_size)
            hits += 1 if hit else 0
        return hits / self.n

    def _refill(self, upto):
        """Rebuild committed cache by greedily replaying trace[0:upto]."""
        cache = self._new_cache()
        if upto <= 0:
            return cache
        p = self._admit_probs(0, upto)
        greedy = (p >= 0.5).astype(np.int64)
        self._apply_window(cache, 0, upto, greedy)
        return cache

    # ---- training ------------------------------------------------------------
    def train(self, num_passes=1, eps=1e-6, verbose=True, eval_interval=10):
        """Run MC elite-sampling. Returns a list of per-window metric dicts.

        Every `eval_interval` windows the greedy policy is evaluated over the full
        trace; the network state achieving the highest hit rate is retained in
        ``self.best_state`` (with ``self.best_hr``). This makes the result robust to
        the high run-to-run variance of MC training and follows the paper's
        "pick the best model" methodology.
        """
        n_elite = max(1, int(round(self.m * self.p_percentile / 100.0)))
        history = []
        self.best_hr = -1.0
        self.best_state = None
        global_w = 0
        converged = False

        for pass_idx in range(num_passes):
            committed = self._new_cache()
            n_windows = (self.n + self.K - 1) // self.K
            for w in range(n_windows):
                start = w * self.K
                end_k = min(start + self.K, self.n)
                kw = end_k - start
                # Skip a trailing partial window: too few requests give a degenerate
                # elite signal (target collapses) that destabilizes the BCE update.
                if kw < self.K:
                    break

                # Periodic refill of the committed cache (drift correction).
                if w > 0 and self.q > 0 and (w % self.q == 0):
                    committed = self._refill(start)

                # Fixed admit probabilities for the extended window.
                end_l = min(end_k + self.L, self.n)
                p_ext = self._admit_probs(start, end_l)
                p_win = p_ext[:kw]
                p_tail = p_ext[kw:]

                # --- 1) Sampling + 2) Scoring ---
                samples = (self.rng.random_sample((self.m, kw)) < p_win[None, :]).astype(np.int64)
                scores = np.empty(self.m, dtype=np.float64)
                for si in range(self.m):
                    scores[si] = self._score_sample(committed, start, samples[si], p_tail)

                # --- 2) Selection: top p-percentile by score ---
                elite_idx = np.argsort(scores)[-n_elite:]
                elite = samples[elite_idx]
                elite_rate = elite.mean(axis=0)                   # per-request elite admit rate

                # --- target: imitate elites, but how the level is handled differs ---
                if self.target_mode == 'advantage':
                    # Pin the overall admit level at 0.5 and keep only the RELATIVE
                    # preference (which objects elites admit more than average). This
                    # removes the admit-all level drift that marginal BCE suffers at
                    # small scale, leaving the net to learn selectivity.
                    centered = elite_rate - elite_rate.mean()
                    target = np.clip(0.5 + self.adv_gain * centered, 1e-3, 1.0 - 1e-3)
                else:  # 'marginal' -- the paper's literal elite-decision BCE
                    target = elite_rate
                target = target.astype(np.float32)

                # --- 3) Learning: BCE imitation of elite decisions ---
                feats = torch.from_numpy(self.features[start:end_k]).to(self.device)
                tgt = torch.from_numpy(target).to(self.device)
                prev = [p.detach().clone() for p in self.net.parameters()]

                p_pred = self.net(feats).squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
                loss = torch.nn.functional.binary_cross_entropy(p_pred, tgt)
                self.opt.zero_grad()
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.opt.step()

                dw = float(sum((p.detach() - q).norm().item()
                               for p, q in zip(self.net.parameters(), prev)))

                # --- 4) Advance committed cache by K (greedy under updated net) ---
                p_commit = self._admit_probs(start, end_k)
                greedy = (p_commit >= 0.5).astype(np.int64)
                committed_hits = self._apply_window(committed, start, end_k, greedy)

                elite_hit_rate = float(scores[elite_idx].mean() / max(kw, 1))
                metrics = {
                    'pass': pass_idx, 'window': w, 'start': start,
                    'loss': float(loss.item()),
                    'mean_p_admit': float(p_win.mean()),
                    'mean_target': float(target.mean()),
                    'elite_score_per_req': elite_hit_rate,
                    'committed_hit_rate': committed_hits / kw,
                    'dw': dw,
                }
                history.append(metrics)
                if verbose:
                    logger.info(
                        f'[MC] pass {pass_idx} win {w:3d} | loss={loss.item():.4f} '
                        f'p_admit={p_win.mean():.3f} elite_hr={elite_hit_rate:.4f} '
                        f'commit_hr={committed_hits / kw:.4f} dw={dw:.2e}')

                # Periodic full-trace greedy eval -> retain the best checkpoint.
                global_w += 1
                if eval_interval and (global_w % eval_interval == 0):
                    hr = self._greedy_eval_full()
                    if hr > self.best_hr:
                        self.best_hr = hr
                        self.best_state = copy.deepcopy(self.net.state_dict())
                    if verbose:
                        logger.info(f'[MC]   full-trace greedy hit rate={hr:.4f} '
                                    f'(best={self.best_hr:.4f})')

                if dw < eps:
                    if verbose:
                        logger.info(f'[MC] converged at pass {pass_idx} window {w} (dw<{eps})')
                    converged = True
                    break
            if converged:
                break

        # Final eval so the last state is also a candidate for best.
        hr = self._greedy_eval_full()
        if hr > self.best_hr:
            self.best_hr = hr
            self.best_state = copy.deepcopy(self.net.state_dict())
        return history
