# RL-Cache on IcarusGym — 구현 평가 및 보강 포인트

**작성:** 2026-07-07 (Claude 세션, worktree `claude/vigilant-franklin-5504b9`)
**목적:** IcarusGym 논문(IEEE TNSM 목표) §VI.B *Reproducibility Evaluation*의 key paper 중 하나인
RL-Cache (Kirilin et al., NetAI 2019)의 현 구현을 평가하고 보강 포인트를 우선순위별로 정리.
**전제(사용자 확인):** 목표는 RL-Cache의 완벽 재현이 아니라 **IcarusGym 플랫폼에 적절히 적응시켜
다른 캐시 알고리즘들과 비교**하는 것.

---

## 0. TL;DR

구현 자체는 **잘 되어 있다.** 알고리즘 충실도(MC elite-sampling), 검증(standalone sim ↔ IcarusGym
100% 일치), 원 논문에 없는 OHR/BHR reward 분석까지 갖췄고, 설계 결정들이 README에 정직하게
문서화되어 있다. 다만 **TNSM §VI.B의 목적(= key paper의 중요 그래프 재현 + 플랫폼 효용 입증)
기준으로 보면** 다음이 부족하다:

| # | 보강 포인트 | 우선순위 |
|---|---|---|
| 1 | Train/eval이 **같은 trace(SEED=42)** — train/test 분리 필요 | **P0** |
| 2 | 원 논문 **key figure 3종과의 매핑 부재** (Fig 3 / Fig 4–6 / Fig 7–9 스타일) | **P0** |
| 3 | Real trace 평가 (계획 완료: `docs/real_trace_eval_plan.md`) | **P0** |
| 4 | **Reject의 일시 삽입이 admission 정책에 불리** — 실측 OHR −20%/BHR −54% 왜곡 (§3) | **P0** |
| 5 | Icarus **네이티브 정책(LRU/LFU 등)과의 비교 부재** — PassiveAgentCache 미활용 | **P1** |
| 6 | AdaptSize 대체 baseline (best static size-threshold) 부재 | **P1** |
| 7 | 다중 seed 평가 + 신뢰구간 (현재 best-checkpoint 단일 수치) | **P1** |
| 8 | "훈련이 IcarusGym 밖(standalone sim)"인 점의 논문 서술 전략 미정 | **P1** |
| 9 | stale 주석/문서, trace wrap-around 등 코드 정리 (참조 구현 대조 반영: §4) | P2 |

---

## 1. 잘 된 점 (유지할 것)

1. **MC elite-sampling의 충실한 구현** (`rl_cache/training/mc_trainer.py`).
   초기 REINFORCE 프로토타입을 폐기하고 논문 §3.2의 실제 메커니즘(K-window 슬라이딩, m개
   decision sample, top-p% elite 선택, K+L 확장 윈도우 + γ 할인 스코어링, q-window마다 refill,
   ‖Δw‖<ε 종료)을 그대로 구현. 파라미터 구조가 논문과 1:1 대응한다
   (논문: K=25k, m=250, p=20%, L=175k, γ=0.99997 / 구현: 20k trace에 맞게 축소, 문서화됨).
2. **Table-1 8-feature를 train/eval이 공유** (`rl_cache/features.py`의 단일 `FeatureExtractor`).
   feature가 cache 상태와 무관하다는 관찰을 이용해 전체 trace의 feature를 1회 사전계산 —
   올바르고 효율적인 설계.
3. **검증 스파이크가 방법론적으로 모범적.** `validate_sim.py`가 force-admit에서 IcarusGym과
   standalone sim의 per-request hit/miss **20001/20001 일치**를 확인했고,
   `compare_baselines.py`가 세 정책 모두에서 IcarusGym vs sim 차이 ≤0.005를 교차 확인.
   "훈련은 sim, 평가는 IcarusGym" 구조의 신뢰 근거가 된다.
4. **Admission → TTL rendering이 플랫폼 목적에 정확히 부합.**
   admit → 큰 TTL, reject → 미세 TTL로 이진 admission을 IcarusGym의 기존 `ICARUSGYM_TTL`
   substrate 위에 얹었다(플랫폼 무수정). dehghan-cache와 공통 substrate → 알고리즘 간 공정 비교라는
   플랫폼 스토리에 맞음. **이번 세션 실측으로 뒷받침됨** (아래 §3: 현 설정에서 capacity eviction이
   지배하여 admit_ttl 값에 무감 → "LRU-equivalent" 주장 성립).
5. **OHR/BHR reward 변형 + cache-size sweep** — 원 논문에 없는 발견(OHR-reward는 size threshold를
   학습해 OHR +45–87%, BHR은 붕괴; BHR-reward는 정책이 뒤집힘). 이것이 "IcarusGym 평가가 원 논문에
   무엇을 더하나"의 핵심 스토리이므로 논문에 반드시 살릴 것.
6. **MC 분산 대응** — seed sweep + full-trace greedy eval 기반 best-checkpoint 유지
   (논문의 "choose the model that gives the highest hit rate" 방법론과 일치).

---

## 2. 보강 포인트 (우선순위별)

### P0 — TNSM 제출 전 필수

#### P0-1. Train/test 분리 (현재: train trace == eval trace)
- 현재 훈련(`train_mc.py`)과 평가(`compare_baselines.py`, `sweep_cache_size.py`, IcarusGym eval)가
  **모두 동일한 `trace_seed42.npz`** 를 사용. 원 논문은 "the training and testing sets do not
  overlap"을 명시하므로 리뷰어가 반드시 지적할 지점.
- `config.py`의 `NUM_EVAL_EPISODES = 200`도 SEED=42 고정이라 **동일 에피소드 200회 반복** —
  통계적으로 무의미(결정적 1회와 동일).
- **즉시 가능한 최소 조치(synthetic 유지):** eval용 trace를 다른 seed(예: 43, 44, 45)로 추출해
  훈련은 seed-42, 평가는 seed-43+에서 수행. NONSTATIONARY workload는 seed에 따라 phase permutation이
  바뀌므로 유의미한 held-out이 된다. `extract_trace.py`에 `--seed` 인자만 추가하면 됨(현재 icarus_config
  SEED 고정값 사용 중).
- **정석 조치:** real trace의 앞부분 훈련 / 뒤 held-out 평가 (P0-3과 함께).

#### P0-2. 원 논문 key figure 3종 재현 (§VI.B의 명시 목표)
IcarusGym 논문 §VI.B 초안: "Key paper에서의 중요 graph를 그대로 재현하여 보여줌. 몇 줄의 code로
재현했음을 명시." 현재 그림들(per-content hit prob, hit-vs-size, OHR/BHR sweep)은 분석용으로는
좋지만 **원 논문 그림과 시각적으로 대응되지 않는다.** 재현 대상 제안:

| 원 논문 | 내용 | 현재 상태 | 필요 작업 |
|---|---|---|---|
| **Fig 4–6** | cache size별 평균 hit rate, 정책별 grouped bar (AdmitAll/SecondHit/AdaptSize/RL-Cache) | 데이터는 `sweep_results.json`에 있음 | `plot_sweep.py`를 논문 스타일(x=cache size, 정책별 bar)로 재구성. ~20 lines |
| **Fig 3** | 시간에 따른 hit-rate 동역학 (1시간 창) — 정책별 곡선 | 없음 | `ttl_sim.simulate`가 per-request `hits`를 이미 반환 → rolling-window HR 플롯. ~30 lines |
| **Fig 7–9** | active bytes 곡선 (시각 t에 활성인 object 총 bytes) — cache size 선택 근거 | 없음 | trace만으로 계산(정책 무관): 각 object의 [first, last] 구간에 size 기여. ~30 lines. 원 논문의 핵심 분석 개념이라 재현 가치 높음 |
- 세 그림 모두 validated sim에서 뽑을 수 있으므로 IcarusGym 재실행 불필요 → 빠르다.
- 논문에는 "재현에 필요했던 플랫폼 측 코드 몇 줄"을 같이 명시 (P1-8의 LOC 집계와 연결).

#### P0-3. Real trace 평가
- 이미 상세 계획 있음: [docs/real_trace_eval_plan.md](docs/real_trace_eval_plan.md)
  (cacheMon/libCacheSim 허브 1순위, Twitter cache-trace 2순위, Akamai Tragen 폴백; 전처리 →
  TraceWorkload → byte 기준 cache-size sweep → held-out 평가 순서까지 정리됨).
- TNSM 심사에서 synthetic-only가 가장 큰 리스크라는 판단(venue 결정 메모)과 일치. **P0-1, P0-2를
  synthetic에서 먼저 완성해 파이프라인을 굳힌 뒤 trace만 갈아끼우는 순서**를 권장.

#### P0-4. Reject의 "일시 삽입" 왜곡 수정 (이번 세션 실측, §3 참조)
- 원 구현(C++ sim)은 reject 시 **아예 삽입하지 않는다** (`if (!admission_decision) return;`).
  반면 우리의 TTL-rendering은 reject도 일단 `put(ttl=0.1)`으로 삽입하는데, **size-aware 모드의
  `put()`은 새 객체가 들어갈 자리를 만들기 위해 기존 항목들을 축출**하므로, reject된 대형 객체가
  0.1초만 머물다 사라지면서도 캐시 내용물을 밀어낸다.
- 실측 영향 (cache ratio 0.1, seed-42 trace): reject를 진짜 no-insert로 바꾸면
  **RL-Cache OHR 0.172 → 0.216 (상대 +25%), BHR 0.0095 → 0.0205 (2.2×)**. SecondHit도 소폭 개선
  (0.0905 → 0.0922). AdmitAll은 reject이 없어 무영향 — 즉 이 왜곡은 **admission-선택적 정책에게만
  불리**해서, 현재 비교표가 RL-Cache를 체계적으로 과소평가하고 있다.
- 조치 옵션:
  - (권장) IcarusGym 쪽에 admission 의미론 추가: `TtlCache.put()`(또는 `IcarusGymLce`)에서
    `ttl ≤ ε`이면 삽입 생략. 플랫폼 논문 입장에서 "admission-류 알고리즘 지원 기능"으로 서술
    가능한 작은 확장. `ttl_sim`에도 동일 반영 후 re-validate + 재학습.
  - (차선) 현 substrate 유지 + 논문에 이 semantic과 실측 격차를 명시하고 no-insert 수치를 함께 보고.
- 주의: 훈련도 같은 substrate에서 이뤄지므로 현 모델은 "일시 삽입" substrate에 최적화되어 있음 —
  substrate를 고치면 **재학습 필요** (train_mc.py 재실행이면 충분).

### P1 — 강력 추천

#### P1-5. Icarus 네이티브 정책과의 비교 (PassiveAgentCache 활용)
- IcarusGym 논문 §III이 강조하는 `PassiveAgentCache`/`PASSIVE_ICARUSGYM_LCE`는 정확히 "RL 정책 vs
  legacy 교체 정책(LRU/LFU/FIFO)의 동일 조건 비교"를 위해 만든 기능인데, **rl-cache 실험이 이를
  전혀 사용하지 않는다.** 현재 baseline(AdmitAll/SecondHit)은 RLCachePolicy 내부 모드
  (`_force_admit`, `_secondhit`)로 구현되어 있음.
- Icarus 네이티브 LRU(/LFU/FIFO)를 같은 workload에서 돌려 비교 그림에 추가하면
  (i) 플랫폼 기능의 실사용 사례가 생기고 (ii) "다른 캐시 알고리즘들과 비교"라는 본래 목적의
  직접 증거가 된다. AdmitAll≈LRU 근사가 있긴 하나, **플랫폼 논문 관점에서는 native 비교 시연
  자체가 가치**다.

#### P1-6. AdaptSize 대체: best static size-threshold baseline (~30 lines)
- 현 결과에서 RL-Cache(OHR)는 사실상 **size threshold를 학습**한다(quartile별 admit
  1.00/1.00/0.22/0.06). 그렇다면 리뷰어의 첫 질문은 "튜닝된 static threshold보다 나은가?"이다.
- `ttl_sim`으로 threshold를 스윕해 `admit iff size ≤ θ*`의 best-θ 성능을 구하고 비교 테이블/그림에
  추가할 것. 원 논문의 AdaptSize 자리를 채우는 'AdaptSize-proxy'로 명시하면 (진짜 AdaptSize의 Markov
  모델 구현 없이) 비교의 공정성을 방어할 수 있다.
- RL-Cache가 best-static과 동급이면 "적응성(non-stationary, threshold 재학습)"으로, 능가하면 그대로
  세일즈 포인트로 서술.

#### P1-7. 다중 seed + 신뢰구간
- MC 훈련 분산이 크다(README에도 명시). 현재는 seed sweep에서 **best만** 보고 —
  TNSM에는 seed별 mean±std(또는 95% CI)도 필요하다.
- 제안: 훈련 seed {0..4} × eval trace {43,44,45} 매트릭스로 표/에러바 생성.
  `sweep_cache_size.py`의 `train_best`가 이미 seed 루프를 돌므로 mean/std 수집만 추가하면 됨.

#### P1-8. "훈련이 IcarusGym 밖"인 점의 논문 서술 전략 결정
- RL-Cache의 MC 학습은 same-window replay가 필요하고 IcarusGym의 blocking event loop는 이를
  미지원 → 훈련은 standalone `ttl_sim`(≈200 lines), 평가만 IcarusGym. 이 구조는 기술적으로 옳지만
  "IcarusGym이 RL 훈련 playground"라는 논문 주장과 긴장 관계에 있으므로 **서술 전략을 정해야 한다**:
  - (a) 원 논문 자체가 offline-train / online-serve 구조 → "훈련은 오프라인(플랫폼 검증 sim),
    서빙·평가는 IcarusGym"으로 자연스럽게 framing. 검증 스파이크(100% 일치)가 근거.
  - (b) 이 경험을 **플랫폼 요구사항 발견**으로 격상: replay/vectorized rollout API가 필요한 RL 계열
    (MC/ES류)이 존재함을 §VI 또는 discussion에서 명시, future work로 연결.
  - (c) IcarusGym에 trace-replay 헬퍼를 실제로 추가하고 플랫폼 기여로 서술.
  - 권장: (a)+(b) 병행. 숨기면 리뷰에서 잡힌다 — 명시가 안전하고, (b)는 오히려 논문의 기여가 된다.
- 부속 작업: **LOC 집계** ("몇 줄로 재현" 주장 뒷받침). 플랫폼 측 추가분(icarus_config의
  NonstationaryWorkload ~100 lines)과 에이전트 측 코드(rl_cache 패키지 ~1,000 lines)를 분리 집계.

### P2 — 코드/문서 정리 (시간 날 때)

1. **Stale 주석/문서 일괄 수정** (코드 공개 시 혼동 방지):
   - `rl_cache/rl_cache_policy.py` 모듈 docstring: "REINFORCE", "6-dim feature" 설명이 현재의
     MC + 8-dim과 불일치 (파일 상단 1–26행).
   - `rl_cache/rl_cache_agent.py`: `DEFAULT_CONFIG['feature_dim'] = 6`, 클래스 docstring의
     "REINFORCE" 서술.
   - `experiments/experiment0/config.py`: `NUM_EPISODES` 주석 "(= 100)", `CACHE_SIZE_MAX` 주석
     "(= 10)", `B_0` 주석 "(= 10.0)"(실제 100), `FEATURE_DIM = 8` 주석의 옛 feature 목록
     (lambda/ttl/hit — 현재 Table-1 8종과 다름).
   - `README.md`: "for the IcarusGym **TCCN** paper" → **TNSM**으로 수정.
2. **trace wrap-around 수정**: `trace_seed42.npz`가 20,001건이며 **마지막 1건이 다음 에피소드의
   첫 요청**(t=19790 → t=1.02로 역행). `extract_all`에는 reset이 없어 이 행의 temporal recency가
   음수가 되는 anomalous feature 1건이 훈련에 들어감. 추출 시 20,000건으로 트림 권장 (1 line).
3. **feature 구성의 참조 구현 대비 편차 문서화** (§4 조사로 우려 완화됨):
   - `f/s`, `f·s`를 log-size 기반으로 계산하는 것(`features.py:96-99`)은 **원 구현과 같은 방향**
     (원 구현도 log gdsf = −log(nobs)+log(1+size), log bhr = log(nobs)+log(1+size)로 log-domain
     composite 사용). 걱정했던 것과 달리 정합 — README에 "log-domain composites, as in the
     reference implementation" 한 줄만 추가하면 됨.
   - 다만 **recency EMA α가 다름**: 원 구현은 α=0.1 고정(C++ FeatureCollector), 우리는 0.5
     (`features.py` `ema_alpha=0.5`). 1-line 변경으로 정합 가능 — 정합 후 재학습해 민감도 확인 권장.
   - 원 구현의 입력 정규화(quantile 10-bin sparse 표준화 + EMA memory vector로 입력 2배)는
     훨씬 정교하지만, 우리의 단순 정규화는 **의도적 단순화**로 README에 명시하면 충분
     (adaptation 목적상 합리적).
4. **성능(real-trace 대비)**: `mc_trainer._score_sample`이 sample마다 `copy.deepcopy(base_cache)` —
   window당 m=400회. 100k–1M trace에서는 병목. dict shallow-copy(+불변 아이템) 또는 numpy 기반
   스냅샷으로 교체 검토.
5. **Legacy REINFORCE 경로 격리**: `main.py --mode train`(REINFORCE)은 README에 legacy로 명시돼
   있으나, 코드 공개 시 혼동 여지 → deprecation 주석 또는 `legacy/` 이동 고려.

---

## 3. 이번 세션의 실측 검증 노트 (2026-07-07)

**admit_ttl 민감도** — "admit_ttl=100초가 에피소드 20,000초 대비 너무 짧아 LRU와 다른 동작을
만들 것"이라는 가설을 세우고 `ttl_sim`으로 검증한 결과, **기각**:

```
admit_ttl=100      : AdmitAll OHR=0.0883 BHR=0.0472 | SecondHit OHR=0.0905 BHR=0.0481
admit_ttl=1000     : (동일)
admit_ttl=1e9      : (동일)
```
- 원인: 현 설정의 byte-capacity(100 slots × 1e4 = 1e6 bytes ≈ 카탈로그 bytes의 ~1%)가 매우 작아
  **capacity eviction이 TTL 만료보다 항상 먼저** 일어남. is_reset=True에서 eviction 순서
  (min expiration = least-recently-used)는 TTL 값과 무관하게 LRU와 동일.
- 함의: (i) "LRU-equivalent admission front end" 주장은 현 설정에서 실측 성립.
  (ii) 단, **large-cache regime**(real trace의 128GB급 실험 등 capacity 압력이 약한 경우)에서는
  TTL 만료가 먼저 발동해 LRU와 달라질 수 있음 → real-trace 실험 시 `ADMIT_TTL > eval horizon`으로
  설정할 것 (P0-3 체크리스트에 포함).
- 낮은 절대 hit rate(AdmitAll 8.8%)는 TTL 문제가 아니라 **작은 byte-capacity + α=0.8 +
  non-stationary phase** 조합의 결과로 설명됨 — 논문에는 이 workload 특성을 명시하면 충분.

**trace wrap-around 발견** — §2 P2-2 참조 (마지막 요청 1건이 t=1.02로 역행).

**reject 일시-삽입 영향 실측** (P0-4의 근거) — `ttl_sim`을 "reject 시 삽입 생략" 변형과 비교:

```
policy        substrate(TTL-render)          no-insert-on-reject
AdmitAll      OHR=0.0883  BHR=0.0472         OHR=0.0883  BHR=0.0472   (무영향)
SecondHit     OHR=0.0905  BHR=0.0481         OHR=0.0922  BHR=0.0499
RL-Cache      OHR=0.1719  BHR=0.0095         OHR=0.2157  BHR=0.0205
```
- reject된 객체가 size-aware `put()`에서 자리를 만들며 기존 항목을 축출하는 것이 원인.
  admission을 적극적으로 쓰는 정책일수록 손해 → 현재 표는 RL-Cache를 과소평가.
- (참고: 여기의 RL-Cache OHR 0.172는 README 표의 0.165와 소폭 다름 — model.pt 재학습 시점 차이로
  추정, 본질 무관.)

---

## 4. GitHub 참조 구현 대조 결과 (조사 완료, 2026-07-07)

`https://github.com/quovadim/RL-Cache` (구 WVadim; Keras/TF, Python 2)의 raw 소스를 직접 확인한
결과. **우리 구현에 유리한 확인 2건 + 정합 후보 몇 건:**

| 항목 | 원 구현 (파일) | 우리 구현 | 판단 |
|---|---|---|---|
| 출력층/손실 | `Dense(2, softmax)` + categorical CE (`environment/model.py`) | sigmoid 1-unit + BCE | **수학적으로 동등** — 문제없음 |
| 활성함수 | ELU | ELU | 일치 |
| 은닉층 | 폭이 줄어드는 피라미드 (`input_dim×multiplier`, multiplier 6–10, 5층, dropout 기본 0) | 64 고정 × 5층 | 사소한 차이 — 문서화만 |
| f/s, f·s | **log-domain** (log gdsf = −log(nobs)+log(1+size); log bhr = 합) | log-size 기반 composite | **같은 방향 — 우려 해소** (P2-3) |
| recency EMA α | **0.1 고정** (C++ `FeatureCollector.cpp`) | 0.5 | 정합 후보 (1-line + 재학습) |
| 입력 정규화 | quantile ~10-bin sparse 표준화 + EMA memory vector로 입력 2배 (λ≈0.99995) | 단순 [0,1] 정규화 | 의도적 단순화로 명시 |
| 윈도우 | K=200k, **overlap 100k (50% 겹침)**; 윈도우 뒤는 discount된 reward | K 단위 비겹침 슬라이드 + K+L 확장 | 논문 §3.2 서술과는 우리도 부합; 차이 문서화 |
| elite 선택 | 예제 config **percentile 90 (top 10%)** — 논문의 "top 20%"와 상이 | p=20% | 원 저자도 config로 조정 — 우리 값 유지 OK |
| elite target | **elite 세션의 sampled 0/1 결정에 CE** (기대값상 marginal admit rate와 동일 gradient); 옵션 `mc` 모드가 reward-가중 mean-action | 기본 'advantage'(의도적 변형, 문서화됨), 'marginal' 옵션 | 우리 'marginal'이 원 방식과 대응. 'advantage' 기본값은 소규모 필요악으로 이미 문서화 — 논문에는 both 보고 고려 |
| **reward** | **`alpha*BHR+(1−alpha)*OHR` 혼합 knob 내장** (예제 alpha=0 → OHR) | reward_mode {ohr,bhr} | **우리 BHR 변형이 원 코드의 설계 공간과 정렬** — 논문에서 인용 가치 (원 저자들도 BHR을 reward 후보로 코드에 넣어둠) |
| reject 처리 | **삽입 안 함** (`CacheSim.cpp: if (!admission_decision) return;`) | tiny-TTL 삽입 (일시 점유) | **실측 왜곡 확인 → P0-4** |
| eviction | ML-LRU는 logical-time rating으로 **정확한 LRU** | TTL-reset (현 설정에서 LRU-동등 실측, §3) | 일치 (현 regime 한정) |
| trace 포맷 | 공백 구분 `timestamp id size` | (time, content, size) npz | real-trace 전처리 시 동일 포맷 재사용 가능 |

핵심 액션: ① P0-4 (reject no-insert) 반영, ② EMA α=0.1 정합 실험, ③ README에 "원 구현 대비 차이"
표 추가 (softmax≡sigmoid, 피라미드 폭, 정규화 단순화, 비겹침 윈도우), ④ 논문에서 BHR-reward 변형을
"원 구현의 미사용 knob을 활성화한 것"으로 서술하면 방어력 상승.

---

## 5. 작업 이어가기 (Handoff)

- **브랜치/위치:** worktree `claude/vigilant-franklin-5504b9`
  (`/home/labry/git/rl-cache/.claude/worktrees/vigilant-franklin-5504b9`), 이 문서 커밋됨.
  main 브랜치 최신 커밋은 `203a879`.
- **환경:** `cd /home/labry/git/rl-cache && source .venv/bin/activate`
  (torch/ray/icarus 설치 완료; `trace_seed42.npz`, `model.pt`는 main checkout의
  `experiments/experiment0/`에 있음 — gitignored 아티팩트).
- **빠른 재개 순서 (회사에서):**
  1. `P0-1`: `extract_trace.py`에 `--seed` 추가 → seed 43 trace 추출 → `compare_baselines.py`/
     `sweep_cache_size.py`가 eval trace를 따로 받도록 수정 → 표 재생성. (반나절)
  2. `P0-2`: figure 3종 스크립트 (`plots/plot_paper_fig3.py`, `plot_paper_fig4.py`,
     `plot_active_bytes.py`). validated sim만 사용 → 빠름. (반나절)
  3. `P1-6`: static size-threshold baseline을 `baselines.py`에 추가 + sweep에 편입. (1–2시간)
  4. 그 다음 `P0-3` real trace 착수 — `docs/real_trace_eval_plan.md`의 TODO #1(트레이스 다운로드,
     인터넷 되는 머신에서)부터.
- **검증 명령 (환경 sanity):**
  ```sh
  python experiments/experiment0/train_mc.py --smoke     # MC trainer 동작 확인
  python experiments/experiment0/compare_baselines.py    # IcarusGym vs sim 교차검증
  ```
