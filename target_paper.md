# IcarusGym 차기 구현 대상 논문 (Target Papers)

**작성:** 2026-07-07 · 멀티에이전트 조사 (venue sweep 5방향 → 후보 21편 → 검증 9편)
**요구조건:** rl-cache처럼 IcarusGym에 올려 다른 캐시 알고리즘과 비교 가능할 것, 최근 1–2년
(2024-07~2026-07), top-level conference 또는 주요 저널만, MDPI 제외, single-agent만(MARL 제외).

---

## 결론: 추천 2편 (같은 그룹의 동반 논문 — 구현 1회로 2편 커버)

두 논문은 **같은 저자 그룹(Niknia, Wang et al.)의 동일한 SMDP 정식화**를 공유한다.
State·action·reward가 사실상 같아서, observation builder + reward + PPO를 한 번 만들면
두 번째 논문은 증분(attention replay 또는 TL 모듈)만 추가하면 된다.

### 1순위 — Niknia & Wang, PPO + Transfer Learning (IEEE IoTJ 2025)

> F. Niknia, P. Wang, et al., **"Edge Caching Optimization With PPO and Transfer Learning
> for Dynamic Environments,"** *IEEE Internet of Things Journal*, vol. 12, no. 16,
> pp. 33605–33620, Aug. 15, 2025. DOI: [10.1109/JIOT.2025.3577904](https://doi.org/10.1109/JIOT.2025.3577904)
> (arXiv: [2411.09812](https://arxiv.org/abs/2411.09812))

- **검증 상태:** Crossref로 서지 확인 완료(철회/정정 없음), single-agent 확인, 적합도 **8/10**.
  이전 세션에서 선정했던 바로 그 논문 — 이번 조사로 **선정이 재확인됨** (9편 중 최고점).
- **MDP:** 단일 edge cache, decision epoch = 각 요청(Poisson 도착)의 SMDP.
  - State (catalog-wide, F=50): 여유 메모리 Mem(t), 캐시 여부 b(t)∈{0,1}^F, utility y(t)
    (freshness×importance), 인기도 d(t), importance i / lifetime l / size z 벡터.
  - Action: 요청된 파일의 **binary admission** a(t)∈{0,1} — rl-cache와 동일 패러다임.
  - Reward: r(t) = w1·[(b·d)(b·y)^T] − w2·Mem(t).
  - 학습: PPO(actor/critic MLP, γ=0.99) + 환경 변화 감지(도착률 이동평균, 인기도 cosine 유사도)
    시 **Transfer Learning**(critic 이식 + actor 재초기화 + demonstration replay, DQfD-style loss).
- **IcarusGym 매핑:** admission → TTL rendering (admit=큰 TTL, reject=미세 TTL) 그대로 재사용.
  파일 lifetime [10,30]은 문자 그대로 TTL → **TtlCache env와 자연스럽게 부합**.
  Zipf+Poisson workload는 Icarus 기본 기능. 새로 만들 것: catalog-wide observation wrapper
  (~6F+1=301차원), PPO(표준), TL 모듈(최대 작업), 비교 baseline(LFS는 공짜, DQfD 표준,
  QDTRL은 선택적).
- **재현 대상 그림:** Fig. 2(초기 환경 PPO 수렴), Fig. 4(인기도 변화 후 적응: TLP vs LFS 최소),
  Fig. 6(요청률 변화 후 적응). 보조: Fig. 3/5 감지기 출력, Table III/IV 감지 지연.
- **리스크:** 공개 코드 없음(수식에서 전부 재구현; w1/w2 스케일, freshness 함수, 캐시 만석 시
  동작이 저술로만 정의됨). hit-ratio vs cache-size 그림이 없어 재현 대상은 reward/수렴/적응
  곡선 — LRU/LFU 비교는 PassiveAgentCache로 만드는 **플랫폼 추가 기여**로 명확히 라벨링할 것.

### 2순위 (동반 구현 추천) — Niknia et al., Attention-Prioritized PPO (IEEE TVT 2025)

> F. Niknia, P. Wang, et al., **"Attention-Enhanced Prioritized Proximal Policy Optimization
> for Adaptive Edge Caching,"** *IEEE Transactions on Vehicular Technology*, vol. 74, no. 8,
> pp. 12853–12863, Aug. 2025. DOI: [10.1109/TVT.2025.3552084](https://doi.org/10.1109/TVT.2025.3552084)
> (arXiv: [2402.14576](https://arxiv.org/abs/2402.14576), DBLP: journals/tvt/NikniaWWAR25)

- **검증 상태:** IEEE Xplore/DBLP 확인, single-agent, 적합도 **7/10**. TVT는 IEEE 주요 저널
  (단, IoTJ/ToN급 목록 밖이라는 점은 유의 — 그래도 "주요 저널" 기준은 충족).
- **왜 추천:** 1순위와 **MDP가 동일** (state/action/reward 같음; eviction은 lowest-utility
  rule). 차이는 학습 쪽 novelty — attention(KQV) 기반 prioritized replay를 PPO에 결합.
  1순위 구현 후 **추가 비용이 가장 작은 논문** (TL 모듈 대신 attention replay만 교체).
- **결정적 장점:** Figs. 6–7이 **hit count / utility vs cache capacity** — IcarusGym의 표준
  "hit ratio vs cache size" sweep 템플릿과 정확히 일치. Figs. 2–3은 Zipf skewness η∈[0,1]
  sweep(1000-request 평가, F=50)이라 소규모 재현에 최적. Table II(MDP vs SMDP)는
  event-triggered TtlCache 매핑의 타당성을 검증해주는 보너스 ablation.
- **리스크:** 공개 코드 없음, PPO(on-policy)에 prioritized replay를 섞는 비표준 구조(learning
  rate가 범위로만 보고됨 → 수렴 곡선 정밀 재현 리스크), 비교 대상 CTD/RLTD는 제3자 DRL
  기법이라 충실 재현하려면 별도 구현 필요(단순 baseline 대체 가능).

**제안 실행 순서:** 공통 기반(obs wrapper + reward + PPO + TtlCache 매핑) → 1순위의 TL 모듈
+ Fig 2/4/6 재현 → (여력 시) 2순위의 attention replay + Fig 6–7 재현. TNSM 논문 §VI.B의
세 번째 case study로는 1순위를 쓰고, 2순위는 옵션(4번째 case study 또는 대체재).

---

## 탈락/보류 후보 (검증 9편 중 나머지)

| 논문 | Venue | 점수 | 탈락 사유 |
|---|---|---|---|
| 3L-Cache | **FAST '25** (코드 공개, artifact badge) | 6 | **RL이 아님**(GBM supervised eviction). eviction-hook env 신설 필요, >10M-request 스케일 전제(소규모에서 학습 안 됨). top venue+코드라 **백업 1순위**로 보류 가치 |
| Semantic Caching for LLM Serving | INFOCOM 2026 | 6 | LLM 질의-응답 캐시 — Icarus 콘텐츠 요청 모델과 도메인 불일치 |
| Latency-Aware Caching w/ Delayed Hits | NSDI '26 | 3 | 파이프라인 아키텍처 중심, RL 없음 |
| CAPSULE (prefetcher) | SIGMETRICS 2026 | 2.5 | 스토리지 prefetch, 도메인 불일치 |
| HeatCache | INFOCOM 2026 | 2 | TCAM rule caching — 콘텐츠 캐싱 아님 |
| PaperCache | HotStorage '25 | 1 | 워크숍(6쪽), 동적 정책 전환 시스템 |
| Heimdall | EuroSys '25 | 1 | 스토리지 I/O admission, ML 파이프라인 논문 |

조사 범위: USENIX(FAST/NSDI/ATC/OSDI/EuroSys), ACM(SIGCOMM/CoNEXT/SIGMETRICS/ICN/WWW),
IEEE(INFOCOM/ICNP/ToN/JSAC/TNSM/TMC/IoTJ/TCCN/TVT), Elsevier(ComNet/ComCom/PerfEval), ICN/NDN
계열. 2024-07~2026-07 window에서 single-agent RL로 단일 캐시의 admission/eviction/TTL을
제어하는 top-venue 논문은 위 후보가 사실상 전부였다 — 이 시기 상위 학회 캐싱 논문의 주류는
(i) learned-index/GBM류 비-RL 기법, (ii) MARL 협력 캐싱(스코프 제외), (iii) LLM 서빙 캐시로
이동해 있음. 순수 single-agent RL 캐싱은 IEEE 저널 쪽에 집중되어 있고, 그중 최상급이 위
Niknia & Wang 페어다.

---

## 다음 단계 (구현 착수 시)

1. arXiv 버전([2411.09812](https://arxiv.org/abs/2411.09812),
   [2402.14576](https://arxiv.org/abs/2402.14576)) PDF 확보 → Table II(하이퍼파라미터) 및
   utility/freshness 함수 정의 정독.
2. dehghan-cache의 utility 부기(freshness/importance)와 겹치는 부분 재사용 검토.
3. 새 repo(예: `~/git/niknia-cache`)에 rl-cache 구조(train 분리 + IcarusGym eval + validated
   sim 패턴) 복제 — 단, 이 논문들은 on-policy PPO라 **IcarusGym 루프 안에서 직접 학습 가능**
   (rl-cache의 MC replay 제약이 없음 → "IcarusGym이 RL 훈련 playground" 서사에 오히려 더 적합).
4. rl-cache의 남은 P0 작업(improve_points.md)을 먼저 마무리한 뒤 착수 권장.
