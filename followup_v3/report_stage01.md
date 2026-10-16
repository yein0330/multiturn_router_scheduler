# followup_v3 단계 0–1 보고서

- **결론**: 관문 1 미달(3개 조건 중 2개 미달). 지시서 규칙에 따라 단계 2로 가지 않고 멈춘다. 손잡이 조정안은 §5, 사용자 결정이 필요한 항목은 [next_decision.md](next_decision.md)에 있다.
- **표기**: [측정] 실행·측정값, [추정] 비용 모델 계산값, [가설] 미검증 해석.
- **성격**: 모든 결과는 사후 탐색이다. 설계는 단계마다 실행 전에 commit했다.
  - 단계 0: `651d3d0`
  - 단계 1: `4561343`
  - paired 지점 선정: 측정 전 commit
- **한정 조건**: 모델 쌍 1개(Qwen3-1.7B / 8B), 고정 라우터 R, 동시성 1, 전용 GPU. 단계 1은 dev 50세션(도메인 hotel·gym)이다. eval 데이터는 생성·동결만 하고 실행하지 않았다.

## 0. 실행·감사 [측정]

| 실행 | 기대/고유/중복 | 호출 | 캐시 예측 불일치 | 잘림 | HTTP 오류 | 비고 |
|---|---|---|---|---|---|---|
| 0-3 라우터 재점수(v1–v2 로그) | 3,937 / 801 고유 프롬프트 | 801×3 + 40×3 (greedy 1 + 후보 점수 2) | – | – | 0 | 토큰 수 일치 100%, greedy 재생성 일치 99.75%, 점수 argmax 일치 99.70% |
| 1 smoke(dev) | 60 / 60 / 0 | 1,090 | 0 | 0 | 0 | 도구 호출 없는 턴 33, 예산 소진 요청 30 |
| 1-2 dev 50세션 | 300 / 300 / 0 | 5,558 | **0** | 19 | 0 | 도구 호출 없는 턴 125, JSON 파싱 실패 턴 12, 도구 오류 1,455/5,443, 예산 소진 요청 212, 라우터 fallback 0 |
| 1-3 프로파일 격자 | 360 / 360 | 360 + 준비 240 | 적중 = 기대 360/360 | – | 0 | |
| 관문 근처 paired | 144 | 144 | matched state 144/144 | – | 0 | 반복 간 출력 변동 0 |
| dev50 라우터 오프라인 점수 | 1,803 | – | – | – | – | 1차 시도 실패, 재실행(아래) |

- 모든 GPU 작업은 `run_locked.sh`(배타적 flock)로 단독 실행했다. 시작 전 GPU 1–3이 비어 있는지 확인했고, 이 작업이 띄운 서버만 종료했다. 실행 기록은 `results/locked_runs.log`에 있다.
- **이상 1 — R 엔진 OOM**: dev50 라우터 오프라인 점수 1차 시도에서 R 엔진이 죽었다. `prompt_logprobs`가 1,413토큰 프롬프트 전체에 log-softmax를 계산하며 약 820 MiB를 요구했는데, `gpu-memory-utilization 0.90`에서는 약 600 MiB만 남아 있었다.
  - 오프라인 점수 전용 설정(`configs/models_R_offline_scoring.json`, 0.80)으로 다시 실행했다. 가중치·revision은 같고 KV 풀만 작아졌다.
  - 정책 실행에는 쓰지 않았다.
- **이상 2 — 대기 조건 오류**: 단계 1 체인 완료 감지 조건이 이전 실행 로그의 기록과 겹쳐, 18:36 완료 후 약 80분 동안 다음 분석이 진행되지 않았다. 실험 데이터에는 영향이 없다.
- 제외한 실행은 없다. 감사표는 [results/stage01/audit.json](results/stage01/audit.json)에 있다.

## 1. 단계 0 (기존 로그, 새 정책 추론 없음)

### 0-1. 사후 선택 여지 — [results/stage0/](results/stage0/) [측정]

- 품질 제약: 세션별 q_i ≥ q_i(BBBBB).
- **"실행한 정책 경로 집합 내 사후 최선"이다.** 경로를 이어 붙이지 않았다. 전역 oracle이나 달성 가능한 상한이 아니다.
- cross-fit은 시나리오 정보를 부여한 진단 기준선이며, 정책이 아니다.

| 조건 | 집합 | 평균 지연(BBBBB 대비) | q | S |
|---|---|---|---|---|
| short | BBBBB | 11.63 s (1.00) | 0.820 | 0.50 |
| short | F2 = {AAAAA, BBBBB} | 9.17 s (0.79) [8.07, 10.32] | 0.868 | 0.56 |
| short | F6 고정 schedule | 7.97 s (0.69) | 0.872 | 0.56 |
| short | F6 + P4/P6/P10 | 7.91 s (0.68) | 0.872 | 0.56 |
| short | cross-fit F6(200분할) | 10.36 s (0.89) | **0.763** | 0.44 |
| long | BBBBB | 14.72 s (1.00) | 0.824 | 0.50 |
| long | F2 | 11.96 s (0.81) | 0.840 | 0.54 |
| long | F6 | 11.37 s (0.77) | 0.848 | 0.56 |
| long | F6 + P4/P6 | 11.32 s (0.77) | 0.848 | 0.56 |
| long | cross-fit F6 | 14.21 s (0.97) | **0.796** | 0.46 |

- 사후 선택 여지의 대부분은 세션별 A/B 이분 선택(F2)에서 나온다. 후보를 F6+로 넓혀서 더 얻는 것은 short 1.26 s(BBBBB 지연의 10.8%), long 0.63 s(4.3%)다.
- cross-fit은 평가 반쪽에서 품질 제약을 지키지 못했다(q가 BBBBB보다 낮음). 시나리오 정보로 schedule 하나를 고르는 것도 세션 간 변동을 감당하지 못했다.
- μ 스칼라화 결과(μ ∈ {0.5, 1, 2, 4}, T_ref는 평가 BBBBB 평균으로 정한 **사후 정규화**)는 `oracle_summary.csv`에 있다.
- 분기 40점의 남은 단계당 지연 차(B − A): short 1.6–2.1 s, long 2.8–3.3 s → `branch_per_remaining_stage.csv`

### 0-2. "A로 충분한가" 신호 — [가설], 탐색

| 레이블 | 조건 | 양성 비율 | 시점 | 프로토콜 | AUC | Brier(기준선) |
|---|---|---|---|---|---|---|
| q_A ≥ q_B | short | 0.34 | initial | dev30 → eval50 | 0.51 | 0.229 (0.224) |
| q_A ≥ q_B | short | 0.34 | after-first-A | dev30 → eval50 | 0.57 | 0.224 (0.224) |
| q_A ≥ q_B | long | 0.36 | initial / after | 시나리오 LOO | 0.34 / 0.41 | 0.284 / 0.274 (0.230) |
| AAAAA 사전식 최선 | short | 0.26 | initial / after | 시나리오 LOO | 0.14 / 0.20 | 0.254 / 0.263 (0.192) |
| AAAAA 사전식 최선 | long | 0.26 | initial / after | 시나리오 LOO | 0.22 / 0.28 | 0.256 / 0.256 (0.192) |

- 특징: 첫 요청의 연산군과 토큰 길이. after-first-A 시점에는 첫 A 응답 길이와 첫 verifier 결과를 더했다. 후자는 이미 낸 A 호출 비용(short 0.40 s, long 0.62 s 평균)을 포함해야 한다.
- **쓸 만한 신호는 없었다.** AUC가 0.5 이하인 것은 n=50, 시나리오 LOO, 원-핫 연산군에서 생긴 과적합으로 보인다. 둘 다 0인 세션은 short 0%, long 2%다.

### 0-3. 라우터 점수 — [results/stage01/router_score_audit.json](results/stage01/router_score_audit.json)

- **tokenization [측정]**: `SMALL` = [SM, ALL], `LARGE` = [L, ARGE]로 다중 토큰이고 첫 토큰부터 다르다. 그래서 첫 토큰 logprob 차는 쓰지 않았다.
- **점수**: 후보 완성 문자열 전체(`…<|im_end|>`)의 조건부 로그우도 차(`prompt_logprobs`)다. 결정 하나당 프롬프트 전용 R 호출 2회가 추가된다. 따라서 "추가 LLM 호출 없음" 조건을 충족하지 않는다.
- **재현 [측정]**:
  - 프롬프트 토큰 수 일치 100%(기록에 해시가 없어 해시 대조는 불가능)
  - greedy 재생성 일치 99.75%
  - 점수 argmax = 기록 결정 99.70%
  - 불일치 20행(고유 프롬프트 5개)은 모두 |margin| ≤ 0.75인 동점 근처였다.
- **동일 상태 레이블에 대한 AUC [측정, 소표본]**:

  | 레이블 출처 | n | B 필요 | margin AUC |
  |---|---|---|---|
  | v2 분기, 다음 호출 | 40 | 13 | 0.90 |
  | v1 첫 요청 | 10 | 3 | 1.00 |
  | 합 | 50 | 16 | 0.89 |
  | 참고: 다른 경로 대리 레이블(P4, report §4) | 250 × 2 | 86 / 75 | 0.69 / 0.69 |

  - 첫 요청 10개에서 R은 모두 SMALL을 골랐지만, margin 순위는 B가 필요한 3개를 정확히 위에 두었다. [가설] R의 출력 결정보다 연속 점수가 정보가 많다. n이 작다.
- ρ 매핑: dev 라우터 호출 margin의 경험적 CDF로 정의하고 동결했다(`configs/rho_mapping.json`). 백분위이지 확률이 아니다.
- **v3 과제에서의 점수 인터페이스: 중단.** dev50 L0 1,803호출의 오프라인 점수 argmax 일치율이 **98.78%(< 99%)** 였다. 불일치 22건은 모두 |margin| ≤ 1.25인 동점 근처다. 도구 세션에서는 동점 근처 비율이 3.5%로 단계 0의 1.9%보다 높다. 규칙에 따라 v3에서는 decision-only 분석을 기본으로 한다.

## 2. 단계 1 — 도구 호출형 표 질의 과제

### 1-1. 생성기 검증 [측정]

- 참조 해법(환경 경유) = 독립 직접 계산: 1,200/1,200 요청 일치.
- 무작위 도구 정책의 요청 통과율: 0.0067.
- 렌더링 예시: [examples/dev-chain-00_always_B_short_last_prompt.txt](examples/dev-chain-00_always_B_short_last_prompt.txt)
- max_tokens 256: smoke에서 잘림 0, 최대 174토큰이라 동결했다. dev50에서는 잘림이 19회(0.34%) 있었다.

### 1-2. dev 50세션 live [측정]

| 조건 | 정책 | 요청 통과 q | 세션 성공 S | 호출/세션 | 지연 s | 전환 | B 비율 |
|---|---|---|---|---|---|---|---|
| short | always_A | 0.405 | 0.14 | 20.9 | 4.57 | 0 | 0 |
| short | always_B | 0.855 | 0.54 | 14.9 | 10.17 | 0 | 1 |
| short | L0 | 0.625 | 0.24 | 17.9 | 8.59 | 4.7 | 0.47 |
| long | always_A | 0.130 | 0.02 | 24.4 | 6.02 | 0 | 0 |
| long | always_B | 0.790 | 0.42 | 14.8 | 11.38 | 0 | 1 |
| long | L0 | 0.575 | 0.20 | 18.1 | 12.15 | 4.3 | 0.71 |

- 출력: A 평균 38.5 / p95 76토큰, B 평균 38.2 / p95 65토큰. v2(약 140토큰)의 약 1/4이다.
- 문맥: short는 p50 1.7K / p95 3.2K, long은 p50 4.3K / p95 9.9K / 최대 21K토큰이다.
- L0는 세션당 4.3–4.7회 전환했다(v2의 약 1회보다 많음). long에서는 항상 B보다 **느리면서** 품질이 낮았다(12.15 s, q 0.575 vs 11.38 s, q 0.790).

### 1-3. 비용 프로파일과 표현 [측정]

hold-out 호출 총지연 오차(MAE / p95, 초). hold-out은 문맥 8K 또는 출력 48인 cell이다.

| 모델 | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| A | 0.131 / 0.357 | 0.091 / 0.234 | 0.007 / 0.017 | 0.007 / 0.016 |
| B | 0.405 / 1.407 | 0.167 / 0.428 | 0.007 / 0.016 | 0.019 / 0.062 |

- C0와 C1은 절편이 음수다(선형식이 attention 항을 흡수하지 못함).
- live 호출 5,558건에 오프라인으로 붙인 예측 오차(MAE)는 다음과 같다(`live_call_cost_errors.csv`).
  - **실현 출력 길이 사용**: C2/C3 0.003–0.016 s, C1 0.05–0.09 s, C0 0.07–0.28 s
  - **결정 시점의 ŷ(고정 평균) 사용**: B는 모든 표현이 0.21–0.22 s, A는 0.06–0.15 s
  - [측정] 결정 시점의 비용 오차는 식 형태보다 **출력 길이 추정이 지배**한다.

### 관문 1 — [results/stage01/gate1.json](results/stage01/gate1.json), [gate_disagreement.csv](results/stage01/gate_disagreement.csv)

결정 지점은 L0에서 현재 모델 ≠ R 추천인 호출이다(short 236, long 217, 전체 L0 호출 897/906). 비용만 비교한 선호이며, 정책 결정이 아니다.

| 조건 | 기준 | 결과 | 판정 |
|---|---|---|---|
| (a) C1–C2 / C1–C3 불일치 | 한 조건 이상에서 ≥ 15%, 분모 ≥ 30 | 최대 **1.4%**(long h=1, 217점 중 3점). 다른 모든 셀 0% | **미달** |
| (b) S(always_B) − S(always_A) | ≥ 0.15 | short 0.40, long 0.40 | 통과 |
| (c) q_A ≥ q_B 세션 비율 | 20–60% | short **18%**, long **14%**(둘 다 0인 세션 0% / 2%) | **미달** |

- **(a)가 미달인 이유 [측정]**: 비용만 비교하면 방향이 결정을 정한다.
  - A→B 추천이면 B 쪽이 항상 더 비싸다(follow 0%).
  - B→A 추천이면 A 쪽이 항상 더 싸다(follow 97–100%).
  - A의 cold prefill이 warm B의 decode보다 싸서, 캐시 상태로는 비용 순서가 뒤집히지 않는다. 손익분기선은 [fig_breakeven_plane.png](results/stage01/fig_breakeven_plane.png)에 있다.
  - C0 쌍도 마찬가지로 0–1.4%다.
- **비용 차이의 크기는 표현마다 다르다** [추정](`deltaC_by_representation.csv`).
  - 결정 지점의 ΔC(follow − keep, h=1) 중앙값은 ±0.4–0.7 s다.
  - C1과 C2의 ΔC 차이는 중앙값 0.015–0.053 s(p90 0.05–0.18 s)로, ΔC의 약 5–10%다.
  - C0와 C2의 차이는 long B→A에서 중앙값 0.28 s로, ΔC의 약 60%다.
  - [가설] 품질과 결합하는 결정(단계 3의 L2)에서는 C0가 결정을 바꿀 수 있다. C1 대 C2/C3의 차이는 작다.
- **관문 근처 paired [측정]**: 24점 × keep/follow × 3회, matched state 144/144, 출력 변동 0.
  - 비용 선호의 부호가 실측과 맞는 비율: C2/C3 96%, C0 96%, C1 92%.
  - long B→A 6점에서는 실측 차가 평균 −0.04 s(거의 0)였다. C1은 +0.01(부호 오류), C2는 −0.24(과대)로 예측했다.
  - **표현이 갈리는 영역은 실재한다. 다만 그 영역의 실측 이득(평균 −0.04 s)은 반복 잡음(범위 중앙값 2.7 ms)보다는 크지만, 실용적으로는 작다.**

## 3. 관측 → 허용되는 주장 → 아직 불가능한 주장

| 관측 | 허용되는 주장 | 아직 불가능한 주장 |
|---|---|---|
| v2 과제에서 사후 선택 여지의 대부분이 A/B 이분 선택에 있음 | 실행한 정책 집합 안에서는 세션별 초기 A/B 선택이 가장 큰 사후 여지다 | 실행 가능한 초기 선택 신호가 있다(0-2에서 찾지 못함) |
| 동일 상태 레이블에서 R margin AUC 0.89(n=50) | R의 연속 점수는 출력 결정보다 B 필요를 더 잘 순위화하는 경향이 있다 | 보정된 성공 확률이다. 새 과제로 전이된다(v3에서는 재현율 98.8%로 점수 인터페이스 중단) |
| v3 결정 지점에서 C0–C3 비용 선호 불일치 ≤ 1.4% | 비용만 비교하는 keep/follow에서는 이 모델 쌍·과제 영역에서 표현 정밀도가 결정을 바꾸지 않는다 | 품질과 결합한 결정에서도 표현이 무관하다(미검증. ΔC 크기 차는 C0에서 큼) |
| 실현 길이로는 C2/C3 오차 수 ms, ŷ로는 0.2 s | 결정 시점 비용 오차는 출력 길이 추정이 지배한다 | 더 정밀한 비용 식이 결정 가치를 만든다(H1 미지지) |
| long에서 L0가 항상 B보다 느리고 품질도 낮음 | 이 R의 매 호출 라우팅은 이 과제·조건에서 항상 B에 지배된다 | 매 호출 라우팅 일반이 불리하다 |
| A 충분 비율 14–18% | 이 과제는 A에게 너무 어렵다(관문 (c) 미달) | 과제 조정 후의 결과 |

## 4. 남은 불확실성

- 단계 1의 수치는 dev 도메인 2개, 50세션에서 나왔다. eval에서 재확인하지 않았다.
- R은 도구 메시지를 ASSISTANT로 표기하는 고정 프롬프트로 동작한다. 도구 과제에 맞춘 라우터가 아니다.
- h=N̂는 이후 호출 비용을 상수로 근사했다(N̂, Δ는 smoke에서 고정). 두 행동이 이후 호출 수를 바꾸는 효과를 반영하지 않는다.
- 비용 오차는 지연 단위(초)다. 가격형(C1-price)은 단계 3 범위라 하지 않았다.
- 프로파일은 pure-system(같은 token prefix)이다. live 결정 지점의 (c_A, c_B) 분포는 대부분 "현재 모델 warm / 대상 cold 또는 partial"이다.

## 5. 손잡이 조정안 (관문 1 미달 → 사용자 결정 필요, 실행하지 않음)

1. **(c) A 충분 비율 올리기**: 요청 혼합에서 hard 비율을 낮추거나(예: H 요청 25%), 모호한 열 이름 대신 정확한 열 이름을 쓴다. long에서는 A가 거의 실패하므로(q 0.13), long 손잡이를 완화한다(filter 미리보기 15 → 5행).
2. **(a) 비용 표현이 결정을 바꿀 수 있는 영역**:
   - (i) 비용만 비교하지 않고 공통 품질 항과 결합한 결정(μ)으로 관문을 정의한다. 표현 간 ΔC 크기 차이가 결정에 반영된다. 단 이것은 단계 3의 L2와 같은 구성이라, 관문 1의 정의 변경이 필요하다.
   - (ii) 속도 차가 작은 모델 쌍이면 캐시 상태가 순서를 뒤집을 수 있다. 사용자 기본값(2모델 고정)을 바꿔야 한다.
   - (iii) 문맥 ≥ 24K와 매우 짧은 출력의 조합이다. 다만 이 과제에서는 A의 품질이 더 떨어진다.
3. 위 조정은 dev에서만 한다. eval은 조정이 끝난 뒤 한 번만 실행한다. 조정 시도 횟수는 모두 기록한다.

## 6. 재현

```bash
cd ~/multiturn_routing_pilot
# 단계 0 (CPU)
.venv/bin/python followup_v3/src/stage0_oracle.py
.venv/bin/python followup_v3/src/stage0_signal.py
followup_v3/run_locked.sh "R" .venv/bin/python followup_v3/src/stage0_router.py
followup_v3/run_locked.sh "R" .venv/bin/python followup_v3/src/stage0_router_labels.py score
.venv/bin/python followup_v3/src/stage0_router_labels.py analyze
# 단계 1
(cd followup_v3/src && for w in smoke dev train eval; do ../../.venv/bin/python make_v3_data.py $w; done)
followup_v3/run_locked.sh "A B R" .venv/bin/python followup_v3/src/agent_runner.py --sessions followup_v3/data/smoke.jsonl --run-id smoke
followup_v3/run_locked.sh "A B R" followup_v3/stage1_chain.sh          # profile grid + dev50
.venv/bin/python followup_v3/src/profile_v3.py fit
.venv/bin/python followup_v3/src/gate1.py analyze && .venv/bin/python followup_v3/src/gate1.py select-paired
followup_v3/run_locked.sh "A B" .venv/bin/python followup_v3/src/gate1.py measure-paired
MODELS_CFG=followup_v3/configs/models_R_offline_scoring.json followup_v3/run_locked.sh "R" .venv/bin/python followup_v3/src/stage1_rho.py
.venv/bin/python followup_v3/src/gate1.py gate
```
