# multiturn_routing_pilot

멀티턴 LLM 작업에서 라우팅 빈도(매 턴, 세션 시작, 주기, 관측 사건)와 캐시 비용 보정이 품질–지연에 미치는 영향을 비교하는 **탐색 pilot**이다. 새 라우터를 학습하지 않으며, 특정 정책의 우월성을 입증하려는 실험도 아니다.

| 문서 | 내용 |
|---|---|
| [design.md](design.md) | 연구 질문, 과제·시나리오, 정책, 캐시 규칙, 지표, 사전 예측, 동결 기록 |
| [report.md](report.md) | 시나리오별 결과, 개선·악화 사례 |
| [limitations.md](limitations.md) | 모델·데이터·평가·비용 측정 한계 |
| [decision.md](decision.md) | 어떤 단순 정책이 충분했고 어디서 판단이 필요했는지 |

## 환경

- 서버: RTX 4090 × 4, 드라이버 535.261.03(CUDA 12.2). GPU 0은 Xorg가 쓰고 있어 사용하지 않는다.
- 모델(가중치 상주, bf16, TP=1):
  - A = Qwen3-1.7B(GPU 1, :8201)
  - B = Qwen3-8B(GPU 2, :8202)
  - R = Qwen3-4B-Instruct-2507(GPU 3, :8203, 라우터 전용)
  - revision은 [configs/models.json](configs/models.json)에 있다.
- 엔진: vLLM 0.19.1(cu128 마지막 빌드, 드라이버 제약). Python 3.12 `.venv/`(git 제외).
- 유료 API, 토큰, 원격 저장소를 쓰지 않는다. git은 로컬에만 있다.

```bash
python3.12 -m venv .venv
~/switchcost/.venv/bin/uv pip install --python .venv/bin/python vllm==0.19.1 httpx pandas matplotlib pytest
```

## 재현

```bash
export PYTHONPATH=src
.venv/bin/python -m mrp.serve start configs/models.json          # A, B, R 서버
.venv/bin/python -m mrp.make_data dev                             # dev 세트
# dev smoke / 보정
.venv/bin/python -m mrp.runner --sessions data/dev/calib_single.jsonl --policies P1,P2,P4 --run-id dev_calib_single
.venv/bin/python -m mrp.runner --sessions data/dev/smoke.jsonl --policies P1,P2,P3,P4,P5,P6 --run-id dev_smoke
.venv/bin/python -m mrp.runner --sessions data/dev/calib_sessions.jsonl --policies P1,P2,P4 --run-id dev_calib_sessions
# B1 비용표 (profiles/cost_profile.jsonl, fit_latency.json; fit.json = 여기에 dev 기반 g_hat를 더한 것)
.venv/bin/python -m mrp.profile --transcript-run dev_smoke --session dev-S1_constant-00
# 동결 후
.venv/bin/python -m mrp.make_data eval
.venv/bin/python -m mrp.runner --sessions data/eval/eval.jsonl --policies P1,P2,P3,P4,P5,P6,P7,P8 \
    --context short --cache retain --run-id evalA --cost-fit profiles/fit.json
.venv/bin/python -m mrp.runner --sessions data/eval/eval.jsonl --policies P1,P2,P4,P6,P7,P8 \
    --context long --cache retain --run-id evalB_long_retain --cost-fit profiles/fit.json
.venv/bin/python -m mrp.runner --sessions data/eval/eval.jsonl --policies P4,P6,P7,P8 \
    --context short --cache evict --run-id evalB_short_evict --cost-fit profiles/fit.json
.venv/bin/python -m mrp.runner --sessions data/eval/eval.jsonl --policies P4,P6,P7,P8 \
    --context long --cache evict --run-id evalB_long_evict --cost-fit profiles/fit.json
.venv/bin/python -m mrp.analyze --runs evalA --out results/analysis_A --timeline "..."
.venv/bin/python -m mrp.serve stop configs/models.json
```

- 실행기는 `results/<run>/sessions.jsonl`에 이미 있는 (세션, 정책, 조건)을 건너뛴다. 따라서 중단 후 같은 명령으로 이어서 실행할 수 있다.
- 정책 실행마다 세 서버의 prefix cache를 reset한다. 다른 사용자의 GPU 작업이 있으면 지연 수치가 오염되므로 실행 전에 `nvidia-smi`를 확인한다.

## 구조

```
configs/   models / policies / cache / eval 설정 (동결 commit 4c44031)
data/      dev(smoke, calib_sessions, calib_single), eval(eval.jsonl) + manifest(seed, sha256)
profiles/  B1 비용 측정(cost_profile.jsonl), 셀 요약(cost_cells.csv), 적합(fit_latency.json, fit.json)
results/   run별 turns.jsonl(요청·응답·라우팅·게이트·캐시·지표) / sessions.jsonl(평가 포함), analysis_*/
src/mrp/
  tasks.py     절차적 과제(도메인, 연산군, 참조 구현, 시나리오 레이아웃)
  sandbox.py   제출 코드 실행: 숨김 evaluator / policy-visible verifier
  client.py    token-id 스트리밍 호출, Prometheus 차분, 클라이언트 캐시 추적
  router.py    고정 zero-shot 라우터 R
  costmodel.py C_hat와 비대칭 게이트
  runner.py    정책 P1–P8 live 세션 실행
  profile.py   B1 비용 replay와 적합
  analyze.py   표·그림
```
