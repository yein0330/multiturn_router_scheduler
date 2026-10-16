# followup_v3 — 서빙 비용이 결정을 바꾸는 영역에서 stay/switch 제어 (단계 0–1)

- 상태: **단계 0–1 완료, 관문 1 미달로 단계 2 진입하지 않음.**
- 결과: [report_stage01.md](report_stage01.md)
- 다음 결정: [next_decision.md](next_decision.md)
- 기존 문서 수정 제안: [errata.md](errata.md)
- 설계·동결 기록: [design.md](design.md)

## 구조

| 경로 | 내용 |
|---|---|
| `configs/` | `rho_mapping.json`(동결 ρ 매핑), `paired_points.json`(측정 전 선정), `models_R_offline_scoring.json`(오프라인 점수 전용 R 설정) |
| `data/manifest_*.json` | smoke/dev/train/eval의 seed·sha256·손잡이. jsonl은 Git 제외, `src/make_v3_data.py`로 재생성 |
| `src/agent_tasks.py` | 도구 환경, 생성기, 참조 해법, 채점 |
| `src/agent_runner.py` | live 실행기(always_A / always_B / L0) |
| `src/profile_v3.py` | 프로파일 격자 측정, C0–C3 적합 |
| `src/gate1.py` | 결정 지점 분석, paired 측정, 관문 판정 |
| `src/stage0_*.py`, `src/stage1_rho.py` | 단계 0 분석, 라우터 점수 |
| `src/validate_tasks.py` | 생성기 검증(참조 = 직접 계산, 무작위 정책) |
| `results/stage0/` | 사후 선택 여지, 신호 탐색, 라우터 margin |
| `results/stage01/` | 감사, 프로파일, 비용 모델, 결정 불일치, paired, 그림 |
| `results/runs/{smoke,dev50}/` | live 원로그(호출별 calls.jsonl, 세션별 sessions.jsonl) |
| `results/locked_runs.log` | 배타 잠금 실행 기록 |
| `examples/` | 세션 렌더링 예시 |

## 재현

모든 GPU 작업은 `run_locked.sh`(flock, 지정한 서버만 기동·종료)로 실행한다. 명령은 [report_stage01.md §6](report_stage01.md#6-재현)에 있다.
