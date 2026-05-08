# Pseudo Panoptic Evaluation — V2

> **Claim:** Zero-shot 3D panoptic reconstruction 의 SQ ceiling (~84) 은 모델 구조 한계가 아니라 **SAM 기반 pseudo supervision 자체의 한계**다. Single-view raw pseudo SQ 81.55 가 PanopticRecon++ Final SQ 84.05 와 ~3% 차이로 일치한다.

## TL;DR

3-scene mean (mainV2 run, 200 frame/scene, PR++ compatible IoU > 0.3):

| 비교 | PQ | **SQ** | RQ |
| --- | --- | --- | --- |
| Raw pseudo (per-view, 본 실험) | 53.50 | **81.55** | 65.11 |
| PanopticRecon++ Final (3D pipeline) | 77.28 | **84.05** | 91.09 |
| PanopticSplatting Final (3D pipeline) | 77.73 | 82.70 | 93.60 |

> Single-frame pseudo mask 의 shape quality 가 prior 3D pipeline 의 final SQ 를 직접 결정한다. 3D pipeline 의 주된 contribution 은 SQ 가 아니라 RQ (cross-view ID consistency 회복).

## Documentation

V2 는 `docsV2/` 에 audience 별로 3개 파일로 분리되어 있음. V1 (`docs/`) 는 reference 로 보존.

| 질문 | 파일 |
| --- | --- |
| 어떤 실험을 진행했고, 무엇을 발견했나, 표는 어떻게 읽는지 | `docsV2/research_note.md` |
|threshold / config / output layout / per-scene taxonomy | `docsV2/spec.md` |
| 재현 순서, 코드 위치, 발생한 이슈들 (V1→V2 변경 포함) | `docsV2/implementation.md` |

## Quick Start

Docker 안에서:

```bash
docker exec -it pseudo_panoptic_eval_dev_cu118 bash
cd /workspace/repos/pseudo-panoptic-eval

# 1. eval 코드 sanity (eval-side 코드 변경 후 항상 먼저)
python3 scripts/checks/test_metrics_synthetic.py

# 2. Small test (3 scene × 5 frame, ~15 min)
python3 scripts/run_experiment.py --config configs/experiments/debug_5.yaml

# 3. V2 main run (3 scene × 200 frame, 실측 25.1분)
#    - PR++ compatible / per-view 만 평가
#    - per-view oracle (raw / +GT label / +GT shape) 자동 포함
#    - Grounded-SAM step 이 V2 fix 로 14× 가속
python3 scripts/run_experiment.py --config configs/experiments/mainV2.yaml

# 개별 step 만 다시 돌리고 싶을 때:
#   --resume          : state.json 보존, 미완료 step 부터 이어서
#   --start-at <id>   : 특정 step 부터 강제 시작
```

## V2 핵심 결과 (mainV2 run, 3-scene mean, per-view, PR++ compatible)

| Variant | Mask shape | Label | PQ | **SQ** | RQ |
| --- | --- | --- | --- | --- | --- |
| `raw` | pseudo | pseudo | 53.50 | 81.55 | 65.11 |
| `+GT label` | pseudo | **GT** | 59.31 | 82.01 | 72.12 |
| `+GT shape` | **GT** | pseudo | 82.15 | **99.17** | 82.40 |

→ Label 만 GT 로 교정해도 SQ 변화 미미 (+0.46pt). Shape 만 GT 로 바꾸면 SQ 99.17 (≈ 100). **Shape 가 SQ 의 유일한 결정자**.

**Source**: `outputs/pseudo-panoptic-eval/mainV2/metrics/{scene}_per-view{,-oracle}_prpp_compatible.csv` × 3 scene mean.

## V1 vs V2

V2 는 V1 (`docs/`) 의 narrative 를 단순화한 것 — 실험 결과 자체는 V1 main run 출력을 그대로 사용.

| | V1 | V2 |
| --- | --- | --- |
| Protocol | strict + PR++ compatible | PR++ compatible 만 |
| Eval scope | per-view + scene-level | per-view 만 |
| Oracle variants | 5 (`original`, `oracle_class`, `oracle_association`, `oracle_class_association`, `oracle_mask`) | 3 (`raw`, `gt_label`, `gt_shape`) |
| Hypothesis | 4 cases | A 단일 |

자세한 변경 사유 → `docsV2/implementation.md` §4.
