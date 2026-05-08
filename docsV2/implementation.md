# Implementation

> 본 실험을 어떻게 재현하는지, 코드가 어디에 있는지, 진행 중 어떤 이슈가 있었는지.

## 1. Reproduction

### 1.1 Docker 환경

```bash
docker exec -it pseudo_panoptic_eval_dev_cu118 bash
cd /workspace/repos/pseudo-panoptic-eval
```

CUDA 11.8 + PyTorch + GroundingDINO + SAM + nvdiffrast (via ScanNet++ toolkit) 가 미리 설치되어 있음.

### 1.2 V2 main run

```bash
# 1. eval 코드 sanity (eval-side 코드 변경 후 항상 먼저)
python3 scripts/checks/test_metrics_synthetic.py

# 2. Small test (3 scene × 5 frame, ~15 min, V1 호환)
python3 scripts/run_experiment.py --config configs/experiments/debug_5.yaml

# 3. V2 main run (3 scene × 200 frame, 실측 25.1분 wall-clock)
#    - per-view + PR++ compatible 만 평가
#    - per-view oracle (raw / +GT label / +GT shape) 자동 포함
#    - Grounded-SAM step 이 V2 fix 로 14x 가속 (§3.4)
python3 scripts/run_experiment.py --config configs/experiments/mainV2.yaml
```

각 step 이 끝나면 `outputs/pseudo-panoptic-eval/mainV2/state.json` 에 status 기록. `--resume` 으로 중단 지점부터 재개, `--start-at <step_id>` 로 특정 step 부터 강제 시작.

**실측 step 별 timing (mainV2 run, 600 frame total):**

| Step | 시간 | 비고 |
| --- | --- | --- |
| `check_models` | 9.9 sec | 체크포인트 sanity |
| `build_manifest` | 0.1 sec | manifest CSV 생성 |
| `render_gt_*` × 3 | 27-30 sec / scene | nvdiffrast batch render |
| `summarize_gt` | 13.4 sec | GT 품질 report |
| **`run_grounded_sam`** | **852 sec (14.2 min)** | **V2 fix: 모델 1회 load + frame 600개 in-process 순회. V1 추정 ~5-7h 대비 ~17-25× 가속** |
| `build_pseudo_panoptic` | 83 sec | per-pixel highest-score wins, 600 frame |
| `evaluate_per_view_*` × 3 | 30-34 sec / scene | per-frame eval, 200 frame/scene |
| `oracle_per_view_*` × 3 | 114-134 sec / scene | 3 variants × 200 frame eval |
| **Total** | **25.1 min** | |

### 1.3 V2 vs V1 yaml 차이

| | V1 (`configs/experiments/main.yaml`) | V2 (`configs/experiments/mainV2.yaml`) |
| --- | --- | --- |
| `metric_profiles` variable | `[prpp_compatible, strict]` | 제거 (hardcoded `prpp_compatible`) |
| `metric_modes` variable | `[per-view, scene-level]` | 제거 (hardcoded `per-view`) |
| `evaluate_*` step | 3 scene × 2 mode × 2 profile = **12 step** | 3 scene × 1 mode × 1 profile = **3 step** |
| Oracle script | `evaluate_oracle.py` (scene-level concat) | **`evaluate_oracle_per_view.py`** (frame 별 독립) |
| Oracle output 파일명 | `{scene}_oracle_{profile}.csv` | `{scene}_per-view-oracle_{profile}.csv` |
| Total expanded steps | 26 | **14** |
| Output root | `/workspace/outputs/pseudo-panoptic-eval/main` | `/workspace/outputs/pseudo-panoptic-eval/mainV2` |

V1 main.yaml 은 V1 호환용으로 보존되어 있어 — V1 outputs 를 재현하거나 strict / scene-level 결과가 필요하면 그대로 사용 가능.

### 1.4 Manual per-view oracle 호출 (이미 main run 결과 있을 때)

mainV2.yaml 의 `oracle_per_view_*` step 만 떼어내서 수동 실행:

```bash
for sc in 1ada7a0617 5748ce6f01 f6659a3107; do
  python3 scripts/pipeline/evaluate_oracle_per_view.py \
    --pseudo-root /workspace/outputs/pseudo-panoptic-eval/mainV2/pseudo_panoptic \
    --gt-root    /workspace/outputs/pseudo-panoptic-eval/mainV2/gt_panoptic \
    --scene $sc --metric-profile prpp_compatible \
    --out-csv    /workspace/outputs/pseudo-panoptic-eval/mainV2/metrics/${sc}_per-view-oracle_prpp_compatible.csv
done
```

3 scene 합계 ~5-7분 (CPU only). 결과 CSV 의 행: `200 frame × 3 variants = 600` per scene.

## 2. Code 위치

### 2.1 Pipeline scripts (`scripts/pipeline/`)

| Step | 파일 | 역할 |
| --- | --- | --- |
| 1 | `build_manifest.py` | Frame 선택, manifest CSV 생성 (frame_id, rgb_path, intrinsic_path, ...) |
| 2 | `render_gt_panoptic.py` | nvdiffrast 로 GT panoptic 렌더, scene 1번에 batch |
| 3a | `run_grounded_sam_batch.py` | Scene 별 frame 순회 driver |
| 3b | `run_grounded_sam_one_image.py` | Frame 1개에 GroundingDINO + SAM 실행 |
| 4 | `build_pseudo_panoptic.py` | result.npz → semantic/instance/panoptic.npy 변환 |
| 5 | `evaluate_pseudo_gt.py` | Per-view 또는 scene-level panoptic eval |
| 6 (V2) | **`evaluate_oracle_per_view.py`** | **V2: per-view 3 variants (raw/gt_label/gt_shape)** |
| 6 (legacy) | `evaluate_oracle.py` | Scene-level 5 variants — V2 narrative 에서는 사용 안 함 |
| 7 | `summarize_gt_render.py` | GT render quality report |

Runner: `scripts/run_experiment.py` — YAML config 의 step 들을 순차 실행, state/resume 관리.

### 2.2 Source modules (`src/`)

| Module | 핵심 함수 | 역할 |
| --- | --- | --- |
| `src/eval/panoptic.py` | `evaluate_panoptic_arrays` (lines 31-121), `make_panoptic` (6-14), `segment_info` (17-28) | PQ/SQ/RQ 산식 — 본 실험의 metric 핵심 |
| `src/eval/oracle.py` | `apply_oracle_class` (88-96), `apply_oracle_mask` (136-152), `best_gt_for_pred` (40-61), `best_pred_for_gt` (64-85) | Variant 생성 (label / shape 교정) |
| `src/eval/instance.py` | `evaluate_instance_arrays` | mCov / mWCov / mPrec / mRec |
| `src/pseudo/build_maps.py` | `build_pseudo_maps` (9-87) | Per-pixel highest-score-wins overlap 해결 + panoptic ID 합성 |
| `src/pseudo/config.py` | `get_label_info`, `get_prompt_to_label` | Class config loader |
| `src/gt/render_nvdiff.py` | `render_panoptic_batch` (47-86) | nvdiffrast wrapper, vertex-to-pixel propagation |
| `src/gt/scannetpp.py` | `load_vertex_annotations` (28-61), `load_scene_class_map` (14-19) | ScanNet++ 3D annotation → per-vertex (sem, inst) |
| `src/gt/render.py` | `save_gt_maps` (17-24) | GT semantic/instance/panoptic.npy 저장 |
| `src/gt/colmap.py` | `read_cameras`, `read_images` | COLMAP intrinsic / extrinsic loader |
| `src/utils/undistort.py` | `compute_undistort_intrinsic`, `get_undistort_maps`, `undistort_image` | cv2.fisheye 기반 undistortion |

### 2.3 핵심 로직 한 줄 요약

- **Pseudo panoptic**: `build_maps.py:54-69` per-pixel `score_map > logits_map` 으로 highest-grounding-score 의 phrase 가 그 픽셀의 label 결정.
- **GT panoptic**: `render_nvdiff.py:37` `vtx_prop[mesh_faces_np[pix_to_face[valid]][:, 0]]` — pix_to_face 에서 첫 vertex 의 label propagate.
- **Panoptic ID**: 양쪽 모두 `sem * OFFSET + inst` (`build_maps.py:86`, `render.py:20`).
- **Eval**: `panoptic.py:108-111` — `SQ = sum_iou/TP`, `RQ = TP/(TP + 0.5·FP + 0.5·FN)`.

## 3. 발생한 문제 + 해결

### 3.1 Snowstorm GT (legacy vertex-splat → nvdiffrast)

**문제**: 초기 구현은 mesh vertex 만 image plane 에 splat 하는 방식. 픽셀이 vertex 사이에 떨어지면 unlabeled → valid_ratio 가 worst-case 0.397 까지 떨어지면서 GT 가 sparse "snowstorm" 형태가 됨. 작은 vertex 영역 vs 넓은 pseudo mask 의 IoU 가 인위적으로 낮아져서 SQ 가 실제보다 낮게 측정됨.

**해결**: nvdiffrast triangle rasterizer 로 교체. mesh face 를 GPU 상에서 rasterize 한 후 face 의 첫 vertex label 을 그 픽셀에 propagate. worst-case valid_ratio 0.979 (sanity smoke test).

**위치**: `src/gt/render_nvdiff.py:73-86`. legacy `src/gt/render.py:render_vertex_maps` 는 제거됨.

### 3.2 Undistort 좌표계 정렬

**문제**: ScanNet++ 의 DSLR 카메라는 fisheye distortion 이 큼. Pseudo 는 distorted RGB 로 inference 하면 GT mesh rendering (undistorted intrinsic 으로 계산) 과 픽셀 정렬이 안 맞음. 결과로 같은 위치에서 sem/inst label 이 어긋난 채 IoU 계산되어 SQ 가 오염.

**해결**: 양쪽 모두 **undistorted DSLR 좌표계** 에서 처리.
- `run_grounded_sam_one_image.py:174-204`: GroundingDINO 입력 RGB 를 cv2.fisheye 로 undistort 후 inference.
- `render_gt_panoptic.py:88-90`: undistorted intrinsic 을 nvdiffrast 에 전달.
- 결과: 두 panoptic map 의 (H, W) 와 좌표계가 정확히 일치 → 픽셀별 비교 가능.

**위치**: `src/utils/undistort.py` (3 helper 함수), `src/gt/render_nvdiff.py:8-10` ("Both inputs and outputs live in undistorted camera space").

### 3.3 nvdiffrast install (CUDA 11.8 호환)

**문제**: `pip install nvdiffrast` 는 build 의존성 (CUDA toolkit, ninja, etc.) 이 까다로워서 일반 CUDA 11.8 image 에서 직접 install 하기 어려움.

**해결**: ScanNet++ official toolkit 의 rasterize 모듈을 직접 import. 이 toolkit 은 nvdiffrast 가 미리 빌드되어 있음.

**위치**: `src/gt/render_nvdiff.py:67`:
```python
from common.utils.rasterize import rasterize_mesh_nvdiffrast_large_batch
```
toolkit path 는 `--toolkit-root /workspace/third-party/scannetpp` 로 prepend.

### 3.4 [성능 이슈, V2 fix 완료] Grounded-SAM 모델 redundant reload

**문제 (V1)**: `run_grounded_sam_batch.py` 가 frame 마다 `subprocess.run()` 으로 새 Python process 를 spawn → `run_grounded_sam_one_image.py` 가 매번 모델 reload:
- GroundingDINO checkpoint (~700 MB) → GPU
- SAM ViT-H checkpoint (~2.4 GB) → GPU
- 합 ~3.1 GB × 600 frame = **1.86 TB** 의 redundant disk → GPU 전송

**영향 (V1)**: Step 3 가 5-7 시간. 실제 inference 는 frame 당 2-3 초인데, 매번 모델 로드에 30-40 초가 추가됨.

**V2 fix**: 두 파일 리팩터링.
- `run_grounded_sam_one_image.py` 에 `load_models()` + `process_image()` 함수 분리. CLI 진입점은 그대로 유지 (단일 frame debug 용).
- `run_grounded_sam_batch.py` 가 `load_models()` 를 **1번** 호출 후, frame 들을 in-process 순회하며 `process_image()` 직접 호출. subprocess 사용 안 함.
- COLMAP 카메라 데이터와 undistort 맵을 module-level dict (`_COLMAP_CACHE`, `_UNDISTORT_CACHE`) 에 캐시 — scene 당 1회 I/O.

**검증**:
- 5 frame smoke test: V2 = **15초** (모델 1회 load 9초 + frame 당 ~1초)
- 600 frame mainV2 production run: **852 sec = 14.2 min** (= 1.42 sec / frame 평균)
- Output bit-identical: V2 의 result.npz 가 V1 과 동일 mask 갯수 / phrase / grounding score (smoke test 검증)

| 구성 | per-frame | 600 frame total |
| --- | --- | --- |
| V1 (subprocess + 매번 모델 reload) | ~30-40 sec | 추정 ~5-7 시간 |
| **V2 (모델 1번 load + in-process 순회)** | **~1.4 sec** | **실측 14.2 분** (run_grounded_sam step) |
| 가속 | ~25× | ~21-30× |

**위치**:
- `scripts/pipeline/run_grounded_sam_one_image.py:121-145` — `load_models()`
- `scripts/pipeline/run_grounded_sam_one_image.py:152-204` — `_prepare_image()` (캐시된 undistort)
- `scripts/pipeline/run_grounded_sam_one_image.py:207-322` — `process_image()` (재사용 가능한 frame 처리 함수)
- `scripts/pipeline/run_grounded_sam_batch.py:88-124` — load 1회 + frame 순회 driver

**호환성**: 기존 CLI 인자 (`--manifest-dir`, `--scene-list`, `--profile`, `--class-profile`, `--undistort`, `--skip-existing`, `--skip-overlay`, `--limit`) 모두 그대로. `configs/experiments/{main,debug_5}.yaml` 의 step 3 명령어 변경 없이 자동으로 V2 가속 적용.

### 3.5 [V2 데이터 추가] Per-view oracle 분해 누락

**문제**: 기존 `evaluate_oracle.py` 는 scene-level concat 후 5 variants 를 평가. V2 narrative 에서 필요한 **per-view label / shape 분해** 가 없음.

**해결**: `scripts/pipeline/evaluate_oracle_per_view.py` 신규 작성. Frame 별 독립 평가, 3 variants (`raw`, `gt_label`, `gt_shape`).

**구조**:
```python
for pred_sem_path in iter_frame_dirs(...):
    pred = load_frame_maps(...)
    gt   = load_frame_maps(...)
    best_gt   = best_gt_for_pred(...)     # ← per-frame best matching
    best_pred = best_pred_for_gt(...)
    variants = {
        "raw":      (pred["semantic"], pred["instance"]),
        "gt_label": apply_oracle_class(..., best=best_gt),
        "gt_shape": apply_oracle_mask(...,  best=best_pred),
    }
    for variant, (sem, inst) in variants.items():
        row = evaluate_panoptic_arrays(sem, inst, gt["semantic"], gt["instance"], ...)
        rows.append(row)
```

**위치**: `scripts/pipeline/evaluate_oracle_per_view.py` (신규). 출력 CSV 는 `metrics/{scene}_per-view-oracle_prpp_compatible.csv`.

**실행 시간**: 3 scene × 200 frame × 3 variants ≈ 5-7분 (CPU only, frame 당 < 1 초).

## 4. V1 → V2 변경 요약

| 영역 | V1 | V2 | 이유 |
| --- | --- | --- | --- |
| Protocol | strict + PR++ compatible 둘 다 | **PR++ compatible 만** | 분석 양 ↓, 발표 narrative 단일화 |
| Eval scope | per-view + scene-level | **per-view 만** | "pseudo 한계" 한 가지에 집중. cross-view ID 분석은 future work |
| Oracle variants | 5 variants (`original`, `oracle_class`, `oracle_association`, `oracle_class_association`, `oracle_mask`) | **3 variants (`raw`, `gt_label`, `gt_shape`)** | 이름 직접화, scene-level 전용 (`*association`) 제거 |
| Hypothesis | 4 cases (A/B/C/D) | **A 하나** | 핵심 결론 분산 방지 |
| 신규 script | — | `evaluate_oracle_per_view.py` | per-view variant 분해 데이터 생성 |
| 신규 yaml | — | `configs/experiments/mainV2.yaml` | V2 narrative 와 정렬된 14-step pipeline |
| Step 3 Grounded-SAM | subprocess per frame (~5-7h 추정) | 모델 1회 load + in-process 순회 (실측 14.2분) | ~21-30× 가속 (§3.4) |
| Total main run | ~7시간 (V1 추정) | **25.1분 (mainV2 실측)** | ~17× 단축 |

**영향 범위**:
- V1 코드 (`evaluate_oracle.py`, `configs/experiments/main.yaml`) 는 보존 → 기존 V1 결과 재현 가능
- V2 신규 항목 (per-view oracle script, mainV2.yaml) 만 추가
- Step 3 fix 는 V1/V2 양쪽 호출 모두에 자동 적용 (CLI 인자 호환 유지)
- 실험 결과 수치는 V1 의 main run output 에서 directly reuse — narrative 만 V2 로 단순화
