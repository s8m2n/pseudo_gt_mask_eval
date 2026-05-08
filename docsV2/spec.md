# Specification

> Machine-readable values for reproduction. Code reads these from config files (single source of truth) — values listed here are for human inspection.

## 1. Thresholds

| Param | Value | Config 위치 | 의미 |
| --- | --- | --- | --- |
| Panoptic IoU threshold | **0.3** | `configs/grounded_sam.yaml` → `evaluation.panoptic_iou_thresholds.prpp_compatible` | TP gate (PR++ compatible) |
| Confidence threshold | 0.25 | `evaluation.profiles.prpp_compatible.mask_filter.confidence_threshold` | grounding score 하한 |
| Min mask area | 1 | `evaluation.profiles.prpp_compatible.mask_filter.min_mask_area` | 1 px 미만 mask 제거 |
| GroundingDINO box threshold | 0.25 | `evaluation.profiles.prpp_compatible.groundingdino.box_threshold` | box prediction confidence |
| GroundingDINO text threshold | 0.25 | `evaluation.profiles.prpp_compatible.groundingdino.text_threshold` | text-region matching confidence |
| GroundingDINO remove_combined | true | `evaluation.profiles.prpp_compatible.groundingdino.remove_combined` | phrase 단위 dedupe |
| skip_large_box_ratio | 0.7 | `evaluation.profiles.prpp_compatible.mask_filter.skip_large_box_ratio` | 화면 70% 이상 box 사전 제거 |
| ignore pred-void overlap | 0.5 | `evaluation.ignore_pred_void_overlap_gt_fraction` | pred 가 void 영역 50% 이상 덮으면 FP 카운트 안 함 |
| subtract void from union | true (PR++) | `evaluation.subtract_void_from_union.prpp_compatible` | union 에서 void overlap 빼서 IoU 살짝 ↑ |
| OFFSET | 1000 | `evaluation.offset` | panoptic_id = sem × OFFSET + inst |
| void_id | 0 | `evaluation.void_id` | unlabeled / out-of-render |

## 2. Config 파일

| 파일 | 내용 |
| --- | --- |
| `configs/classes_scannetpp.yaml` | Scene 별 prompt list (17-33개), prompt_to_label, gt_label_to_label (GT alias mapping), labels (thing/stuff/evaluated meta) |
| `configs/grounded_sam.yaml` | Threshold 들 (위 표), evaluation 파라미터 |
| `configs/models.yaml` | GroundingDINO checkpoint path (`swint_ogc`), SAM checkpoint path (`vit_h`) |
| `configs/scenes_main.txt` | Main run 의 3 scene list (`1ada7a0617`, `5748ce6f01`, `f6659a3107`) |
| `configs/scenes_sanity.txt` | Sanity run 의 scene list |
| `configs/experiments/main.yaml` | 200 frame/scene main run pipeline |
| `configs/experiments/debug_5.yaml` | 5 frame/scene debug run |

## 3. Output layout

V2 main run 은 `outputs/pseudo-panoptic-eval/mainV2/` (Docker 안: `/workspace/outputs/pseudo-panoptic-eval/mainV2/`) 에 저장. `configs/experiments/mainV2.yaml` 의 `output_root` 변수에서 정의.

```
mainV2/
├── manifests/{scene}.csv                          # frame 선택 결과 (frame_id, rgb_path, intrinsic_path, ...)
├── pseudo_raw/{scene}/{frame}/                    # Step 3 (Grounded-SAM raw output)
│   ├── result.npz                                 # masks, boxes, scores, phrases
│   ├── input_undist.png                           # undistorted RGB (GroundingDINO 입력)
│   └── metadata.json
├── pseudo_panoptic/{scene}/{frame}/               # Step 4 (build_pseudo_panoptic)
│   ├── semantic.npy                               # (H, W) int32
│   ├── instance.npy                               # (H, W) int32
│   ├── panoptic.npy                               # (H, W) int64 = sem × 1000 + inst
│   └── segments.json                              # per-mask metadata (class, score, area)
├── gt_panoptic/{scene}/{frame}/                   # Step 2 (render_gt_panoptic)
│   ├── semantic.npy                               # (H, W) int32
│   ├── instance.npy                               # (H, W) int32
│   ├── panoptic.npy                               # (H, W) int64
│   ├── valid.npy                                  # (H, W) bool — eval 영역
│   ├── semantic_preview.png                       # palette 시각화
│   └── metadata.json                              # renderer="nvdiffrast_undistorted", intrinsic, valid_ratio
├── metrics/                                       # Step 5/6 (V2: PR++ compatible 만)
│   ├── {scene}_per-view_prpp_compatible.csv       # raw per-view (1 row / frame, 200 row × 3 scene)
│   ├── {scene}_per-view-oracle_prpp_compatible.csv  # V2 oracle: 3 variants × 200 frame = 600 row × 3 scene
│   └── gt_render_summary.csv                      # GT render quality report
├── run_report.md                                  # runner 진행 상황 + step status
└── state.json                                     # step 별 status (resume 용)
```

V1 run output (`outputs/pseudo-panoptic-eval/main/`) 는 별도 트리에 보존 — strict + scene-level 결과 포함. V2 narrative 에서는 사용 안 함.

## 4. Models

| Model | Variant | Checkpoint path (Docker 안) |
| --- | --- | --- |
| GroundingDINO | swint_ogc | `/workspace/models/groundingdino_swint_ogc.pth` |
| SAM | vit_h | `/workspace/models/sam_vit_h_4b8939.pth` |

## 5. Per-scene class taxonomy

각 scene 의 prompt list / label 매핑 / thing-stuff split 은 `configs/classes_scannetpp.yaml` 의 `profiles.prpp_compatible.scenes.{scene}.{prompts, prompt_to_label, gt_label_to_label, labels}` 에 보존.

- Scene 별 prompt 갯수: 17 (`1ada7a0617`) / **33** (`5748ce6f01`, `office chair` 1개 추가) / 23 (`f6659a3107`).
- Thing classes: chair, table, sofa, bed, cabinet, door, trash can, monitor, keyboard, etc.
- Stuff classes: wall, floor, ceiling, curtain.
- Evaluated flag: 일부 class 는 `evaluated: 0` 으로 평가에서 제외 (rare class).

## 6. Frame selection

- Main run: scene 당 200 frame, manifest 의 `valid_ratio_descending` order (가장 잘 렌더된 frame 우선) → `build_manifest.py` 결정성 보장.
- Sanity / debug: 5 또는 100 frame.
- 좌표계: 모든 frame 이 undistorted DSLR (cv2.fisheye.undistortImage) 으로 사전 처리됨.
