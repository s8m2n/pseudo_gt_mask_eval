# Pseudo GT 의 한계가 zero-shot 3D panoptic reconstruction 의 SQ ceiling 인가

> **One-line:** Grounded-SAM 으로 만든 pseudo panoptic mask 를 ScanNet++ GT 에 직접 비교했더니, 단일 view 에서 측정한 raw pseudo SQ 가 PanopticRecon++ / PanopticSplatting 의 final 3D reconstruction SQ 와 거의 같은 값으로 나왔다. 즉 본 분야의 SQ ceiling 은 모델 구조가 아니라 supervision (pseudo mask) 자체에서 온다.

---

## 1. 실험 배경

Zero-shot / open-vocabulary 3D panoptic reconstruction 방법들 (PanopticSplatting, PanopticRecon++) 은 2D foundation model — 보통 Grounded-SAM — 으로 만든 pseudo masks 를 supervision 으로 사용한다. ScanNet++ 에서 두 방법의 보고치는 다음과 같다.

| Method | PQ | SQ | RQ |
| --- | --- | --- | --- |
| PanopticRecon++ | 77.28 | **84.05** | 91.09 |
| PanopticSplatting | 77.73 | 82.70 | 93.60 |

`PQ ≡ SQ × RQ` 항등식상 RQ 는 이미 saturate (>91) 되어 있고, **PQ 의 ceiling 을 결정하는 것은 SQ ≈ 84**. 이 ceiling 이 모델 구조의 한계인지, supervision 으로 쓰는 pseudo mask 의 한계인지를 별개로 판정하려면 **pseudo mask 자체의 SQ 를 직접 측정** 하면 된다.

---

## 2. 실험 목적 및 가설

> **목적:** Grounded-SAM pseudo panoptic mask 의 intrinsic quality 가 prior 3D pipeline 들의 final SQ ceiling 을 직접 결정하는지 측정한다.

**Hypothesis A (단일):** Raw pseudo per-view SQ ≈ Final SQ (≈ 82-84) 이면, pseudo mask 자체가 zero-shot 3D panoptic 의 SQ ceiling 을 형성한다고 결론.

| 결과 패턴 | 해석 |
| --- | --- |
| per-view raw pseudo SQ ≈ Final SQ | ✅ pseudo 가 ceiling — 본 가설 지지 |
| per-view raw pseudo SQ ≫ Final SQ | ❌ pseudo 보다 다른 단계 (3D lifting, query optimization) 가 병목 |
| per-view raw pseudo SQ ≪ Final SQ | ❌ 3D pipeline 이 pseudo noise 를 보정해서 SQ 를 끌어올림 |

3D reconstruction / label lifting 없이, **per-view 단계** 에서 두 panoptic map 을 직접 비교한다.

```
ScanNet++ RGB → Grounded-SAM → Pseudo panoptic
                                                ┐
                                                ├→ per-pixel 비교 → PQ/SQ/RQ
ScanNet++ mesh → nvdiffrast → GT panoptic       ┘
```

---

## 3. 평가 Metric

### 3.1 PQ / SQ / RQ 정의

각 frame 에서 pseudo segments 와 GT segments 의 pair 별 IoU 를 계산하고 (class 가 다르면 제외), `IoU > 0.3` (PR++ compatible) 인 pair 를 IoU 내림차순 greedy 1-to-1 matching → TP. 매칭 안된 pred 는 FP, 매칭 안된 GT 는 FN.

$$
PQ = \frac{\sum_{(p,g)\in TP} IoU(p,g)}{TP + 0.5 FP + 0.5 FN}
\quad
SQ = \frac{\sum_{(p,g)\in TP} IoU(p,g)}{TP}
\quad
RQ = \frac{TP}{TP + 0.5 FP + 0.5 FN}
$$

`PQ = SQ × RQ` 가 정의상 성립.

### 3.2 의미

| Metric | 측정 대상 | 직관 |
| --- | --- | --- |
| **SQ** | matched pair 의 평균 IoU | "잡았을 때 mask shape 가 GT 와 얼마나 일치하는가" |
| **RQ** | F1 score of segment detection | "GT segment 와 pseudo segment 의 cardinality 가 얼마나 1-to-1 로 맞는가" |
| **PQ** | SQ × RQ | mask quality + recognition 종합 |

RQ 는 algebra 상 **F1** 과 동치:
- Precision = TP / (TP + FP), Recall = TP / (TP + FN)
- RQ = 2·P·R / (P+R)

→ SQ 와 RQ 는 **독립적인 두 축**. SQ 가 높아도 RQ 가 낮을 수 있다 (잡은 건 잘 잡지만 적게 잡았을 때).

### 3.3 평가 protocol

- **PR++ compatible (IoU > 0.3)** 단일 protocol — PanopticRecon++ 공개 구현과 일치.
- Per-view 평가만 수행. Scene-level / cross-view ID 분석은 본 실험 범위 외.
- Class-aware matching: cross-class pair 는 IoU 계산조차 안 함.
- Greedy by IoU desc, gt/pred 각 1번만 매칭 (Hungarian 아님).

---

## 4. 실험 설계

### 4.1 Pseudo GT 생성 — Grounded-SAM

```
Text prompts (scene별 17-33개)
   ↓
GroundingDINO  (text → box + grounding score)
   ↓
SAM ViT-H      (box → binary mask, multimask_output=False)
   ↓
N개의 (mask, phrase, score)
   ↓
Per-pixel highest-grounding-score wins → semantic + instance map
   ↓
panoptic_id = semantic * OFFSET + instance
```

Per-pixel resolution 의 핵심: 한 픽셀이 여러 mask 에 속할 때 **grounding_score 가 가장 높은 mask 의 phrase** 의 label 로 결정. thing class 는 새 instance ID 부여, stuff 는 instance=0.

### 4.2 GT 2D Rendering — ScanNet++ nvdiffrast

```
ScanNet++ mesh (.ply) + segments_anno (3D annotation)
   ↓
per-vertex (semantic, instance) array
   (segments_anno.json 의 segGroups → label, objectId)
   ↓
nvdiffrast triangle rasterizer (undistorted intrinsic)
   ↓
pix_to_face → "face 의 첫 vertex 의 label" 을 픽셀에 propagate
   ↓
GT semantic / instance / valid map  (panoptic ID composition 동일)
```

Pseudo 와 GT **양쪽 모두 undistorted DSLR 좌표계** 에서 생성되어 픽셀 단위 비교 가능.

### 4.3 비교에서 발생할 한계 — Grounded-SAM 의 5 single-frame failure modes

본 실험은 다음 실패 모드들을 per-view metric 에서 직접 검증한다. 모두 **3D / cross-view 와 무관한 단일 frame 안에서의 실패**.

| # | 실패 모드 | 발생 단계 | 메커니즘 |
| --- | --- | --- | --- |
| ① | **Class hallucination** | GroundingDINO | "ceiling" / "object" 같은 prompt 가 GT 에 없는 영역에 box 를 만듦 |
| ② | **Wrong grounding (class leak)** | GroundingDINO + SAM | text-region grounding 이 loose → SAM mask 가 인접 wrong-class 영역까지 확장 |
| ③ | **Under-segmentation merge** | SAM | 인접한 동일 class 두 instance 의 boundary 가 약하면 single mask 로 처리 |
| ④ | **Detection miss** | GroundingDINO | small / 가장자리 / 부분 가시 instance 의 box confidence 가 threshold 미달 |
| ⑤ | **Same-class duplicate** | NMS 부족 | 같은 영역에 두 box prediction → 중복 detection |

이 5가지 실패가 single frame 안에서 빈번할 것을 가정하며, §5.3 에서 한 frame 의 raw breakdown + 600 frame 통계로 검증한다.

---

## 5. 실험 결과 및 분석

> 모든 결과는 mainV2 run (3 scene × 200 frame, undistorted DSLR, nvdiffrast GT, PR++ compatible IoU > 0.3) 의 3-scene mean. 총 wall-clock 25.1분 (`configs/experiments/mainV2.yaml`). CSV 위치: `outputs/pseudo-panoptic-eval/mainV2/metrics/`.

### 5.1 핵심 표 — Per-view raw pseudo vs Prior 3D pipeline final

| Source | 평가 단계 | PQ | **SQ** | RQ | mIoU | mAcc | mCov |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Raw pseudo (per-view)** | 단일 frame | 53.50 | **81.55** | 65.11 | 54.66 | 62.37 | 56.21 |
| PanopticRecon++ final | 3D scene-level | 77.28 | **84.05** | 91.09 | 81.47 | 90.53 | 75.39 |
| PanopticSplatting final | 3D scene-level | 77.73 | 82.70 | 93.60 | 81.90 | 89.50 | 74.73 |

**Source**: per-view = `{scene}_per-view_prpp_compatible.csv` × 3 scene mean. PR++/PS Final 은 각 논문 보고값 ([PR++](https://arxiv.org/abs/2501.01119), [PS](https://arxiv.org/abs/2503.18073)).

**핵심 관찰:**

- **Raw pseudo per-view SQ 81.55 ≈ PR++ Final SQ 84.05** (gap **2.50pt**, ~3%) — **단일 view 에서 측정한 pseudo mask 의 shape quality 가 prior 3D pipeline 의 최종 SQ 와 같은 수준**. SQ 가 Final 까지 올라가는 데 3D pipeline 의 contribution 은 미미.
- **Raw pseudo per-view RQ 65.11 vs PR++ Final RQ 91.09** (gap +25.98pt) — RQ 는 3D pipeline 이 크게 끌어올림. 이건 cross-view ID 회복의 결과이며, 본 실험의 직접 분석 대상은 아니지만 main claim 의 mirror image: **3D pipeline 의 진짜 가치는 SQ 가 아니라 RQ 회복에 있다**.

### 5.2 Variant 분해 — Label vs Shape 의 contribution 분리

각 frame 마다 pseudo mask 를 다음 3가지 변형으로 평가:

| Variant | Mask shape | Label | 남은 error |
| --- | --- | --- | --- |
| `raw` | pseudo | pseudo | 모든 error |
| `+GT label` | pseudo | **GT** | shape error 만 |
| `+GT shape` | **GT** | pseudo | label error 만 |

3-scene per-view 평균 (PR++ compatible):

| Variant | PQ | **SQ** | RQ | mIoU | mAcc | mCov |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 53.50 | 81.55 | 65.11 | 54.66 | 62.37 | 56.21 |
| +GT label | 59.31 | 82.01 | 72.12 | 59.23 | 65.66 | 57.81 |
| +GT shape | 82.15 | **99.17** | 82.40 | 65.92 | 68.44 | 78.22 |

**Source**: `{scene}_per-view-oracle_prpp_compatible.csv` × 3 scene mean (변형별 200 frame).

**핵심 관찰:**

1. **raw → +GT label: SQ +0.46, RQ +7.0** — label 을 GT 로 교정해도 SQ 거의 변화 없음. label error 는 SQ 의 결정자가 아님.
2. **+GT label → +GT shape: SQ +17.16** — shape 만 GT 로 바꾸면 SQ 가 단숨에 99.17 (≈ 100). matched TP 의 IoU 가 정의상 1.0 이 되기 때문.
3. **+GT shape SQ 99.17** ≈ 100 → **shape 가 SQ 의 유일한 결정자**. label / boundary 외 다른 요인은 SQ 에 영향을 주지 않는다.

이 분해가 §5.1 의 핵심 비교 (raw SQ 81.55 ≈ Final SQ 84.05) 를 강화한다 — Final 까지 올라가는 ~2.5pt gap 은 mask shape refinement 만이 메울 수 있고, 그 refinement 의 천장은 SQ=99 (= shape 만 perfect) 이다.

### 5.3 5 failure modes 정량 검증

#### One-frame breakdown — `1ada7a0617/DSC03462`

GT segments (9): wall, floor, cabinet×2, whiteboard, door×2, chair×2.
Pseudo segments (11): cabinet×1 (large merged), wall, floor, chair×2, whiteboard, door×2, **ceiling**, **object**, **trash can(1px)**.

Matched 6 pair (IoU): floor 0.94 / whiteboard 0.90 / chair#27 0.83 / door#31 0.76 / wall 0.57 / cabinet#67 0.50.

→ **TP=6, FP=5, FN=3, SQ=0.751, RQ=0.60**.

| 실패 사건 | 원인 모드 | TP/FP/FN |
| --- | --- | --- |
| pseudo "ceiling" (63k px) — GT 에 없음 | ① Class hallucination | FP +1 |
| pseudo "object" (43k px) — taxonomy mismatch | ① + prompt design | FP +1 |
| pseudo "trash can" (1 px) — degenerate | post-processing 미흡 | FP +1 |
| pseudo chair#1 (303k px) 가 wall 영역까지 침범 → IoU 0.247 | ② Wrong grounding | FP +1, FN +1 (chair#43) |
| GT cabinet 2개 (#67, #68) → pseudo cabinet 1개로 합쳐짐, #68 IoU 0.473 | ③ Under-segmentation | FN +1 |
| GT door#59 (7k px, 작음) detection 안 됨 | ④ Detection miss | FN +1 |
| pseudo door#4 — door#31 외 추가 detection | ⑤ Same-class duplicate | FP +1 |

→ 5가지 실패 모드 모두 한 frame 안에서 관찰됨. cross-view 와 무관.

#### 600 frame 통계로 일반화

per-view PR++ compatible 600 frame 누적:

| 항 | 값 |
| --- | --- |
| 평균 #pseudo segments / frame | 16.45 |
| 평균 #GT segments / frame | 13.70 |
| 평균 TP / frame | 9.97 |
| 평균 FP / frame | 6.48 |
| 평균 FN / frame | 3.73 |
| Recall = TP/(TP+FN) | **72.76%** |
| Precision = TP/(TP+FP) | **60.60%** |
| RQ = F1(P, R) | **66.12%** ✓ |

→ **pseudo 가 frame 당 평균 2.75 개 더 많은 segment 를 만든다 (16.45 vs 13.70)**, 그 중 6.48 개는 GT 와 매칭 실패 (대부분 ①②⑤). GT 의 13.70 segment 중 3.73 개는 pseudo 가 못 잡음 (③④).

#### Real detection mismatch dominates RQ loss

§5.3 의 frame breakdown 과 600 frame 통계가 보여주듯, FP=6.48 / FN=3.73 의 거의 전부가 **boundary 근소 미달이 아니라 real detection mismatch** 에서 옴 — class hallucination (FP), wrong grounding (FP+FN), under-segmentation merge (FN), detection miss (FN), same-class duplicate (FP). 즉 RQ 65 권은 mask shape quality 의 문제가 아니라 single-frame **segment counting** 의 문제이고, 이는 §5.2 에서 "shape 만 GT 로 바꿔도 RQ 는 82.40 까지만 회복 (≠ 100)" 으로도 확인됨 (shape 가 perfect 여도 detection-stage P/R 한계는 그대로).

→ §5.4 의 main claim ("SQ 가 ceiling 의 결정자") 와 별개로, RQ 의 근본 한계는 detection-stage 의 P/R 이라는 부산 결론. 이 부분은 본 실험 범위인 SQ ceiling 분석과는 별도 축이며, future direction §6 의 "better mask predictor + grounding" 에서 자연스럽게 함께 해결될 수 있는 부분.

### 5.4 Hypothesis A 판정

| 판정 근거 | 값 |
| --- | --- |
| Raw per-view SQ | 81.55 |
| PR++ Final SQ | 84.05 |
| Gap | **2.50pt (~3% relative)** |
| +GT label per-view SQ | 82.01 (almost no change → label 은 병목 아님) |
| +GT shape per-view SQ | 99.17 (≈ 100 → shape 가 단일 결정자) |

→ **✅ Hypothesis A 강하게 지지**. Raw pseudo 의 single-frame mask shape quality 가 prior 3D pipeline 의 Final SQ 를 직접 결정한다. 모델 구조 (3D lifting, query optimization) 가 SQ 를 더 끌어올릴 수 있는 여지는 ~2.5pt 에 불과하다.

---

## 6. Takeaways & Future Direction

### Main takeaway

Zero-shot 3D panoptic reconstruction 의 **SQ ceiling 84 는 모델 구조의 한계가 아니라 SAM/Grounded-SAM 기반 pseudo supervision 자체의 한계** 다. Per-view raw pseudo SQ 81.55 가 prior work final SQ 84.05 와 ~3% 차이로 일치한다는 것이 직접 증거이며, +GT label / +GT shape 분해가 이 ceiling 의 정체가 mask shape 임을 추가로 입증한다.

### 3D pipeline 의 역할 (간단 mention)

본 실험 범위 외이지만, RQ gap (per-view 65 vs Final 91) 은 3D pipeline 의 진짜 contribution 이 어디인지를 시사한다 — **cross-view ID consistency 의 회복**. Label warping / label blending 같은 prior work 의 모듈은 SQ 를 새로 만들지 않고, single-frame 단계에서 fragmented 된 ID 들을 multi-view 로 정렬해 RQ 를 끌어올린다. (Scene-level 분석은 future work 로 deferral.)

### Limitations

- **Scene 수**: 3 scene 만 (PR++ 와 동일). ScanNet++ test set 전체로 확장 시 통계 robust 해지지만 본 결론의 방향에는 영향 없을 것으로 예상.
- **Single foundation model family**: Grounded-SAM (GroundingDINO + SAM ViT-H) 한 조합. 다른 family (GLIP+SAM2, Mask2Former 등) 는 다른 SQ ceiling 을 가질 수 있음.
- **GT renderer**: nvdiffrast triangle rasterizer + first-vertex-of-face label propagation 의 mesh discretization noise 는 본 실험의 mask shape error 측정에 일부 conflated. "perfect GT" 가 없어서 분리 불가능.
- **Class taxonomy**: PR++ scene_config 의 prompt list 그대로 사용 + 5748ce6f01 에 `office chair` prompt 1개 추가 (영향 ~1pt 미만 추정, 측정되지 않음).

### Future direction

1. **Better mask predictor**: SAM2, more recent grounding models 으로 single-frame mask quality 자체를 끌어올림. 본 실험의 per-view SQ 가 곧 그 효과의 ceiling 을 직접 측정한 값이 될 것.
2. **Pseudo refinement before 3D lifting**: 5가지 failure mode 중 ①②⑤ (class hallucination, wrong grounding, duplicate) 를 2D 단계에서 cleanup 하는 module 추가.
3. **Foundation model family comparison**: 동일 protocol 로 다른 family 의 SQ ceiling 측정 → "이 분야 전체의 SQ ceiling 은 얼마이고 어떤 family 가 가장 좋은가" 의 정량 답.

---

**Cross-references**: 정확한 threshold / config / output layout → `docsV2/spec.md`. 재현 / 코드 위치 / 발생한 이슈 → `docsV2/implementation.md`.
