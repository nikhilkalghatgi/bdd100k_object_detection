# Road-User Detection on BDD100K — Technical Report

Detecting five road-user classes (**car, bus, person, truck, bike**) in BDD100K driving images,
taken through data analysis, training, evaluation, failure analysis, and improvement.

---

## 1. Data and preprocessing

Training on all 70k images was not practical on a single laptop GPU, so we
sample a subset. Random sampling would preserve BDD's extreme class imbalance, so instead images
are picked **round-robin across classes, rarest first** (Sort classes rarest first by how many images contain them).

| class | share of objects in full BDD | share in our 10k subset |
|---|---|---|
| car | 83.6% | 66.2% |
| person | 10.7% | 22.4% |
| truck | 3.5% | 5.2% |
| bike | 0.8% | 3.3% |
| bus | 1.4% | 3.0% |

Bike and bus each gain roughly 3x representation. The imbalance is reduced, not solved.

**Fixed split.** The val subset is drawn with a fixed seed and never changed, so every experiment
below is scored on identical images.

---

## 2. Model choice

**YOLO11s**, pretrained on COCO. Reasoning:

- Single-stage detector — real-time by design, which is the point for ADAS.
- The `s` size (9.5M params, 21.8 GFLOPs @640) trains in a few hours on an RTX 4060 Laptop,
  so several experiments fit in the available time. `n` would underfit; `m`/`l` would allow
  only one run.
- COCO pretraining already covers car/bus/truck/person/bicycle, so fine-tuning converges fast.

Hardware for all runs: RTX 4060 Laptop, 8 GB VRAM.

---

## 3. Experiment 1 — pilot baseline (2k images)

**Purpose.** Prove the pipeline end to end and get a first read on failure modes.

**Setup.** 2,000 train / 500 val images, YOLO11s @ 640 px, batch 16, 50 epochs.

**Results.** mAP50 **0.5354**, mAP50-95 0.3088, precision 0.6310, recall 0.4912.

| class | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| car | 0.7415 | 0.6259 | 0.6904 | 0.4109 |
| person | 0.6962 | 0.4623 | 0.5518 | 0.2532 |
| bus | 0.6347 | 0.4692 | 0.5362 | 0.3922 |
| truck | 0.5460 | 0.4316 | 0.4476 | 0.2910 |
| bike | 0.5365 | 0.4670 | 0.4510 | 0.1968 |

**Problems found.**

1. **Too little data.** 2,000 images is under 3% of BDD. Rare classes saw only a few hundred
   instances, so their numbers were as much noise as signal.
2. **A 500-image val set is too small to slice.** Splitting it by weather or time of day left
   buckets of 2–45 images — far too few to trust.
3. **Small objects already stood out as the weak point**: recall 0.395 on small objects vs
   0.881 on large.

**Decision.** Scale the data up before drawing any conclusions.

---

## 4. Experiment 2 — main baseline (10k images)

**What changed and why.** Train images 2,000 → **10,000** and val 500 → **2,000**, addressing
problems 1 and 2 above. Everything else held constant (YOLO11s, 640 px, batch 16, 50 epochs).

**Results.** mAP50 **0.6134**, mAP50-95 0.3747, precision 0.6902, recall 0.5540. Converged at
epoch 39 and stayed flat to 49 — so 50 epochs was sufficient, and more would not help.
Training time 4.54 h.

| class | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| car | 0.7739 | 0.6883 | 0.7489 | 0.4597 |
| bus | 0.6723 | 0.5600 | 0.6224 | 0.4642 |
| person | 0.7385 | 0.5189 | 0.6082 | 0.2903 |
| truck | 0.6372 | 0.5317 | 0.5832 | 0.4082 |
| bike | 0.6292 | 0.4709 | 0.5045 | 0.2509 |

*(Not directly comparable to Experiment 1, which used a different, smaller val set.)*

**What this table exposes.** Three problems are visible before any slicing:

- **Recall is the weak side everywhere.** Every class has precision well above recall
  (e.g. person 0.739 vs 0.519). The model is cautious — when it fires it is usually right, but it
  misses a lot. For ADAS that is the wrong way round: a missed pedestrian is far more dangerous
  than a spurious box.
- **mAP50-95 collapses relative to mAP50**, and worst for the small-object classes: person drops
  0.608 → 0.290 and bike 0.505 → 0.251, roughly halving, while car only falls 0.749 → 0.460.
  Scores hold at a loose IoU and fall apart at a strict one, meaning boxes are found but poorly
  positioned.
- **A 24-point spread between best and worst class** (car 0.749, bike 0.505). Bike is the weakest
  on every single metric.

### Weakness 1 — small objects

Recall by object area (COCO buckets, measured at confidence 0.25):

| size | count | share | recall |
|---|---|---|---|
| small (<32²) | 13,279 | 42% | **0.4515** |
| medium (32²–96²) | 12,722 | 41% | 0.7812 |
| large (>96²) | 5,279 | 17% | 0.9237 |

The worst-performing bucket is also the **largest** — 42% of all ground-truth objects. Per class
it is worse still: bus-small 0.088, truck-small 0.147, bike-small 0.181, person-small 0.359.

This is a resolution problem. Input images are 1280×720, but training runs at 640 px, halving
every dimension: a 20-pixel-wide distant car becomes 10 pixels.

Supporting evidence: **highway is the worst scene** (mAP50 0.546 vs 0.624 for city street) —
highways are exactly where the relevant vehicles are far away and therefore small.

### Weakness 2 — imprecise box localisation

False positives by cause:

| cause | count |
|---|---|
| localization (overlaps a real object, but IoU < 0.5) | 4,710 |
| duplicate | 1,741 |
| background (nothing there) | 1,715 |
| class confusion | 1,017 |

Localization dominates: the model *finds* objects but draws sloppy boxes. The same effect shows
in the gap between mAP50 (0.613) and mAP50-95 (0.375) — scores collapse as the IoU requirement
tightens. Small objects amplify this, since being a few pixels off destroys IoU on a tiny box.

### Class imbalance — mitigated, and not the real bottleneck

The sampler (§1) reduced imbalance but did not remove it: car still outnumbers bus 22:1. The
results, however, show that **object size hurts far more than class frequency does**.

Bus is the *rarest* class (4,789 instances) yet scores second best (mAP50 0.622), while person has
7x more instances and scores lower (0.608) — because only 17% of bus objects are small, against
50% for person. More directly, compare the two spreads:

- **Between classes**, mAP50 ranges 0.505 (bike) to 0.749 (car) — a **1.5x** spread.
- **Within a single class**, small-to-large recall ranges 0.088 → 0.807 for bus (**9.2x**),
  0.147 → 0.777 for truck (5.3x), 0.181 → 0.833 for bike (4.6x).

The gap inside each class is several times larger than the gap between classes. That is why the
improvement effort targeted resolution rather than further rebalancing. Bike remains the weakest
class because it is hit by both problems at once — rare *and* mostly small.

Two caveats: no random-sampling control run was trained, so the sampler's benefit is reasoned
rather than measured; and rare-class data is still unused — full BDD holds 11,672 bus and 7,210
bike instances against the 4,789 and 5,187 in our subset.

### Other findings (real but secondary)

- **Occlusion costs ~27 points of recall**: 0.845 visible vs 0.581 occluded.
- **Class confusion is almost entirely vehicle-vs-vehicle**: truck↔car 500, truck↔bus 257,
  bus↔car 152 — 89% of all class confusions. person↔bike is only 62. BDD's vans and pickups are
  genuinely ambiguous; this is a label-taxonomy issue more than a model failure.
- **Weather and lighting matter less than expected**: night 0.592 vs daytime 0.616, snowy 0.579
  vs clear 0.597. Real, but far smaller than the size effect — so an improvement aimed at night
  would have been aimed at the wrong problem.

**Decision.** Attack small-object detection. Two candidate fixes: add a finer-grained detection
head (cheap), or train at full resolution (expensive). Test the cheap one first.

---

## 5. Experiment 3 — P2 detection head

**What changed and why.** YOLO11 detects at strides 8, 16 and 32 (P3/P4/P5). We added a fourth
head at **stride 4 (P2)**, giving a 160×160 grid instead of 80×80 as the finest scale — more
spatial precision for small objects. Since Ultralytics ships no P2 variant for YOLO11, the config
was written by hand (`yolo11-p2.yaml`), porting the YOLOv8-P2 head onto the YOLO11 backbone.

COCO weights were transferred into the new architecture (297 of 593 tensors — the backbone
matches, the re-indexed head does not), so this is still fine-tuning, not training from scratch.

Cost: **29.8 GFLOPs vs 21.8**, a 37% increase. Batch reduced 16 → 8 to fit VRAM. 50 epochs,
4.40 h, converged at epoch 40.

**Results.**

| metric | baseline | P2 | change |
|---|---|---|---|
| mAP50 | 0.6134 | **0.6245** | +0.0111 |
| mAP50-95 | 0.3747 | 0.3798 | +0.0051 |
| precision | 0.6902 | 0.6821 | −0.0081 |
| recall | 0.5540 | 0.5548 | +0.0008 |

| class | precision | recall | mAP50 | mAP50-95 | mAP50 vs baseline |
|---|---|---|---|---|---|
| car | 0.7897 | 0.6840 | 0.7696 | 0.4755 | +0.0208 |
| person | 0.7349 | 0.5326 | 0.6316 | 0.3037 | **+0.0235** |
| bus | 0.6363 | 0.5615 | 0.6211 | 0.4607 | −0.0013 |
| truck | 0.6404 | 0.5384 | 0.5934 | 0.4121 | +0.0102 |
| bike | 0.6094 | 0.4573 | 0.5065 | 0.2471 | +0.0020 |

Person — the most small-object-heavy class — improved most, which is the intended direction. But
the baseline's structural problems are all still present: recall still trails precision in every
class, person's mAP still halves from 0.632 to 0.304 under strict IoU, and bike is still last on
every metric.

False positives dropped sharply: localization 4,710 → 3,689 (−1,021) and background 1,715 → 1,376
(−339).

**An important measurement caveat.** The fixed-confidence slice metrics appear to get *worse*
(small-object recall 0.4515 → 0.4439). This is misleading. mAP is threshold-free, but those slice
numbers are measured at a fixed confidence of 0.25, and the P2 model is systematically **less
confident**. At the same cutoff it emits fewer boxes — fewer false ones (−1,360) and marginally
fewer true ones. The two models are being compared at different operating points, so the
fixed-confidence slice table is not a fair basis for comparison here. Judge this experiment on mAP.

**Verdict: a real but poor-value improvement.** +1.1 mAP50 points for +37% compute, and the
small-object gap is essentially untouched (small recall still ~0.44 vs ~0.92 for large).

**What this tells us.** The bottleneck is not the stride of the detection head. It is that
information is destroyed *before* the network sees it, when a 1280×720 image is squeezed into
640 px. No head at any stride can recover pixels that no longer exist. That points directly at
the remaining experiment.

---

## 6. Experiment 4 — full-resolution training (1280 px)

**What changed and why.** Input resolution 640 → **1280 px**, matching BDD's native width, so
distant objects keep their original pixel size instead of being halved. Experiment 3 showed that
adding resolution *inside* the network barely helped; this test adds it at the *input*, which is
where the information is actually being lost.

**Setup.** YOLO11s @ 1280 px, batch 6 (VRAM-limited), **88.8 GFLOPs** — 4.1x the baseline.
Stopped at epoch 36 of a planned 40 once the curve had been flat for ~8 epochs; best weights come
from epoch 35. Training took 5.5 h at 9.1 min/epoch — only **1.64x** the baseline's wall clock
despite 4.1x the arithmetic, because the 640 run was partly dataloader-bound and left the GPU idle.

**Results.**

| metric | baseline | 1280 | change |
|---|---|---|---|
| mAP50 | 0.6134 | **0.6837** | **+0.0703** |
| mAP50-95 | 0.3747 | **0.4270** | **+0.0523** |
| precision | 0.6902 | 0.7364 | +0.0462 |
| recall | 0.5540 | 0.6022 | +0.0482 |

| class | precision | recall | mAP50 | mAP50-95 | mAP50 vs baseline |
|---|---|---|---|---|---|
| car | 0.8364 | 0.7276 | 0.8190 | 0.5164 | +0.0701 |
| person | 0.7951 | 0.5902 | 0.7080 | 0.3630 | **+0.0998** |
| bus | 0.6727 | 0.6119 | 0.6728 | 0.5071 | +0.0504 |
| truck | 0.6665 | 0.5797 | 0.6425 | 0.4549 | +0.0593 |
| bike | 0.7114 | 0.5016 | 0.5763 | 0.2934 | +0.0718 |

Every class improved on every metric. Person — the most small-object-heavy class — gained most.

**This one is a real capability gain, not a threshold shift.** Experiment 3 needed a caveat
because P2 traded fewer false positives for fewer true ones. Here precision *and* recall both rose,
and every false-positive category fell at the same time:

| FP cause | baseline | 1280 | change |
|---|---|---|---|
| localization | 4,710 | 3,210 | −1,500 |
| duplicate | 1,741 | 1,145 | −596 |
| background | 1,715 | 1,344 | −371 |
| class confusion | 1,017 | 944 | −73 |

The 32% drop in localization errors is the direct evidence that sharper input produces
better-positioned boxes — and it is mirrored in mAP50-95 rising faster in relative terms than mAP50.

**Did the small-object gap close?** Substantially, though not completely:

| size | baseline | 1280 | change |
|---|---|---|---|
| small | 0.4515 | **0.5263** | **+0.0748** |
| medium | 0.7812 | 0.8018 | +0.0207 |
| large | 0.9237 | 0.9191 | −0.0045 |

The gains land exactly where the diagnosis predicted — concentrated in small objects, negligible
for large ones (which were already at 0.92 and had nothing to gain). Per class at small size:
truck **+0.1367**, bus **+0.1351**, bike **+0.1164**, person +0.0945, car +0.0636 — the worst
performers improved the most. Bus-small went from 0.088 to 0.223, a 2.5x improvement.

The small-to-large ratio narrowed from 2.0x to 1.7x. Small objects remain the weakest bucket, so
resolution reduced the problem rather than eliminating it.

**Scene slices confirm the causal chain.** Highway was the baseline's worst scene, which we
attributed to distant vehicles. It gained the most: **+0.0879** (0.5456 → 0.6335). Night gained
least (+0.0543), consistent with low-light being a separate, smaller problem that resolution does
not address. Occluded-object recall also improved (0.5813 → 0.6284), as partial objects become
easier to resolve with more pixels.

**Cost-effectiveness — the honest comparison.** On marginal return per unit of compute, the two
improvements are closer than the headline suggests:

| improvement | extra GFLOPs | mAP50 gain | gain per GFLOP |
|---|---|---|---|
| P2 head | +8.0 | +0.0111 | 0.0014 |
| 1280 input | +67.0 | +0.0703 | 0.0010 |

P2 is marginally more efficient per FLOP, but it plateaus: it cannot deliver the size of gain the
problem requires, no matter how cheaply. Resolution is the only lever that moved the needle
meaningfully. Which of these is "right" depends on the deployment budget — a point returned to
in §8.

---

## 7. Summary

| # | experiment | change | GFLOPs | mAP50 | mAP50-95 | small recall | verdict |
|---|---|---|---|---|---|---|---|
| 1 | 2k pilot | establish pipeline | 21.8 | 0.5354* | 0.3088* | 0.395* | too small to conclude from |
| 2 | 10k baseline | 5x more data | 21.8 | 0.6134 | 0.3747 | 0.4515 | reference point |
| 3 | P2 head | detect at stride 4 | 29.8 | 0.6245 | 0.3798 | 0.4439 | +1.1 pts for +37% compute — poor value |
| 4 | **1280 px** | native resolution | 88.8 | **0.6837** | **0.4270** | **0.5263** | **+7.0 pts for 4.1x compute — the fix** |

\* different val set; not directly comparable.

**The through-line.** Failure analysis on the baseline pointed at small objects (42% of all boxes,
recall 0.45 against 0.92 for large) and imprecise localization. Two fixes were tested against that
diagnosis. Adding a finer detection head at the same input size moved almost nothing, which ruled
out network stride as the bottleneck. Feeding the network full-resolution images moved everything —
and moved it most for exactly the classes, object sizes and scenes the diagnosis had flagged. 
**The
conclusion is that the limiting factor was never model capacity or architecture, but information
discarded before inference began.**

---

## 8. Limitations and the path to production

**Limitations of this work.**

- Trained on 10k of 70k available images — more data would lift all numbers.

**Toward a production ADAS system.**

- **Latency**: export to TensorRT FP16, then INT8 with calibration — typically 2–3x and 4–6x
  faster respectively, for 1–2 mAP points. Report p95 latency at batch 1 including pre- and
  post-processing, not mean throughput.
- **Temporal**: detect every Nth frame and track between (ByteTrack/Kalman) — cuts effective cost
  several-fold and stabilises detections across frames.
- **Safety-relevant metrics**: recall at a fixed low false-positive rate matters far more than
  mAP, and should be reported per distance band. A missed pedestrian and a missed distant car are
  not equivalent failures.
- **Robustness**: validate on held-out geographies, night and adverse weather; monitor for domain
  shift and out-of-distribution input in the field.

---

## 9. Future work — next steps by weakness

Ordered by how much they would move the final model (Experiment 4, mAP50 0.6837).

### 9.1 Small objects — still the largest gap

Resolution narrowed it but did not close it: recall 0.526 small vs 0.919 large, **6,290 small
objects still missed**. Error analysis shows these are genuine misses, not mislabels — class
confusions total only 944 across all sizes — so the object is never found at all.

- **Combine P2 with 1280.** The two fixes were only ever tested *separately*. A stride-4 head at
  full resolution gives a 320×320 finest grid, and the combination is untested. This is the
  cheapest remaining experiment and the obvious next run.
- **Tiled inference (SAHI).** Split each frame into overlapping crops, detect at native scale,
  merge. Usually the single biggest small-object gain available without retraining — but it
  multiplies inference cost, so it suits an offline/auto-labelling role more than live ADAS.
- **Push resolution further (1536/1920).** Expect diminishing returns against steeply rising cost;
  worth one run to locate the plateau rather than assuming it.
- **Multi-scale training** (`scale` augmentation range, or randomised input size) so small
  instances are seen more often during training.
- **Architectural alternatives**: a BiFPN-style neck, or a query-based detector (RT-DETR), both of
  which handle scale variation differently from a stride-based FPN.

### 9.2 Occlusion — the second-largest gap

Recall 0.628 occluded vs 0.868 visible, a 24-point gap that resolution barely touched.

- **Soft-NMS or Set-NMS** instead of greedy NMS, so heavily overlapping true objects are not
  suppressed by each other. Cheap, inference-only, directly targets crowds.
- **Temporal tracking.** An object occluded for a few frames is still there. A tracker
  (ByteTrack/Kalman) carries identity across the gap and is the standard production answer.
- **Mine crowded frames**: the worst-image list is dominated by dense city scenes with 30-50
  objects; oversampling these would target the failure directly.

### 9.3 Vehicle class confusion

944 class confusions, of which **609 are vehicle-vs-vehicle** (truck→car 272, truck→bus 146,
bus→truck 114, bus→car 77). Notably, truck and bus lose *more* objects to misclassification than
to small-size misses — for these two classes this is the dominant failure, not detection.

- **Merge car/truck/bus into a "vehicle" superclass.** Removes 609 errors outright, and for
  collision avoidance the subtype rarely changes the control decision. Redefines the task, so it is
  a judgement call — but a defensible one.
- **Hierarchical head**: detect "vehicle", then classify subtype as a second stage, so a subtype
  error no longer costs the detection.
- **Rebalance rare classes.** Full BDD holds 11,672 bus and 7,210 bike instances against the 4,789
  and 5,187 used here. Selecting *every* bus/truck/bike image before filling the remainder doubles
  rare-class exposure at zero training cost. Expect this to help bus and truck specifically — it
  will not help small-object recall, which is a detection problem.
- **Label audit** of ambiguous vans/pickups; some share of this gap is annotation noise, not model
  error.


### 9.4 Scale, rigour and deployment

- **Train on all 70k images.** At 640 this is ~10 h for 15 epochs (more data needs fewer epochs);
  at 1280 it is ~4 days locally, or roughly $20-30 of cloud A100 time.
- **Distillation**: use the 1280 model as a teacher for a smaller/lower-resolution student, aiming
  to keep most of the accuracy at a fraction of the 88.8 GFLOPs.
- **Measure latency and quantize** — benchmark p95 at batch 1, then TensorRT FP16 and INT8, to turn
  the accuracy/compute tradeoff from a reasoned argument into an evidenced one.
