# Road-User Detection on BDD100K using Yolov11s

Object detection for five classes - **car, bus, person, truck, bike** — trained and
evaluated on the BDD100K driving dataset, taken through the full pipeline: data analysis →
preprocessing → model selection → training → evaluation → failure analysis → improvement.

**Note:** Sample predictions drawn on val images are available in **[bdd_yolo/experiments/yolo11s_1280_bdd5_10k/predictions_bbox](bdd_yolo/experiments/yolo11s_1280_bdd5_10k/predictions_bbox)**

Full Technical Report: **[bdd_yolo/TECHNICAL_REPORT.md](bdd_yolo/TECHNICAL_REPORT.md)**

---

## Results

All runs use YOLO11s on an identical fixed validation split (2,000 images), except the pilot.

| experiment | data | input | GFLOPs | mAP50 | mAP50-95 | small-object recall |
|---|---|---|---|---|---|---|
| pilot | 2k | 640 | 21.8 | 0.5354\* | 0.3088\* | 0.395\* |
| baseline | 10k | 640 | 21.8 | 0.6134 | 0.3747 | 0.4515 |
| + P2 head | 10k | 640 | 29.8 | 0.6245 | 0.3798 | 0.4439 |
| **+ full resolution** | 10k | **1280** | 88.8 | **0.6837** | **0.4270** | **0.5263** |

\* pilot used a smaller 500-image val split; not directly comparable.

**Headline finding:** small objects (42% of all ground-truth boxes) were the dominant failure
mode. Adding a stride-4 detection head cost +37% compute for +1.1 mAP50. Training at native
1280 resolution instead cost 4.1x compute for **+7.0 mAP50** and closed a meaningful part of the
small-object gap — confirming the bottleneck was information destroyed by downsampling, not
model capacity.

---

## Setup

Requires Python 3.10+, an NVIDIA GPU, and the BDD100K dataset extracted.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics pandas matplotlib tabulate
```

On Windows the default PyPI `torch` wheel is CPU-only — use the CUDA index URL above or training
will be roughly 50x slower.

Expected dataset layout:

```
dataset/
  train/images/*.jpg          70,000 images, 1280x720
  train/annotations/bdd100k_labels_images_train.json
  val/images/*.jpg            10,000 images
  val/annotations/bdd100k_labels_images_val.json
  test/*.jpg                  20,000 images (unlabelled, unused)
```

---

## Usage

Everything is driven by `bdd_yolo/config.py` — the scripts take no command-line arguments.
Edit the config, then run the scripts in order.

```bash
cd bdd_yolo
python prepare_dataset.py    # build the YOLO-format subset
python train.py              # train
python evaluate.py           # metrics, slices, failure analysis
```

### Choosing an experiment

`config.py` has one switch that selects model, input size and batch together:

```python
EXPERIMENT = "baseline"   # baseline | highres | p2

_PRESETS = {              # MODEL, IMGSZ, BATCH
    "baseline": ("yolo11s.pt", 640, 16),
    "highres":  ("yolo11s.pt", 1280, 6),
    "p2":       (PROJECT_ROOT / "yolo11s-p2.yaml", 640, 8),
}
```

Output folders are named automatically as `{model}_{imgsz}_{dataset}`, so experiments never
overwrite each other and `evaluate.py` always targets the run you just trained.

Re-run `prepare_dataset.py` only when `N_TRAIN`, `N_VAL` or `SEED` change — input resolution is a
training-time setting and does not affect the data on disk.

### Resuming an interrupted run

```bash
python -c "from ultralytics import YOLO; YOLO('experiments/<name>/weights/last.pt').train(resume=True)"
```

---

## Layout

```
analyze_bdd100k.py          standalone dataset analysis (classes, sizes, weather, scenes)
bdd100k_report/             its output — report.md, CSVs, charts

bdd_yolo/
  config.py                 all settings: paths, dataset, train presets, eval
  prepare_dataset.py        BDD JSON -> YOLO labels + balanced subset sampling
  train.py                  training with per-epoch ETA logging
  evaluate.py               metrics, scenario/size slices, failure analysis
  yolo11-p2.yaml            custom YOLO11 + stride-4 head (not shipped by ultralytics)
  TECHNICAL_REPORT.md       full write-up

  datasets/<name>/          generated YOLO-format data (images, labels, data.yaml)
  experiments/<name>/       weights, training curves, confusion matrix, args.yaml
             └── eval/      report.md, per-class + slice CSVs, charts, failure overlays
```

---

## What evaluation produces

`evaluate.py` writes to `experiments/<name>/eval/`:

- **`report.md`** — everything below, in one readable document
- **Per-class metrics** — precision, recall, mAP50, mAP50-95
- **Scenario slices** — mAP by weather, scene and time of day, each computed by re-running
  validation on that subset of images
- **Size slices** — recall by COCO size bucket (small / medium / large), per class
- **Occlusion slices** — recall for occluded and truncated objects
- **Failure analysis** — false positives classified as localization / duplicate / background /
  class confusion, plus the most-confused class pairs
- **Failure overlays** — the worst images annotated: green = correct, red = false positive,
  yellow = missed

Slice metrics are recall at a fixed confidence threshold, so they are only comparable between
models that share an operating point — see the note in the technical report.

---

## Notes and limitations

- Trained on 10k of 70k available images; full-data training at 1280 would take roughly 4 days
  on a single laptop GPU.
- BDD's `test` split is unlabelled, so BDD's `val` split serves as the held-out set, and the
  validation subset is drawn from it with a fixed seed.
- Single seed per configuration, so differences under ~0.5 mAP points should not be over-read.
- Hardware used: RTX 4060 Laptop (8 GB). Baseline ~4.5 h, 1280 run ~6 h.
