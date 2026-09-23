from pathlib import Path

# ----------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent
BDD_ROOT = PROJECT_ROOT.parent / "archive (1)"

TRAIN_JSON = BDD_ROOT / "train/annotations/bdd100k_labels_images_train.json"
VAL_JSON = BDD_ROOT / "val/annotations/bdd100k_labels_images_val.json"
TRAIN_IMAGES = BDD_ROOT / "train/images"
VAL_IMAGES = BDD_ROOT / "val/images"

DATASETS_ROOT = PROJECT_ROOT / "datasets"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

CLASSES = {"car": 0, "bus": 1, "person": 2, "truck": 3, "bike": 4}
ID_TO_CLASS = {v: k for k, v in CLASSES.items()}
IMAGE_W, IMAGE_H = 1280, 720

# ----------------------------------------------------------------- dataset
N_TRAIN = 10000
N_VAL = 2000
SEED = 42
DATASET_NAME = f"bdd5_{N_TRAIN // 1000}k"

# ----------------------------------------------------------------- train
EXPERIMENT = "highres"  # baseline | highres | p2

# the "s" in yolo11s-p2.yaml only selects the scale; the file on disk is yolo11-p2.yaml
_PRESETS = {  # MODEL, IMGSZ, BATCH
    "baseline": ("yolo11s.pt", 640, 16),
    "highres": ("yolo11s.pt", 1280, 6),
    "p2": (str(PROJECT_ROOT / "yolo11s-p2.yaml"), 640, 8),
}
MODEL, IMGSZ, BATCH = _PRESETS[EXPERIMENT]

PRETRAINED = "yolo11s.pt"  # transferred into .yaml models, which have no weights of their own
EPOCHS = 40  # baseline and p2 ran 50 and both peaked at epoch 39-40
WORKERS = 4
DEVICE = "0"
PATIENCE = 10
GPU_MEMORY_FRACTION = None # 0.55  # hard cap on VRAM (0.55 of 8GB = ~4.5GB); None = no cap
TRAIN_OVERRIDES = {}  # extra ultralytics args, e.g. {"mosaic": 0.5, "lr0": 0.01}

EXPERIMENT_NAME = f"{Path(MODEL).stem}_{IMGSZ}_{DATASET_NAME}"

# ----------------------------------------------------------------- eval
EVAL_EXPERIMENT = EXPERIMENT_NAME  # set to a folder name to evaluate an older run

CONF = 0.25
IOU = 0.5
MIN_SLICE_IMAGES = 50
WORST_N = 12

# ----------------------------------------------------------------- benchmark
BENCH_EXPERIMENTS = [  # runs to compare; each needs weights/best.pt
    "yolo11s_640_bdd5_10k",
    "yolo11s-p2_640_bdd5_10k",
    "yolo11s_1280_bdd5_10k",
]
BENCH_IMAGES = 200  # timed iterations per config, batch 1
BENCH_WARMUP = 20
BENCH_HALF = True  # also measure FP16
BENCH_TENSORRT = False  # also export to TensorRT FP16 and measure (needs: pip install tensorrt)

# ----------------------------------------------------------------- prediction visuals
PRED_N = 100  # images to save
PRED_MIN_PER_CLASS = 10  # keep sampling until every class appears in at least this many
PRED_SEED = 0
