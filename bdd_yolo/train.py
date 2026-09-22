import time
from datetime import timedelta

import torch
from ultralytics import YOLO

from config import (BATCH, DATASET_NAME, DATASETS_ROOT, DEVICE, EPOCHS, EXPERIMENT_NAME,
                    EXPERIMENTS_ROOT, GPU_MEMORY_FRACTION, IMGSZ, MODEL, PATIENCE, PRETRAINED,
                    SEED, TRAIN_OVERRIDES, WORKERS)

_state = {}


def on_train_start(trainer):
    _state["t0"] = time.time()


def on_fit_epoch_end(trainer):
    done, total = trainer.epoch + 1, trainer.epochs
    elapsed = time.time() - _state.get("t0", time.time())
    per_epoch = elapsed / done
    eta = per_epoch * (total - done)
    print(f"[{done}/{total}] {per_epoch:.0f}s/epoch | elapsed {timedelta(seconds=int(elapsed))} "
          f"| ETA {timedelta(seconds=int(eta))} | finish ~{time.strftime('%H:%M', time.localtime(time.time() + eta))}")


def main():
    print("experiment:", EXPERIMENT_NAME, TRAIN_OVERRIDES or "")
    if GPU_MEMORY_FRACTION:
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, int(DEVICE))
    model = YOLO(MODEL)
    if MODEL.endswith(".yaml") and PRETRAINED:
        model.load(PRETRAINED)
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.train(
        data=str(DATASETS_ROOT / DATASET_NAME / "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=str(EXPERIMENTS_ROOT),
        name=EXPERIMENT_NAME,
        pretrained=True,
        patience=PATIENCE,
        seed=SEED,
        exist_ok=True,
        **TRAIN_OVERRIDES,
    )


if __name__ == "__main__":
    main()
