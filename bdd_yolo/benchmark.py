import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from config import (BENCH_EXPERIMENTS, BENCH_HALF, BENCH_IMAGES, BENCH_TENSORRT, BENCH_WARMUP,
                    CONF, EXPERIMENTS_ROOT)


def load_images(val_dir, n):
    # decode up front so we time the model, not the disk
    return [cv2.imread(str(p)) for p in sorted(val_dir.glob("*.jpg"))[:n]]


def bench(model, images, imgsz, half):
    for im in images[:BENCH_WARMUP]:
        model.predict(im, imgsz=imgsz, conf=CONF, half=half, verbose=False)
    torch.cuda.synchronize()

    total, pre, inf, post, dets = [], [], [], [], []
    for im in images:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(im, imgsz=imgsz, conf=CONF, half=half, verbose=False)[0]
        torch.cuda.synchronize()
        total.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"])
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])
        dets.append(len(r.boxes))
    return {
        "mean": np.mean(total), "p50": np.percentile(total, 50),
        "p95": np.percentile(total, 95), "p99": np.percentile(total, 99),
        "preprocess": np.mean(pre), "inference": np.mean(inf), "postprocess": np.mean(post),
        "fps_p50": 1000 / np.percentile(total, 50), "objects": np.mean(dets),
    }


def main():
    print(torch.cuda.get_device_name(0))
    rows = []

    for name in BENCH_EXPERIMENTS:
        exp_dir = EXPERIMENTS_ROOT / name
        weights = exp_dir / "weights/best.pt"
        if not weights.exists():
            print("skip (no weights):", name)
            continue

        train_args = yaml.safe_load((exp_dir / "args.yaml").read_text())
        imgsz = int(train_args.get("imgsz", 640))
        data_cfg = yaml.safe_load(Path(train_args["data"]).read_text())
        images = load_images(Path(data_cfg["path"]) / data_cfg["val"], BENCH_IMAGES)

        variants = [("PyTorch FP32", False)] + ([("PyTorch FP16", True)] if BENCH_HALF else [])
        for label, half in variants:
            r = bench(YOLO(str(weights)), images, imgsz, half)
            r.update(experiment=name, imgsz=imgsz, variant=label)
            rows.append(r)
            print(f"{name:26s} {label:13s} p50 {r['p50']:6.2f} ms  p95 {r['p95']:6.2f} ms  "
                  f"{r['fps_p50']:5.1f} fps")

        if BENCH_TENSORRT:
            engine = YOLO(str(weights)).export(format="engine", half=True, imgsz=imgsz, device=0)
            r = bench(YOLO(str(engine)), images, imgsz, False)
            r.update(experiment=name, imgsz=imgsz, variant="TensorRT FP16")
            rows.append(r)
            print(f"{name:26s} {'TensorRT FP16':13s} p50 {r['p50']:6.2f} ms  p95 {r['p95']:6.2f} ms  "
                  f"{r['fps_p50']:5.1f} fps")

    out = EXPERIMENTS_ROOT / "latency_benchmark.md"
    lines = ["# Inference latency", "",
             f"Batch 1, {BENCH_IMAGES} real val images after {BENCH_WARMUP} warmup iterations, "
             f"CUDA-synchronised, on {torch.cuda.get_device_name(0)}.",
             "Images are decoded up front, so disk I/O is excluded.", "",
             "| experiment | imgsz | variant | p50 ms | p95 ms | p99 ms | fps (p50) | pre | infer | NMS | objects |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['imgsz']} | {r['variant']} | {r['p50']:.2f} | {r['p95']:.2f} "
            f"| {r['p99']:.2f} | {r['fps_p50']:.1f} | {r['preprocess']:.2f} | {r['inference']:.2f} "
            f"| {r['postprocess']:.2f} | {r['objects']:.1f} |")
    lines += ["", "A 30 fps camera allows 33.3 ms per frame, shared with everything else in the stack."]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\nsaved to", out)


if __name__ == "__main__":
    main()
