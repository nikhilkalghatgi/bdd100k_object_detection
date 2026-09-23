import random
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from config import (CONF, EVAL_EXPERIMENT, EXPERIMENTS_ROOT, ID_TO_CLASS, PRED_MIN_PER_CLASS,
                    PRED_N, PRED_SEED)

COLORS = {"car": "#00A3FF", "person": "#00E676", "bus": "#FF9500",
          "truck": "#FF3B30", "bike": "#C77DFF"}


def get_font(size=16):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(path, boxes, classes, confs, font, out_path):
    image = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(image)
    for (x1, y1, x2, y2), cid, conf in zip(boxes, classes, confs):
        name = ID_TO_CLASS[int(cid)]
        color = COLORS[name]
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{name} {conf:.2f}"
        tw = d.textlength(label, font=font)
        th = font.size + 4 if hasattr(font, "size") else 16
        ty = y1 - th if y1 - th >= 0 else y1
        d.rectangle([x1, ty, x1 + tw + 6, ty + th], fill=color)
        d.text((x1 + 3, ty + 1), label, fill="black", font=font)
    image.save(out_path, quality=90)


def main():
    exp_dir = EXPERIMENTS_ROOT / EVAL_EXPERIMENT
    out_dir = exp_dir / "predictions_bbox"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = yaml.safe_load((exp_dir / "args.yaml").read_text())
    imgsz = int(train_args.get("imgsz", 640))
    data_cfg = yaml.safe_load(Path(train_args["data"]).read_text())
    val_dir = Path(data_cfg["path"]) / data_cfg["val"]

    paths = sorted(val_dir.glob("*.jpg"))
    random.Random(PRED_SEED).shuffle(paths)
    print(f"{EVAL_EXPERIMENT} | imgsz {imgsz} | {len(paths)} val images available")

    model = YOLO(str(exp_dir / "weights/best.pt"))
    font = get_font()

    counts = Counter()
    saved = []
    for path in paths:
        if len(saved) >= PRED_N and all(counts[c] >= PRED_MIN_PER_CLASS for c in COLORS):
            break

        r = model.predict(str(path), imgsz=imgsz, conf=CONF, verbose=False)[0]
        if not len(r.boxes):
            continue
        present = {ID_TO_CLASS[int(c)] for c in r.boxes.cls.tolist()}

        needed = {c for c in COLORS if counts[c] < PRED_MIN_PER_CLASS}
        # take it if it fills a quota, or if we still have slots to fill
        if not (present & needed) and len(saved) >= PRED_N:
            continue

        draw(path, r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
             r.boxes.conf.cpu().numpy(), font, out_dir / path.name)
        saved.append(path.name)
        for c in present:
            counts[c] += 1

    print(f"\nsaved {len(saved)} images to {out_dir}")
    print("images containing each class:")
    for c in COLORS:
        flag = "" if counts[c] >= PRED_MIN_PER_CLASS else "   <-- below target"
        print(f"  {c:7s} {counts[c]:4d}{flag}")


if __name__ == "__main__":
    main()
