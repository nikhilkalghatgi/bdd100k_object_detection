import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from config import (CLASSES, DATASET_NAME, DATASETS_ROOT, ID_TO_CLASS, IMAGE_H, IMAGE_W,
                    N_TRAIN, N_VAL, SEED, TRAIN_IMAGES, TRAIN_JSON, VAL_IMAGES, VAL_JSON)


def load_split(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    boxes = {}
    for item in data:
        objs = [(CLASSES[l["category"]], l["box2d"])
                for l in (item.get("labels") or [])
                if l.get("category") in CLASSES and l.get("box2d")]
        if objs:
            boxes[item["name"]] = objs
    return boxes


# round-robin over classes, rarest first, so bike/bus stay represented
def select(boxes, target, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for name, objs in boxes.items():
        for cid, _ in objs:
            by_class[cid].append(name)
    for c in by_class:
        rng.shuffle(by_class[c])
    order = sorted(by_class, key=lambda c: len(by_class[c]))
    pools = {c: iter(by_class[c]) for c in order}
    selected = set()
    while len(selected) < target:
        added = False
        for c in order:
            for name in pools[c]:
                if name not in selected:
                    selected.add(name)
                    added = True
                    break
            if len(selected) >= target:
                break
        if not added:
            break
    return sorted(selected)


def write_split(dataset_root, names, boxes, image_dir, split):
    img_out = dataset_root / "images" / split
    lbl_out = dataset_root / "labels" / split
    for d in (img_out, lbl_out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    counts = Counter()
    for name in names:
        src = image_dir / name
        if not src.exists():
            continue
        lines = []
        for cid, b in boxes[name]:
            x1, x2 = max(0, min(b["x1"], IMAGE_W)), max(0, min(b["x2"], IMAGE_W))
            y1, y2 = max(0, min(b["y1"], IMAGE_H)), max(0, min(b["y2"], IMAGE_H))
            lines.append(f"{cid} {(x1+x2)/2/IMAGE_W:.6f} {(y1+y2)/2/IMAGE_H:.6f} "
                         f"{(x2-x1)/IMAGE_W:.6f} {(y2-y1)/IMAGE_H:.6f}")
            counts[ID_TO_CLASS[cid]] += 1
        shutil.copy2(src, img_out / name)
        (lbl_out / f"{Path(name).stem}.txt").write_text("\n".join(lines))
    return counts


def main():
    dataset_root = DATASETS_ROOT / DATASET_NAME
    dataset_root.mkdir(parents=True, exist_ok=True)
    print(f"building {dataset_root} ({N_TRAIN} train / {N_VAL} val)")

    train_boxes = load_split(TRAIN_JSON)
    val_boxes = load_split(VAL_JSON)
    print(f"images with target classes -> train: {len(train_boxes)}, val: {len(val_boxes)}")

    train_counts = write_split(dataset_root, select(train_boxes, N_TRAIN, SEED),
                               train_boxes, TRAIN_IMAGES, "train")
    val_counts = write_split(dataset_root, select(val_boxes, N_VAL, SEED),
                             val_boxes, VAL_IMAGES, "val")

    names_block = "\n".join(f"  {i}: {ID_TO_CLASS[i]}" for i in sorted(ID_TO_CLASS))
    (dataset_root / "data.yaml").write_text(
        f"path: {dataset_root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names_block}\n"
    )

    print("train images:", len(list((dataset_root / "images/train").glob("*.jpg"))), dict(train_counts))
    print("val images:", len(list((dataset_root / "images/val").glob("*.jpg"))), dict(val_counts))
    print("wrote", dataset_root / "data.yaml")


if __name__ == "__main__":
    main()
