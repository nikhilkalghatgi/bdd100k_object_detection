import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO

from config import (CLASSES, CONF, EVAL_EXPERIMENT, EXPERIMENTS_ROOT, ID_TO_CLASS, IOU,
                    MIN_SLICE_IMAGES, VAL_JSON, WORST_N)

ATTRIBUTES = ["weather", "scene", "timeofday"]
SIZE_BUCKETS = [("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, np.inf)]


def size_bucket(area):
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= area < hi:
            return name
    return "large"


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def load_gt(val_dir):
    names = {p.name for p in val_dir.glob("*.jpg")}
    with open(VAL_JSON, encoding="utf-8") as f:
        data = json.load(f)
    gt = {}
    for item in data:
        if item["name"] not in names:
            continue
        objs = []
        for l in item.get("labels") or []:
            if l.get("category") in CLASSES and l.get("box2d"):
                b, a = l["box2d"], l.get("attributes", {})
                objs.append({"cls": CLASSES[l["category"]],
                             "box": [b["x1"], b["y1"], b["x2"], b["y2"]],
                             "occluded": bool(a.get("occluded")),
                             "truncated": bool(a.get("truncated"))})
        gt[item["name"]] = {"objs": objs, "attrs": item.get("attributes", {})}
    return gt


def match(gt_objs, pred_boxes, pred_cls, pred_conf, iou_thr):
    gt_boxes = np.array([o["box"] for o in gt_objs], dtype=float).reshape(-1, 4)
    gt_cls = np.array([o["cls"] for o in gt_objs], dtype=int)
    ious = box_iou(pred_boxes, gt_boxes)

    matched, tps, fps = set(), [], []
    for pi in np.argsort(-pred_conf):
        best, best_gi = -1.0, -1
        for gi in range(len(gt_objs)):
            if gi in matched or gt_cls[gi] != pred_cls[pi]:
                continue
            if ious[pi, gi] > best:
                best, best_gi = ious[pi, gi], gi
        if best >= iou_thr:
            matched.add(best_gi)
            tps.append((int(pi), best_gi))
        else:
            any_iou = ious[pi].max() if len(gt_objs) else 0.0
            if any_iou >= iou_thr:
                gi = int(ious[pi].argmax())
                kind = "duplicate" if gt_cls[gi] == pred_cls[pi] else "class_confusion"
            elif any_iou >= 0.1:
                kind = "localization"
            else:
                kind = "background"
            fps.append((int(pi), kind))
    fns = [gi for gi in range(len(gt_objs)) if gi not in matched]
    return tps, fps, fns


def draw_failures(path, gt_objs, pred_boxes, pred_cls, tps, fps, fns, out_path):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for pi, _ in tps:
        draw.rectangle(pred_boxes[pi].tolist(), outline="lime", width=2)
    for pi, kind in fps:
        draw.rectangle(pred_boxes[pi].tolist(), outline="red", width=2)
        draw.text((pred_boxes[pi][0], max(0, pred_boxes[pi][1] - 11)),
                  f"FP {ID_TO_CLASS[int(pred_cls[pi])]} ({kind})", fill="red")
    for gi in fns:
        draw.rectangle(gt_objs[gi]["box"], outline="yellow", width=2)
        draw.text((gt_objs[gi]["box"][0], max(0, gt_objs[gi]["box"][1] - 11)),
                  f"MISS {ID_TO_CLASS[gt_objs[gi]['cls']]}", fill="yellow")
    image.save(out_path, quality=90)


def bar_plot(labels, values, ylabel, title, out_path):
    plt.figure(figsize=(8, 5))
    plt.bar([str(l) for l in labels], values, color="#4C72B0")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def eval_subset(model, data_cfg, paths, tag, out_dir, imgsz):
    list_path = out_dir / f"_{tag}.txt"
    list_path.write_text("\n".join(paths))
    yaml_path = out_dir / f"_{tag}.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(dict(data_cfg, val=str(list_path)), f)
    return model.val(data=str(yaml_path), split="val", imgsz=imgsz, plots=False, verbose=False,
                     project=str(out_dir), name=f"subset_{tag}", exist_ok=True)


def main():
    exp_dir = EXPERIMENTS_ROOT / EVAL_EXPERIMENT
    weights = exp_dir / "weights/best.pt"
    out_dir = exp_dir / "eval"
    (out_dir / "failures").mkdir(parents=True, exist_ok=True)
    print("experiment:", exp_dir.name)

    train_args = yaml.safe_load((exp_dir / "args.yaml").read_text())
    imgsz = int(train_args.get("imgsz", 640))
    data_yaml = Path(train_args["data"])
    data_cfg = yaml.safe_load(data_yaml.read_text())
    val_dir = Path(data_cfg["path"]) / data_cfg["val"]

    gt = load_gt(val_dir)
    print(f"{len(gt)} val images, imgsz {imgsz}")

    model = YOLO(str(weights))

    # ---- overall + per class ----
    overall = model.val(data=str(data_yaml), split="val", imgsz=imgsz, plots=True,
                        project=str(out_dir), name="overall", exist_ok=True)
    per_class = pd.DataFrame([{
        "class": overall.names[int(idx)],
        "precision": float(overall.box.p[i]),
        "recall": float(overall.box.r[i]),
        "mAP50": float(overall.box.ap50[i]),
        "mAP50-95": float(overall.box.ap[i]),
    } for i, idx in enumerate(overall.box.ap_class_index)])
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    bar_plot(per_class["class"], per_class["mAP50"], "mAP50", "mAP50 by class",
             out_dir / "map50_by_class.png")
    print(per_class)

    # ---- scenario slices ----
    tables = {}
    for attr in ATTRIBUTES:
        buckets = defaultdict(list)
        for name, g in gt.items():
            buckets[g["attrs"].get(attr, "unknown")].append(str(val_dir / name))
        rows = []
        for value, paths in buckets.items():
            if len(paths) < MIN_SLICE_IMAGES:
                continue
            tag = re.sub(r"[^A-Za-z0-9]+", "_", f"{attr}_{value}")
            m = eval_subset(model, data_cfg, paths, tag, out_dir, imgsz)
            rows.append({attr: value, "n_images": len(paths),
                         "precision": float(m.box.mp), "recall": float(m.box.mr),
                         "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)})
        df = pd.DataFrame(rows).sort_values("mAP50", ascending=False)
        df.to_csv(out_dir / f"{attr}_metrics.csv", index=False)
        bar_plot(df[attr], df["mAP50"], "mAP50", f"mAP50 by {attr}", out_dir / f"map50_by_{attr}.png")
        tables[attr] = df
        print(f"\n{attr}:\n{df}")

    # ---- prediction pass: size / occlusion slices + error types ----
    size_gt, size_hit = Counter(), Counter()
    occ_gt, occ_hit = Counter(), Counter()
    fp_kinds, conf_pairs = Counter(), Counter()
    per_image = []

    torch.cuda.empty_cache()
    results = model.predict(source=str(val_dir), stream=True, conf=CONF,
                            imgsz=imgsz, verbose=False)
    for r in results:
        path = Path(r.path)
        objs = gt[path.name]["objs"]
        pb = r.boxes.xyxy.cpu().numpy()
        pc = r.boxes.cls.cpu().numpy().astype(int)
        pconf = r.boxes.conf.cpu().numpy()
        tps, fps, fns = match(objs, pb, pc, pconf, IOU)

        hit = {gi for _, gi in tps}
        for gi, o in enumerate(objs):
            b = o["box"]
            bucket = size_bucket((b[2] - b[0]) * (b[3] - b[1]))
            cls = ID_TO_CLASS[o["cls"]]
            size_gt[(cls, bucket)] += 1
            occ_gt[("occluded" if o["occluded"] else "visible", cls)] += 1
            occ_gt[("truncated" if o["truncated"] else "whole", cls)] += 1
            if gi in hit:
                size_hit[(cls, bucket)] += 1
                occ_hit[("occluded" if o["occluded"] else "visible", cls)] += 1
                occ_hit[("truncated" if o["truncated"] else "whole", cls)] += 1

        for pi, kind in fps:
            fp_kinds[kind] += 1
            if kind == "class_confusion":
                gi = int(box_iou(pb[pi:pi + 1], np.array([o["box"] for o in objs],
                                                         dtype=float).reshape(-1, 4)).argmax())
                conf_pairs[(ID_TO_CLASS[objs[gi]["cls"]], ID_TO_CLASS[int(pc[pi])])] += 1

        per_image.append({"image": path.name, "n_gt": len(objs), "fn": len(fns), "fp": len(fps),
                          "errors": len(fns) + len(fps), "timeofday": gt[path.name]["attrs"].get("timeofday"),
                          "weather": gt[path.name]["attrs"].get("weather"),
                          "_payload": (path, objs, pb, pc, tps, fps, fns)})

    size_rows = []
    for cls in ID_TO_CLASS.values():
        for bucket, _, _ in SIZE_BUCKETS:
            n = size_gt[(cls, bucket)]
            if n:
                size_rows.append({"class": cls, "size": bucket, "n_gt": n,
                                  "recall": size_hit[(cls, bucket)] / n})
    size_df = pd.DataFrame(size_rows)
    totals = size_df.groupby("size", as_index=False).apply(
        lambda g: pd.Series({"class": "ALL", "n_gt": g["n_gt"].sum(),
                             "recall": (g["recall"] * g["n_gt"]).sum() / g["n_gt"].sum()}),
        include_groups=False)
    size_df = pd.concat([totals[["class", "size", "n_gt", "recall"]], size_df], ignore_index=True)
    size_df.to_csv(out_dir / "size_metrics.csv", index=False)
    all_sizes = size_df[size_df["class"] == "ALL"]
    bar_plot(all_sizes["size"], all_sizes["recall"], f"recall @ conf {CONF}",
             "recall by object size", out_dir / "recall_by_size.png")
    print("\nsize:\n", size_df)

    occ_rows = [{"group": grp, "class": cls, "n_gt": n, "recall": occ_hit[(grp, cls)] / n}
                for (grp, cls), n in sorted(occ_gt.items()) if n]
    occ_df = pd.DataFrame(occ_rows)
    occ_df.to_csv(out_dir / "occlusion_metrics.csv", index=False)
    occ_totals = occ_df.groupby("group").apply(
        lambda g: (g["recall"] * g["n_gt"]).sum() / g["n_gt"].sum(), include_groups=False)
    print("\nocclusion:\n", occ_totals)

    err_df = pd.DataFrame(sorted(fp_kinds.items(), key=lambda x: -x[1]), columns=["fp_type", "count"])
    err_df.to_csv(out_dir / "error_types.csv", index=False)
    conf_df = pd.DataFrame([{"gt": g, "predicted": p, "count": c}
                            for (g, p), c in conf_pairs.most_common(10)])
    conf_df.to_csv(out_dir / "class_confusions.csv", index=False)
    print("\nFP types:\n", err_df)

    worst = sorted(per_image, key=lambda x: -x["errors"])[:WORST_N]
    for rank, row in enumerate(worst):
        path, objs, pb, pc, tps, fps, fns = row["_payload"]
        draw_failures(path, objs, pb, pc, tps, fps, fns,
                      out_dir / "failures" / f"{rank:02d}_{row['fn']}fn_{row['fp']}fp_{path.name}")
    worst_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_payload"} for r in worst])
    worst_df.to_csv(out_dir / "worst_images.csv", index=False)

    # ---- report ----
    lines = ["# Evaluation report", "",
             f"Experiment: `{exp_dir.name}` | imgsz {imgsz} | val images {len(gt)}",
             f"Slice metrics below the mAP tables are recall at conf {CONF}, IoU {IOU}.", "",
             "## Overall", "",
             f"- mAP50: {overall.box.map50:.4f}", f"- mAP50-95: {overall.box.map:.4f}",
             f"- precision: {overall.box.mp:.4f}", f"- recall: {overall.box.mr:.4f}", "",
             "## Per class", "", per_class.to_markdown(index=False, floatfmt=".4f"), "",
             "![mAP50 by class](map50_by_class.png)", ""]
    for attr in ATTRIBUTES:
        lines += [f"## By {attr}", "", tables[attr].to_markdown(index=False, floatfmt=".4f"), "",
                  f"![mAP50 by {attr}](map50_by_{attr}.png)", ""]
    lines += ["## By object size", "",
              "Buckets are COCO convention on box area: small <32^2, medium 32^2-96^2, large >96^2 px.", "",
              size_df.to_markdown(index=False, floatfmt=".4f"), "",
              "![recall by size](recall_by_size.png)", "",
              "## By occlusion / truncation", "", occ_df.to_markdown(index=False, floatfmt=".4f"), "",
              "## False positive types", "", err_df.to_markdown(index=False), ""]
    if not conf_df.empty:
        lines += ["Most common class confusions:", "", conf_df.to_markdown(index=False), ""]
    lines += ["## Worst images", "", worst_df.to_markdown(index=False), "",
              "Overlays in `failures/` — green = correct, red = false positive, yellow = missed.", "",
              "Curves and confusion matrix: `overall/`"]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nsaved to", out_dir)


if __name__ == "__main__":
    main()
