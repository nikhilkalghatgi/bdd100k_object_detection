"""
BDD100K dataset analysis script.

Expects the standard BDD100K detection layout:
  archive (1)/
    train/images/*.jpg, train/annotations/bdd100k_labels_images_train.json
    val/images/*.jpg,   val/annotations/bdd100k_labels_images_val.json
    test/*.jpg   (no labels)

Usage:
    python analyze_bdd100k.py --data-dir "archive (1)" --out-dir report
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from PIL import Image

TARGET_CATEGORIES = ["car", "bus", "person", "truck", "bike"]


def load_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_split(name, images_dir, json_path, target_categories, sample_size=500):
    stats = {
        "split": name,
        "num_images_on_disk": 0,
        "num_annotated_images": 0,
        "categories": Counter(),
        "weather": Counter(),
        "scene": Counter(),
        "timeofday": Counter(),
        "occluded": Counter(),
        "truncated": Counter(),
        "box_dims_by_category": defaultdict(lambda: {"width": [], "height": [], "area": []}),
        "image_sizes": Counter(),
        "file_sizes_kb": [],
    }

    if os.path.isdir(images_dir):
        files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        stats["num_images_on_disk"] = len(files)

        sample = random.sample(files, min(sample_size, len(files)))
        for fname in sample:
            fpath = os.path.join(images_dir, fname)
            try:
                stats["file_sizes_kb"].append(os.path.getsize(fpath) / 1024)
                with Image.open(fpath) as img:
                    stats["image_sizes"][img.size] += 1
            except Exception:
                pass

    if json_path and os.path.isfile(json_path):
        data = load_labels(json_path)
        stats["num_annotated_images"] = len(data)

        for item in data:
            attrs = item.get("attributes", {})
            stats["weather"][attrs.get("weather", "unknown")] += 1
            stats["scene"][attrs.get("scene", "unknown")] += 1
            stats["timeofday"][attrs.get("timeofday", "unknown")] += 1

            labels = item.get("labels", []) or []

            for lbl in labels:
                cat = lbl.get("category", "unknown")
                if cat not in target_categories:
                    continue

                stats["categories"][cat] += 1

                lattrs = lbl.get("attributes", {})
                if "occluded" in lattrs:
                    stats["occluded"][lattrs["occluded"]] += 1
                if "truncated" in lattrs:
                    stats["truncated"][lattrs["truncated"]] += 1

                box = lbl.get("box2d")
                if box:
                    w = box["x2"] - box["x1"]
                    h = box["y2"] - box["y1"]
                    dims = stats["box_dims_by_category"][cat]
                    dims["width"].append(w)
                    dims["height"].append(h)
                    dims["area"].append(w * h)

    return stats


def plot_bar(counter, title, out_path, top_n=None, xlabel="", ylabel="Count", horizontal=False):
    items = counter.most_common(top_n)
    if not items:
        return
    labels, values = zip(*items)
    labels = [str(l) for l in labels]

    plt.figure(figsize=(10, 6))
    if horizontal:
        plt.barh(labels, values, color="#4C72B0")
        plt.gca().invert_yaxis()
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)
    else:
        plt.bar(labels, values, color="#4C72B0")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_size_distribution(cat, dims, title_prefix, out_path):
    widths, heights, areas = dims["width"], dims["height"], dims["area"]
    if not widths:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, values, label, color in [
        (axes[0], widths, "Width (px)", "#4C72B0"),
        (axes[1], heights, "Height (px)", "#DD8452"),
        (axes[2], areas, "Area (px^2)", "#55A868"),
    ]:
        ax.hist(values, bins=30, color=color)
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
    fig.suptitle(f"{title_prefix}: {cat} bounding box size distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def summarize_numeric(values):
    if not values:
        return {}
    values = sorted(values)
    n = len(values)
    return {
        "count": n,
        "min": values[0],
        "max": values[-1],
        "mean": sum(values) / n,
        "median": values[n // 2],
    }


def build_markdown_report(all_stats, target_categories, figures_dir, out_path):
    lines = ["# BDD100K Dataset Analysis Report", "",
             f"Filtered to categories: {', '.join(target_categories)}", ""]

    lines.append("## Overview")
    lines.append("")
    lines.append("| Split | Images on disk | Annotated images | Total objects (filtered) |")
    lines.append("|---|---|---|---|")
    for s in all_stats:
        total_objects = sum(s["categories"].values())
        lines.append(
            f"| {s['split']} | {s['num_images_on_disk']} | {s['num_annotated_images']} | {total_objects} |"
        )
    lines.append("")

    for s in all_stats:
        split = s["split"]
        lines.append(f"## {split.upper()} split")
        lines.append("")

        if s["image_sizes"]:
            lines.append("### Image resolutions (sampled)")
            lines.append("")
            lines.append("| Resolution (W x H) | Count |")
            lines.append("|---|---|")
            for (w, h), cnt in s["image_sizes"].most_common():
                lines.append(f"| {w} x {h} | {cnt} |")
            lines.append("")

        fsz = summarize_numeric(s["file_sizes_kb"])
        if fsz:
            lines.append(
                f"File size (KB, sampled): min={fsz['min']:.1f}, max={fsz['max']:.1f}, "
                f"mean={fsz['mean']:.1f}, median={fsz['median']:.1f}"
            )
            lines.append("")

        if s["categories"]:
            lines.append("### Object class distribution (filtered)")
            lines.append("")
            lines.append("| Category | Count | % of filtered objects |")
            lines.append("|---|---|---|")
            total = sum(s["categories"].values())
            for cat, cnt in s["categories"].most_common():
                lines.append(f"| {cat} | {cnt} | {cnt/total*100:.2f}% |")
            lines.append("")
            lines.append(f"![Class distribution]({figures_dir}/{split}_categories.png)")
            lines.append("")

        for attr_name, title in [("weather", "Weather"), ("scene", "Scene"), ("timeofday", "Time of day")]:
            counter = s[attr_name]
            if counter:
                lines.append(f"### {title} distribution")
                lines.append("")
                lines.append(f"| {title} | Count |")
                lines.append("|---|---|")
                for val, cnt in counter.most_common():
                    lines.append(f"| {val} | {cnt} |")
                lines.append("")
                lines.append(f"![{title} distribution]({figures_dir}/{split}_{attr_name}.png)")
                lines.append("")

        if s["occluded"] or s["truncated"]:
            lines.append("### Occlusion / truncation (filtered categories)")
            lines.append("")
            lines.append(f"Occluded: {dict(s['occluded'])}")
            lines.append("")
            lines.append(f"Truncated: {dict(s['truncated'])}")
            lines.append("")

        if s["box_dims_by_category"]:
            lines.append("### Bounding box size analysis by category")
            lines.append("")
            lines.append("| Category | N | Median W | Mean W | Median H | Mean H | Median Area | Mean Area |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for cat in target_categories:
                dims = s["box_dims_by_category"].get(cat)
                if not dims or not dims["width"]:
                    continue
                w = summarize_numeric(dims["width"])
                h = summarize_numeric(dims["height"])
                a = summarize_numeric(dims["area"])
                lines.append(
                    f"| {cat} | {w['count']} | {w['median']:.1f} | {w['mean']:.1f} | "
                    f"{h['median']:.1f} | {h['mean']:.1f} | {a['median']:.0f} | {a['mean']:.0f} |"
                )
            lines.append("")

            for cat in target_categories:
                dims = s["box_dims_by_category"].get(cat)
                if dims and dims["width"]:
                    lines.append(f"![{cat} size distribution]({figures_dir}/{split}_{cat}_size.png)")
                    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Analyze the BDD100K dataset and generate a report.")
    parser.add_argument("--data-dir", default="archive (1)", help="Path to the BDD100K root folder")
    parser.add_argument("--out-dir", default="bdd100k_report", help="Where to write the report and figures")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of images to sample per split for resolution/file-size stats")
    parser.add_argument("--categories", nargs="+", default=TARGET_CATEGORIES, help="Object categories to include in the report")
    args = parser.parse_args()

    random.seed(0)

    figures_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    splits = [
        ("train", os.path.join(args.data_dir, "train", "images"),
         os.path.join(args.data_dir, "train", "annotations", "bdd100k_labels_images_train.json")),
        ("val", os.path.join(args.data_dir, "val", "images"),
         os.path.join(args.data_dir, "val", "annotations", "bdd100k_labels_images_val.json")),
        ("test", args.data_dir + os.sep + "test", None),
    ]

    all_stats = []
    for name, images_dir, json_path in splits:
        print(f"Analyzing {name} split...")
        stats = analyze_split(name, images_dir, json_path, args.categories, sample_size=args.sample_size)
        all_stats.append(stats)

        if stats["categories"]:
            plot_bar(stats["categories"], f"{name}: object class distribution",
                      os.path.join(figures_dir, f"{name}_categories.png"),
                      xlabel="Category", horizontal=True)
        for attr in ("weather", "scene", "timeofday"):
            if stats[attr]:
                plot_bar(stats[attr], f"{name}: {attr} distribution",
                          os.path.join(figures_dir, f"{name}_{attr}.png"), xlabel=attr)
        for cat, dims in stats["box_dims_by_category"].items():
            plot_size_distribution(cat, dims, name, os.path.join(figures_dir, f"{name}_{cat}_size.png"))

    # Save raw stats as JSON (Counters -> dict, tuple keys -> strings)
    def serialize(obj):
        if isinstance(obj, Counter):
            return {str(k): v for k, v in obj.items()}
        if isinstance(obj, defaultdict):
            return {k: v for k, v in obj.items()}
        return obj

    json_out = []
    for s in all_stats:
        s_copy = {k: serialize(v) for k, v in s.items()}
        json_out.append(s_copy)
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, default=str)

    build_markdown_report(all_stats, args.categories, "figures", os.path.join(args.out_dir, "report.md"))
    print(f"\nDone. Report written to {os.path.join(args.out_dir, 'report.md')}")
    print(f"Figures in {figures_dir}")
    print(f"Raw stats in {os.path.join(args.out_dir, 'stats.json')}")


if __name__ == "__main__":
    main()
