from collections import Counter, defaultdict
from pathlib import Path
import yaml


def issue(issues, key, path, msg, max_examples):
    issues[key]["count"] += 1
    if len(issues[key]["examples"]) < max_examples:
        issues[key]["examples"].append((path.name, msg))


def images_in(path, suffixes):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def check_label(path, nc, issues, class_counts, max_examples):
    boxes = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            issue(issues, "bad_label_format", path, f"line {line_no}: {len(parts)} fields", max_examples)
            continue

        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError as exc:
            issue(issues, "bad_label_format", path, f"line {line_no}: {exc}", max_examples)
            continue

        if not 0 <= cls < nc:
            issue(issues, "class_id_out_of_range", path, f"line {line_no}: class={cls}", max_examples)
        else:
            class_counts[cls] += 1

        if any(v < 0 or v > 1 for v in (x, y, w, h)):
            issue(issues, "bbox_value_out_of_0_1", path, f"line {line_no}: {x} {y} {w} {h}", max_examples)
        if w <= 0 or h <= 0:
            issue(issues, "bbox_non_positive_size", path, f"line {line_no}: w={w}, h={h}", max_examples)
        if x - w / 2 < 0 or x + w / 2 > 1:
            issue(issues, "bbox_x_extends_outside_image", path, f"line {line_no}: x={x}, w={w}", max_examples)
        if y - h / 2 < 0 or y + h / 2 > 1:
            issue(issues, "bbox_y_extends_outside_image", path, f"line {line_no}: y={y}, h={h}", max_examples)

        boxes += 1
    return boxes


def main():
    data_yaml = Path("/home/milo/workspace/LYG_manhover_detection_workflow/model_train/Manhole-Cover-5Class-3/data.yaml")
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    max_examples = 3

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data["path"])
    nc, names = int(data["nc"]), data["names"]
    issues = defaultdict(lambda: {"count": 0, "examples": []})
    totals = Counter()
    class_counts = Counter()

    print(f"dataset: {root}")
    print(f"classes: {names}")

    for key in ("train", "val", "test"):
        split = "valid" if key == "val" else key
        images_dir = root / data[key]
        labels_dir = images_dir.parent / "labels"
        images = images_in(images_dir, suffixes)
        labels = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        boxes = 0

        if not images_dir.exists():
            issue(issues, "missing_images_dir", images_dir, "missing images directory", max_examples)
            continue
        if not labels_dir.exists():
            issue(issues, "missing_labels_dir", labels_dir, "missing labels directory", max_examples)

        for image in images:
            if image.stem not in label_stems:
                issue(issues, "missing_label_for_image", image, "no matching label", max_examples)
        for label in labels:
            if label.stem not in image_stems:
                issue(issues, "label_without_image", label, "no matching image", max_examples)
            boxes += check_label(label, nc, issues, class_counts, max_examples)

        totals.update({"images": len(images), "labels": len(labels), "boxes": boxes})
        print(f"{split}: images={len(images)}, labels={len(labels)}, boxes={boxes}")

    print(f"total: images={totals['images']}, labels={totals['labels']}, boxes={totals['boxes']}")
    print("class distribution:")
    for i, name in enumerate(names):
        print(f"  {i} {name}: {class_counts[i]}")

    bad = {k: v for k, v in issues.items() if v["count"]}
    if not bad:
        print("issues: none")
        return

    print("issues:")
    for key, value in sorted(bad.items()):
        print(f"  {key}: {value['count']}")
        for file_name, msg in value["examples"]:
            print(f"    example: {file_name}: {msg}")


if __name__ == "__main__":
    main()
