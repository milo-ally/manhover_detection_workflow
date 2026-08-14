import json
from collections import defaultdict
from pathlib import Path


root = Path("/home/milo/workspace/LYG_manhover_detection_workflow/model_train/Manhole-Cover-5Class-3")
names = ["good", "broke", "lose", "uncovered", "circle"]
name_to_id = {name: i for i, name in enumerate(names)}

for split in ("train", "valid", "test"):
    split_dir = root / split
    coco = json.loads((split_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    images = {image["id"]: image for image in coco["images"]}
    category_map = {}
    for category in coco["categories"]:
        name = category["name"].lower()
        name = "broke" if name == "broken" else name
        if name in name_to_id:
            category_map[category["id"]] = name_to_id[name]

    labels = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] not in category_map:
            continue

        image = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        labels[ann["image_id"]].append(
            (
                category_map[ann["category_id"]],
                (x + w / 2) / image["width"],
                (y + h / 2) / image["height"],
                w / image["width"],
                h / image["height"],
            )
        )

    for image in coco["images"]:
        image_path = split_dir / image["file_name"]
        if image_path.exists():
            image_path.rename(images_dir / image["file_name"])

        lines = [
            f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for cls, x, y, w, h in labels[image["id"]]
        ]
        (labels_dir / f"{Path(image['file_name']).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    print(f"{split}: converted {len(coco['images'])} images")

(root / "data.yaml").write_text(
    f"path: {root}\n"
    "train: train/images\n"
    "val: valid/images\n"
    "test: test/images\n\n"
    f"nc: {len(names)}\n"
    f"names: {names}\n",
    encoding="utf-8",
)
print(f"data.yaml written to: {root / 'data.yaml'}")
