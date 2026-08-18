#!/usr/bin/env python3
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="dataset/calib_images")
    parser.add_argument("--output", default="dataset/calibration.txt")
    args = parser.parse_args()
    root = Path(args.images).resolve()
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"no calibration images found: {root}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")
    print(f"wrote {len(images)} images to {output}")


if __name__ == "__main__":
    main()
