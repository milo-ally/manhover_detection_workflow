#!/usr/bin/env python3
"""Pre/post-processing for the five-class manhole-cover detector."""

import argparse
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ["good", "broke", "lose", "uncovered", "circle"]
COLORS = [(46, 204, 113), (52, 73, 235), (0, 165, 255), (0, 0, 255), (255, 191, 0)]
INPUT_SIZE = 640
OUTPUT_SHAPE = (1, 9, 8400)


def letterbox(image, size=INPUT_SIZE):
    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    resized = (int(round(width * gain)), int(round(height * gain)))
    pad_w, pad_h = (size - resized[0]) / 2, (size - resized[1]) / 2
    if (width, height) != resized:
        interpolation = cv2.INTER_LINEAR if gain > 1 else cv2.INTER_AREA
        image = cv2.resize(image, resized, interpolation=interpolation)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, gain, (pad_w, pad_h)


def preprocess(image_path, output_dir):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image, _, _ = letterbox(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = np.expand_dims(np.ascontiguousarray(image, dtype=np.uint8), 0)
    destination = Path(output_dir) / "images.bin"
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensor.tofile(destination)
    print(f"Saved: {destination}")
    print(f"Input: shape={tensor.shape}, dtype={tensor.dtype}, layout=NHWC, color=RGB")


def box_iou(box, boxes):
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(1)
    area1 = np.clip(box[2:] - box[:2], 0, None).prod()
    area2 = np.clip(boxes[:, 2:] - boxes[:, :2], 0, None).prod(1)
    return intersection / (area1 + area2 - intersection + 1e-7)


def nms(boxes, scores, threshold):
    order, keep = scores.argsort()[::-1], []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        order = order[1:][box_iou(boxes[current], boxes[order[1:]]) <= threshold]
    return keep


def find_output(directory):
    files = sorted(Path(directory).rglob("*.bin"))
    expected_bytes = int(np.prod(OUTPUT_SHAPE)) * np.dtype(np.float32).itemsize
    preferred = [path for path in files if "output0" in path.name]
    for path in preferred + files:
        if path.stat().st_size == expected_bytes:
            return path
    details = ", ".join(f"{path.name} ({path.stat().st_size} bytes)" for path in files)
    raise ValueError(f"No output0 tensor matching {OUTPUT_SHAPE}. Found: {details or 'none'}")


def decode(output, image_shape, gain, pad, conf_thres, iou_thres, max_det=300):
    prediction = output[0].T
    if not np.isfinite(prediction).all():
        raise ValueError("output0 contains NaN or Inf")
    class_scores = prediction[:, 4:]
    classes = class_scores.argmax(1)
    scores = class_scores[np.arange(len(prediction)), classes]
    selected = scores >= conf_thres
    prediction, classes, scores = prediction[selected], classes[selected], scores[selected]
    if len(prediction) == 0:
        return np.empty((0, 6), dtype=np.float32)

    xywh, boxes = prediction[:, :4], np.empty_like(prediction[:, :4])
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    keep = []
    for class_id in np.unique(classes):
        indices = np.where(classes == class_id)[0]
        keep.extend(indices[nms(boxes[indices], scores[indices], iou_thres)].tolist())
    keep = sorted(keep, key=lambda index: float(scores[index]), reverse=True)[:max_det]
    boxes = boxes[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / gain
    height, width = image_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return np.column_stack((boxes, scores[keep], classes[keep])).astype(np.float32)


def draw(image, detections):
    for x1, y1, x2, y2, score, class_id in detections:
        class_id = int(class_id)
        color = COLORS[class_id]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(image, p1, p2, color, 2)
        label = f"{CLASS_NAMES[class_id]} {score:.3f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(p1[1], text_h + 6)
        cv2.rectangle(image, (p1[0], top - text_h - 6), (p1[0] + text_w + 6, top), color, -1)
        cv2.putText(image, label, (p1[0] + 3, top - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def postprocess(image_path, output_dir, output_image, conf_thres, iou_thres):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    _, gain, pad = letterbox(image.copy())
    output_path = find_output(output_dir)
    output = np.fromfile(output_path, dtype=np.float32).reshape(OUTPUT_SHAPE)
    detections = decode(output, image.shape[:2], gain, pad, conf_thres, iou_thres)
    destination = Path(output_image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), draw(image, detections)):
        raise RuntimeError(f"Failed to save: {destination}")
    print(f"Loaded: {output_path}")
    print(f"Detections: {len(detections)}")
    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection
        print(f"  {CLASS_NAMES[int(class_id)]} {score:.4f} [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    print(f"Saved: {destination}")


def main():
    parser = argparse.ArgumentParser(description="AX650 simulation I/O for the manhole detector")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pre_processing", action="store_true")
    mode.add_argument("--post_processing", action="store_true")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--axmodel_path", required=True,
                        help="Kept for command compatibility; pulsar2 run executes the model")
    parser.add_argument("--intermediate_path", required=True)
    parser.add_argument("--output_image", default="lyg_detect_out.jpg")
    parser.add_argument("--conf_thres", type=float, default=0.25)
    parser.add_argument("--iou_thres", type=float, default=0.45)
    args = parser.parse_args()
    if args.pre_processing:
        preprocess(args.image_path, args.intermediate_path)
    else:
        postprocess(args.image_path, args.intermediate_path, args.output_image,
                    args.conf_thres, args.iou_thres)


if __name__ == "__main__":
    main()
