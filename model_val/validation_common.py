#!/usr/bin/env python3
"""Dependency-light detection validation shared by ONNX Runtime and AXEngine."""

import time
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


def load_data_config(data_path):
    yaml_path = Path(data_path).resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    root = Path(data.get("path", "."))
    root = root if root.is_absolute() else (yaml_path.parent / root).resolve()
    image_dir = (root / data.get("val", "images")).resolve()
    label_dir = (root / data.get("labels", "labels")).resolve()
    raw_names = data.get("names", [])
    names = ([str(raw_names[key]) for key in sorted(raw_names)]
             if isinstance(raw_names, dict) else [str(name) for name in raw_names])
    if int(data.get("nc", len(names))) != len(names) or not names:
        raise ValueError("data YAML requires one name for every class")
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"invalid dataset directories: {image_dir}, {label_dir}")
    return image_dir, label_dir, names


def resolve_input(shape):
    dims = [int(value) if isinstance(value, (int, np.integer)) and value > 0 else -1
            for value in shape]
    if len(dims) != 4:
        raise ValueError(f"expected 4D input, got {shape}")
    if dims[-1] == 3:
        height, width, layout = dims[1], dims[2], "NHWC"
    elif dims[1] == 3:
        height, width, layout = dims[2], dims[3], "NCHW"
    else:
        raise ValueError(f"cannot determine input layout: {shape}")
    if height <= 0 or width <= 0:
        raise ValueError(f"dynamic image dimensions are unsupported: {shape}")
    return (height, width), layout


def letterbox(image, new_shape):
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = new_shape
    gain = min(dst_h / src_h, dst_w / src_w)
    resized = (int(round(src_w * gain)), int(round(src_h * gain)))
    pad_w, pad_h = (dst_w - resized[0]) / 2, (dst_h - resized[1]) / 2
    if (src_w, src_h) != resized:
        interpolation = cv2.INTER_LINEAR if gain > 1 else cv2.INTER_AREA
        image = cv2.resize(image, resized, interpolation=interpolation)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, gain, (pad_w, pad_h)


def prepare_input(image, input_hw, layout, backend):
    tensor, gain, pad = letterbox(image, input_hw)
    tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    if layout == "NCHW":
        tensor = tensor.transpose(2, 0, 1)
    if backend == "onnx":
        tensor = tensor.astype(np.float32) / 255.0
    elif backend == "axmodel":
        tensor = tensor.astype(np.uint8, copy=False)
    else:
        raise ValueError(f"unsupported backend: {backend}")
    return np.expand_dims(np.ascontiguousarray(tensor), 0), gain, pad


def box_iou(box1, box2):
    if not len(box1) or not len(box2):
        return np.zeros((len(box1), len(box2)), dtype=np.float32)
    top_left = np.maximum(box1[:, None, :2], box2[None, :, :2])
    bottom_right = np.minimum(box1[:, None, 2:], box2[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(2)
    area1 = np.clip(box1[:, 2:] - box1[:, :2], 0, None).prod(1)
    area2 = np.clip(box2[:, 2:] - box2[:, :2], 0, None).prod(1)
    return intersection / (area1[:, None] + area2[None, :] - intersection + 1e-7)


def nms(boxes, scores, threshold):
    order, keep = scores.argsort()[::-1], []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        overlaps = box_iou(boxes[current:current + 1], boxes[order[1:]])[0]
        order = order[1:][overlaps <= threshold]
    return keep


def decode_output(output, image_shape, gain, pad, nc, conf_thres, iou_thres, max_det):
    prediction = np.asarray(output)
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(f"invalid output rank: {output.shape}")
    if prediction.shape[0] == 4 + nc:
        prediction = prediction.T
    elif prediction.shape[1] != 4 + nc:
        raise ValueError(f"output must contain 4 + nc ({4 + nc}) channels: {output.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("model output contains NaN or Inf")
    class_scores = prediction[:, 4:]
    classes = class_scores.argmax(1)
    scores = class_scores[np.arange(len(prediction)), classes]
    selected = scores >= conf_thres
    prediction, classes, scores = prediction[selected], classes[selected], scores[selected]
    if not len(prediction):
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


def load_labels(label_path, image_shape, nc):
    if not label_path.exists():
        return np.empty((0, 5), dtype=np.float32)
    rows = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} requires class x_center y_center width height")
        row = [float(value) for value in values]
        class_id = int(row[0])
        if row[0] != class_id or not 0 <= class_id < nc:
            raise ValueError(f"{label_path}:{line_number} invalid class id: {row[0]}")
        if any(value < 0 or value > 1 for value in row[1:]):
            raise ValueError(f"{label_path}:{line_number} coordinates must be normalized")
        rows.append(row)
    if not rows:
        return np.empty((0, 5), dtype=np.float32)
    labels = np.asarray(rows, dtype=np.float32)
    height, width = image_shape
    xywh, boxes = labels[:, 1:], np.empty_like(labels[:, 1:])
    boxes[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2) * width
    boxes[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2) * height
    boxes[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2) * width
    boxes[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2) * height
    return np.column_stack((labels[:, 0], boxes)).astype(np.float32)


def match_predictions(predictions, targets):
    correct = np.zeros((len(predictions), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(predictions) or not len(targets):
        return correct
    ious = box_iou(targets[:, 1:5], predictions[:, :4])
    same_class = targets[:, 0:1] == predictions[None, :, 5]
    for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
        target_ids, prediction_ids = np.where((ious >= threshold) & same_class)
        pairs = sorted(zip(target_ids.tolist(), prediction_ids.tolist()),
                       key=lambda pair: float(ious[pair[0], pair[1]]), reverse=True)
        used_targets, used_predictions = set(), set()
        for target_id, prediction_id in pairs:
            if target_id in used_targets or prediction_id in used_predictions:
                continue
            used_targets.add(target_id)
            used_predictions.add(prediction_id)
            correct[prediction_id, threshold_index] = True
    return correct


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(np.interp(x, mrec, mpre), x))


def calculate_metrics(stats, names):
    correct = np.concatenate([item[0] for item in stats], 0)
    scores = np.concatenate([item[1] for item in stats], 0)
    pred_classes = np.concatenate([item[2] for item in stats], 0)
    target_classes = np.concatenate([item[3] for item in stats], 0)
    results = []
    for class_id, name in enumerate(names):
        pred_mask = pred_classes == class_id
        target_count = int((target_classes == class_id).sum())
        pred_count = int(pred_mask.sum())
        ap = np.full(len(IOU_THRESHOLDS), np.nan)
        precision = recall = 0.0
        if target_count and pred_count:
            order = np.argsort(-scores[pred_mask])
            class_correct = correct[pred_mask][order]
            tp, fp = class_correct.cumsum(0), (~class_correct).cumsum(0)
            recall_curve = tp / (target_count + 1e-16)
            precision_curve = tp / (tp + fp + 1e-16)
            precision, recall = float(precision_curve[-1, 0]), float(recall_curve[-1, 0])
            ap = np.array([compute_ap(recall_curve[:, index], precision_curve[:, index])
                           for index in range(len(IOU_THRESHOLDS))])
        elif target_count:
            ap.fill(0.0)
        results.append({"id": class_id, "name": name, "targets": target_count,
                        "precision": precision, "recall": recall, "map50": float(ap[0]),
                        "map": float(np.nanmean(ap)) if not np.isnan(ap).all() else float("nan")})
    return results


def format_report(results, image_count, class_images, conf_thres, inference_ms):
    valid = [row for row in results if row["targets"]]
    average = lambda key: float(np.mean([row[key] for row in valid])) if valid else 0.0
    lines = [f"Validation Results ({image_count} images, conf={conf_thres:g}):",
             f"{'Class':<14} {'Images':>7} {'Targets':>8} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}",
             f"{'all':<14} {image_count:>7} {sum(row['targets'] for row in results):>8} "
             f"{average('precision'):>8.4f} {average('recall'):>8.4f} "
             f"{average('map50'):>8.4f} {average('map'):>10.4f}"]
    for row in results:
        map50 = f"{row['map50']:.4f}" if np.isfinite(row["map50"]) else "-"
        map_all = f"{row['map']:.4f}" if np.isfinite(row["map"]) else "-"
        lines.append(f"{row['name']:<14} {int(class_images[row['id']]):>7} {row['targets']:>8} "
                     f"{row['precision']:>8.4f} {row['recall']:>8.4f} {map50:>8} {map_all:>10}")
    lines.append(f"Mean inference time: {inference_ms:.3f} ms/image (model run only)")
    return "\n".join(lines) + "\n"


def run_validation(infer, input_shape, backend, data_path, conf_thres, iou_thres,
                   max_det, save_path, limit=0):
    image_dir, label_dir, names = load_data_config(data_path)
    images = sorted(path for path in image_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    images = images[:limit] if limit > 0 else images
    if not images:
        raise ValueError(f"no validation images found in {image_dir}")
    input_hw, layout = resolve_input(input_shape)
    print(f"Backend: {backend}; input: {input_shape} {layout}; images: {len(images)}")
    stats, class_images, elapsed = [], np.zeros(len(names), dtype=np.int64), 0.0
    for index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"failed to read {image_path}")
        tensor, gain, pad = prepare_input(image, input_hw, layout, backend)
        start = time.perf_counter()
        output = infer(tensor)
        elapsed += time.perf_counter() - start
        predictions = decode_output(output, image.shape[:2], gain, pad, len(names),
                                    conf_thres, iou_thres, max_det)
        label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
        targets = load_labels(label_path, image.shape[:2], len(names))
        if len(targets):
            class_images[np.unique(targets[:, 0]).astype(int)] += 1
        stats.append((match_predictions(predictions, targets),
                      predictions[:, 4] if len(predictions) else np.empty(0),
                      predictions[:, 5] if len(predictions) else np.empty(0),
                      targets[:, 0] if len(targets) else np.empty(0)))
        if index == len(images) or index % 25 == 0:
            print(f"Processed {index}/{len(images)}")
    if sum(len(item[3]) for item in stats) == 0:
        raise ValueError("validation set contains no YOLO detection labels")
    report = format_report(calculate_metrics(stats, names), len(images), class_images,
                           conf_thres, elapsed * 1000 / len(images))
    print(report, end="")
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(f"Saved: {destination.resolve()}")
    return report
