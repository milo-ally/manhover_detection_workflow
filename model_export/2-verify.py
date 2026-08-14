from ultralytics import YOLO


def main():
    model = YOLO("/home/milo/workspace/LYG_manhover_detection_workflow/model_export/model/manhole-cover-yolo11s-production.onnx")
    metrics = model.val(
        data="/home/milo/workspace/LYG_manhover_detection_workflow/model_train/Manhole-Cover-5Class-3/data.yaml",
        split="val",
        imgsz=640,
        batch=1,
        device="cpu",
        workers=8,
        conf=0.001,
        iou=0.7,
        max_det=300,
        half=False,
        dnn=False,
        plots=True,
        save_json=False,
        project="/home/milo/workspace/LYG_manhover_detection_workflow/model_export/runs",
        name="onnx-val",
        exist_ok=True,
        verbose=False,
    )

    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"precision: {metrics.box.mp:.4f}")
    print(f"recall: {metrics.box.mr:.4f}")
    print("results: /home/milo/workspace/LYG_manhover_detection_workflow/model_export/runs/onnx-val")


if __name__ == "__main__":
    main()
