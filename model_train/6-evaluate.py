from ultralytics import YOLO


model = YOLO("/home/milo/workspace/LYG_manhover_detection_workflow/model_train/runs/manhole-cover-yolo11s-production/weights/best.pt")
model.val(
    data="/home/milo/workspace/LYG_manhover_detection_workflow/model_train/Manhole-Cover-5Class-3/data.yaml",
    split="val",
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    conf=0.001,
    iou=0.7,
    max_det=300,
    plots=True,
    save_json=False,
    project="/home/milo/workspace/LYG_manhover_detection_workflow/model_train/runs",
    name="eval-val",
    exist_ok=True,
)
