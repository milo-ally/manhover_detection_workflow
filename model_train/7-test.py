from ultralytics import YOLO


model = YOLO("/home/milo/workspace/LYG_manhover_detection_workflow/model_train/runs/manhole-cover-yolo11s-production/weights/best.pt")
model.predict(
    source="/home/milo/workspace/LYG_manhover_detection_workflow/model_train/Manhole-Cover-5Class-3/test/images",
    imgsz=640,
    device=0,
    conf=0.25,
    iou=0.7,
    max_det=300,
    save=True,
    save_txt=True,
    save_conf=True,
    project="/home/milo/workspace/LYG_manhover_detection_workflow/model_train/runs",
    name="test-predict",
    exist_ok=True,
)
