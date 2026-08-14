from shutil import copy2
from os import makedirs
from ultralytics import YOLO


def main():
    model = YOLO("../model_train/runs/manhole-cover-yolo11s-production/weights/best.pt")
    exported = model.export(
        format="onnx",
        imgsz=640,
        batch=1,
        device=0,
        half=False,
        dynamic=False,
        simplify=True,
        opset=12,
        nms=False,
        optimize=False,
        int8=False,
        verbose=False,
    )

    makedirs("model", exist_ok=True)
    target = "model/manhole-cover-yolo11s-production.onnx"
    copy2(exported, target)
    print(f"ONNX exported: {target}")


if __name__ == "__main__":
    main()
