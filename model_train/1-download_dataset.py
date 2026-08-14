from roboflow import Roboflow


dataset = (
    Roboflow(api_key="tkAUEW15WDChi9VonEuf")
    .workspace("liujunxiang")
    .project("manhole-cover-zsmly")
    .version(3)
    .download(
        "coco",
        location="/home/milo/workspace/LYG_manhover_detection_workflow/model_train/Manhole-Cover-5Class-3",
        overwrite=True,
    )
)

print(f"Dataset downloaded to: {dataset.location}")
