# 使用说明

本工程比较同一独立验证集上的 ONNX 与 AX650 AXModel 精度。两个入口共用 `validation_common.py`，因此 RGB letterbox、`output0` 解码、分类别 NMS、标签匹配和 mAP 口径一致。

```
ONNX 输入    images [1,3,640,640] FP32 RGB / 255
AXModel 输入 images [1,640,640,3] U8 NHWC RGB
共同输出     output0 [1,9,8400] FP32
类别         good、broke、lose、uncovered、circle
```

## 1. 准备文件

使用未参与训练和量化校准的业务验证集，目录约定以 `data_gpu.yaml` / `data_npu.yaml`
为准（`val: images/val`、`labels: labels/val`）：

```
model_val/ax650/
  images/val/00001.jpg
  labels/val/00001.txt
  model/yolo11s-manhole-detection.onnx
  model/yolo11s-manhole-detection.axmodel
```

注意：`images/`、`labels/`、`model/`、`runs/` 以及 `axengine-*.whl` 均不提交 Git
（已被 `.gitignore` 忽略），使用前需要按上述目录约定自行放置：验证图放 `images/val/`、
标签放 `labels/val/`（与图片同名 `.txt`）、模型放 `model/`，`axengine` wheel 放本目录
`./axengine-0.1.3-py3-none-any.whl`。

每行标签为归一化 YOLO 检测格式：

```
class_id x_center y_center width height
```

类别 ID：`0=good`、`1=broke`、`2=lose`、`3=uncovered`、`4=circle`。无目标图片可没有标签文件或使用空文件。

## 2. ONNX 验证

可直接使用 Pulsar2 容器。在宿主机 PowerShell 的 `LYG_workflow_1` 目录执行：

```
docker run -it --net host --rm -v "${PWD}:/workflow" pulsar2:4.0
```

进入容器后执行基础命令（原有逻辑，不生成指标 json）：

```
cd /workflow/model_val/ax650
python src_gpu/val_detect_manhover_onnx.py \
  --onnx_model model/yolo11s-manhole-detection.onnx \
  --data data_gpu.yaml \
  --device cpu \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --max-det 300 \
  --save-path runs/yolo11s-manhole-detection_onnx.txt \
  --prediction-path runs/yolo11s-manhole-detection_onnx_predictions.jsonl \
  --image-dir runs/yolo11s-manhole-detection_onnx_images
```

新增可选参数 `--metrics-json`，用于输出结构化指标 json 文件，不传该参数不会产生额外文件，原有行为完全不变。

带 metrics-json 参数完整运行示例：

```
cd /workflow/model_val/ax650
python src_gpu/val_detect_manhover_onnx.py \
  --onnx_model model/yolo11s-manhole-detection.onnx \
  --data data_gpu.yaml \
  --device cpu \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --max-det 300 \
  --save-path runs/yolo11s-manhole-detection_onnx.txt \
  --prediction-path runs/yolo11s-manhole-detection_onnx_predictions.jsonl \
  --image-dir runs/yolo11s-manhole-detection_onnx_images \
  --metrics-json runs/yolo11s-manhole-detection_onnx_metrics.json
```

选择推理设备时使用 `--device`：`cpu` 或指定 GPU `cuda0` / `cuda1` / `cuda2` 等；例如强制 GPU 0 使用 `--device cuda0`，流程冒烟测试可使用 `--device cpu --limit 10`。

## 3. AXModel 验证

此步骤必须在 AX650N 板端执行，不在 x86 Pulsar2 容器中执行。

基础运行命令：

```
cd /workflow/model_val/ax650
pip3 install ./axengine-0.1.3-py3-none-any.whl
python3 src_npu/val_detect_manhover_npu.py \
  --axmodel model/yolo11s-manhole-detection.axmodel \
  --data data_npu.yaml \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --max-det 300 \
  --save-path runs/yolo11s-manhole-detection_axmodel.txt \
  --prediction-path runs/yolo11s-manhole-detection_axmodel_predictions.jsonl \
  --image-dir runs/yolo11s-manhole-detection_axmodel_images
```

同样支持可选参数 `--metrics-json`，板端完整示例：

```
cd /workflow/model_val/ax650
pip3 install ./axengine-0.1.3-py3-none-any.whl
python3 src_npu/val_detect_manhover_npu.py \
  --axmodel model/yolo11s-manhole-detection.axmodel \
  --data data_npu.yaml \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --max-det 300 \
  --save-path runs/yolo11s-manhole-detection_axmodel.txt \
  --prediction-path runs/yolo11s-manhole-detection_axmodel_predictions.jsonl \
  --image-dir runs/yolo11s-manhole-detection_axmodel_images \
  --metrics-json runs/yolo11s-manhole-detection_axmodel_metrics.json
```

两边必须使用同一份 `images/val/`、`labels/val/` 和默认的 `--conf-thres 0.001 --iou-thres 0.7 --max-det 300`。

验证时终端会逐张输出标注数和检测数，`*_predictions.jsonl` 每行保存一张图片的类别、置信度和 `xyxy` 检测框，`*_images/` 保存完成后处理并绘制检测框的结果图片。

当传入`--metrics-json`，会额外输出 json 文件，内部包含 mAP50、mAP50-95、Precision、Recall 的全局汇总指标。

## 4. 查看后处理结果图片

计算 mAP 时使用 `--conf-thres 0.001`。仅人工查看实际检测效果时，建议使用 `--conf-thres 0.25`，减少低置信度检测框。

## 5. 判定

比较总体和各类别的 `mAP50`、`mAP50-95`。建议 AXModel 的绝对下降不超过 `0.01`，并人工检查至少 20 张困难样本；最终阈值以项目要求为准。板端还需连续推理至少 100 次，记录平均 / P95 延迟、峰值内存和异常次数。

`src_gpu` 和 `src_npu` 仅保留当前 5 类检测模型的验证入口。

### metrics-json 输出格式示例

```
{
  "summary": {
    "images": 8,
    "total_targets": 26,
    "precision": 0.8812,
    "recall": 0.8433,
    "mAP50": 0.8621,
    "mAP50_95": 0.6345
  }
}
```

> 指标来源：脚本解析 txt 报告文件，复用 validation_common 内部已经完成的 mAP 计算逻辑，不会二次重新执行 IoU/AP 运算，保证 json 内指标与 txt 报告数值完全一致。

## 考核使用提示

1. 运行脚本时带上`--metrics-json`参数，分别得到 ONNX 与 AXModel 两份 metrics json 文件。
2. 可以自行编写简短小脚本读取两份 json，直接生成 markdown 对比表格复制粘贴进 model_convert_shturl。
3. txt 评估报告依旧保留，截图放到 evidence 文件夹作为提交证据。
4. validation_common.py 全程未做任何修改，维持仓库原始版本。
