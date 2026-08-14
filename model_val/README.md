# model_val 使用说明

本工程比较同一独立验证集上的 ONNX 与 AX650 AXModel 精度。两个入口共用 `validation_common.py`，因此 RGB letterbox、`output0` 解码、分类别 NMS、标签匹配和 mAP 口径一致。

```text
ONNX 输入    images [1,3,640,640] FP32 RGB / 255
AXModel 输入 images [1,640,640,3] U8 NHWC RGB
共同输出     output0 [1,9,8400] FP32
类别         good、broke、lose、uncovered、circle
```

## 1. 准备文件

使用未参与训练和量化校准的业务验证集：

```text
model_val/
  images/00001.jpg
  labels/00001.txt
  model/manhole-cover-yolo11s-production.onnx
  model/manhole-cover-yolo11s-production.axmodel
```

每行标签为归一化 YOLO 检测格式：

```text
class_id x_center y_center width height
```

类别 ID：`0=good`、`1=broke`、`2=lose`、`3=uncovered`、`4=circle`。无目标图片可没有标签文件或使用空文件。

## 2. ONNX 验证

可直接使用 Pulsar2 容器。在宿主机 PowerShell 的 `LYG_workflow_1` 目录执行：

```powershell
docker run -it --net host --rm -v "${PWD}:/workflow" pulsar2:4.0
```

进入容器后：

```bash
cd /workflow/model_val
python src_gpu/val_detect_manhover_onnx.py \
  --onnx_model model/manhole-cover-yolo11s-production.onnx \
  --data data_gpu.yaml --provider auto \
  --save-path runs/manhole-cover-yolo11s-production_onnx.txt
```

强制 GPU 时使用 `--provider cuda`；流程冒烟测试可使用 `--provider cpu --limit 10`。

## 3. AXModel 验证

此步骤必须在 AX650N 板端执行，不在 x86 Pulsar2 容器中执行：

```bash
cd /workflow/model_val
pip3 install ./axengine-0.1.3-py3-none-any.whl
python3 src_npu/val_detect_manhover_npu.py \
  --axmodel model/manhole-cover-yolo11s-production.axmodel \
  --data data_npu.yaml \
  --save-path runs/manhole-cover-yolo11s-production_axmodel.txt
```

两边必须使用同一份 `images/`、`labels/` 和默认的 `--conf-thres 0.001 --iou-thres 0.7 --max-det 300`。

## 4. 判定

比较总体和各类别的 `mAP50`、`mAP50-95`。建议 AXModel 的绝对下降不超过 `0.01`，并人工检查至少 20 张困难样本；最终阈值以项目要求为准。板端还需连续推理至少 100 次，记录平均/P95 延迟、峰值内存和异常次数。

`src_gpu` 和 `src_npu` 仅保留当前 5 类检测模型的验证入口。
