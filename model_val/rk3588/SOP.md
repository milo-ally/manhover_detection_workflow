# RK3588 模型验证 SOP

## 1. 准备模型

模型文件不提交 Git（`.gitignore` 忽略 `models/*`），使用前需要把转换产物复制到本目录
`models/`（转换见 `model_convert/rk3588`，验证数据 `images/`、`labels/` 同样不提交）：

```bash
ls -lh models/manhole-cover-yolo11s-production.rknn
```

验证集位于本目录的 `images/` 和 `labels/`，不需要从其他目录加载。

## 2. ONNX 基线

主机安装 `requirements_onnx.txt` 后运行：

```bash
cd model_val/rk3588
python3 -m venv .venv-onnx
source .venv-onnx/bin/activate
pip install -r requirements_onnx.txt
python src/val_detect_manhole_onnx.py \
  --onnx models/manhole-cover-yolo11s-production.onnx \
  --data data_rknn.yaml \
  --device cpu \
  --limit 0
```

选择推理设备时使用 `--device`：`cpu` 或指定 GPU `cuda0` / `cuda1` / `cuda2` 等（需 `onnxruntime-gpu`）。

## 3. RK3588 板端运行

RKNN-Toolkit2 2.3.2 官方仓库中的 Lite wheel 位于：

```text
rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

本次官方 `rknn_model_zoo` 参考仓库提交为：
`bad6c7334531becaf90a561988519b7bec34d0ab`。

根据板端 Python 版本选择 cp310/cp311/cp312 对应 wheel，复制到板端后安装：

```bash
pip3 install rknn_toolkit_lite2-2.3.2-*.whl
pip3 install -r requirements_rknn.txt
```

把整个 `model_val/rk3588` 目录复制到板端，执行：

```bash
cd model_val/rk3588
python3 src/val_detect_manhole_rknn.py \
  --rknn models/manhole-cover-yolo11s-production.rknn \
  --data data_rknn.yaml \
  --conf-thres 0.001 \
  --iou-thres 0.7
```

结果保存到 `runs/`：

```text
manhole-cover-yolo11s-production_rknn.txt
manhole-cover-yolo11s-production_rknn_predictions.jsonl
manhole-cover-yolo11s-production_rknn_images/
```

## 4. 判定

比较 ONNX 与 RKNN 的 `mAP50`、`mAP50-95`。RKNN 量化模型的下降是否可接受，
以项目阈值为准；建议绝对下降不超过 `0.01`，并人工检查困难样本。确认后再进入
`model_deployment/rk3588` 的 C/C++ 推理链路。

注意：主机没有 RK3588 NPU，`rknn.init_runtime(target="rk3588")` 需要连接
真实板端或 ADB 设备；因此本机只能完成 ONNX 检查和 RKNN 编译，不能声称完成
板端 NPU 精度验证。
