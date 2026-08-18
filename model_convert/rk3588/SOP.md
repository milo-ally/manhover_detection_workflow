# RK3588 模型转换 SOP

## 1. 固定版本

使用 Rockchip 官方仓库：

```text
https://github.com/airockchip/rknn-toolkit2.git
RKNN-Toolkit2: 2.3.2
```

官方模型库也提供 RK3588 的 YOLO11 转换和 C/C++ 示例：

```text
https://github.com/airockchip/rknn_model_zoo.git
```

仓库目录已经加入 `.gitignore`。重新准备环境时执行：

```bash
cd model_convert/rk3588
mkdir -p third_party
git clone --depth 1 --branch master \
  https://github.com/airockchip/rknn-toolkit2.git third_party/rknn-toolkit2
git -C third_party/rknn-toolkit2 rev-parse HEAD
```

本次使用提交：`59a913d172e7f5ff03c9076e2ec7b1b1288ffd08`。

## 2. 安装转换环境

当前主机为 x86_64、Python 3.12。下载官方 wheel：

```bash
cd model_convert/rk3588
python3 -m venv .venv
source .venv/bin/activate

wget -O packages/rknn_toolkit2-2.3.2-cp312-cp312-linux_x86_64.whl \
  https://github.com/airockchip/rknn-toolkit2/raw/master/\
rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

pip install --index-url https://pypi.org/simple \
  -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu \
  --trusted-host download.pytorch.org torch==2.4.0+cpu
pip install --index-url https://pypi.org/simple --no-deps \
  packages/rknn_toolkit2-2.3.2-cp312-cp312-linux_x86_64.whl
```

必须固定以下版本。RKNN-Toolkit2 2.3.2 使用了 `onnx.mapping`，过新的
ONNX 会失败；过新的 protobuf/numpy 也会导致构建失败：

```text
onnx 1.16.1
protobuf 4.25.4
numpy 1.26.4
```

## 3. 准备模型和校准集

转换使用本目录 `models/` 中已经准备好的 ONNX：

```bash
python scripts/inspect_onnx.py models/manhole-cover-yolo11s-production.onnx
python scripts/make_calibration_list.py
```

校准图片存放在本目录 `dataset/calib_images/`，校准列表也写入本目录。
转换前应看到输入 `[1, 3, 640, 640]` 和输出 `[1, 9, 8400]`。

## 4. 转换部署用 FP RKNN

```bash
python scripts/convert_rknn.py \
  --onnx models/manhole-cover-yolo11s-production.onnx \
  --dataset dataset/calibration.txt \
  --output output/manhole-cover-yolo11s-production.rknn \
  --dtype fp
```

本次实际结果：转换成功，生成约 12 MB 的 RKNN 文件。转换使用：

```python
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
            target_platform="rk3588")
rknn.build(do_quantization=False)
```

FP 产物保留类别分数的精度。C/C++ 部署仍应查询输入输出 tensor 属性，并在
`rknn_outputs_get` 中设置 `want_float = 1` 获取浮点输出。

如果显式使用 `--dtype i8`，不要直接用于当前部署；坐标和类别分数共用输出量化
范围，可能出现类别分数全部为 0 的情况。

## 5. 交付到模型验证

```bash
cp output/manhole-cover-yolo11s-production.rknn \
  ../../model_val/rk3588/models/
```

转换主机只能确认 ONNX 已被 RKNN 编译器接受；RKNN 的真实 NPU 输出和精度
必须在 RK3588 板端通过 `model_val/rk3588` 验证，不能用主机上的 ONNX 结果替代。
