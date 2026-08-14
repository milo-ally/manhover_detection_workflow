# 井盖五分类检测完整工作流

本仓库用于完成数据准备、YOLO11s 训练、ONNX 导出、AX650N 模型转换、仿真和精度验收。

```text
类别: good、broke、lose、uncovered、circle
训练输入: 1 x 3 x 640 x 640, FP32, RGB / 255
ONNX 输出: output0, 1 x 9 x 8400, FP32
AXModel 输入: images, 1 x 640 x 640 x 3, U8, NHWC, RGB
AXModel 输出: output0, 1 x 9 x 8400, FP32
目标硬件: AX650N
```

所有工具脚本按当前工作区硬编码了路径：

```text
/home/milo/workspace/LYG_manhover_detection_workflow
```

仓库移动到其他位置后，先修改 `model_train/1-7` 和 `model_export/1-2` 中的绝对路径。

## 1. 模型训练

### 1.1 安装环境

建议使用 Python 3.10 和独立 Conda 环境：

```bash
conda create -n lyg_manhover_detection_train_export python=3.10 -y
conda activate lyg_manhover_detection_train_export

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics roboflow PyYAML
```

检查训练环境：

```bash
cd /home/milo/workspace/LYG_manhover_detection_workflow/model_train
python 3-check_environment.py
```

输出应确认 `torch.cuda.is_available()` 为 `True`，并能识别 NVIDIA GPU。

### 1.2 下载和转换数据集

数据源：<https://universe.roboflow.com/liujunxiang/manhole-cover-zsmly/dataset/3>

```bash
python 1-download_dataset.py
python 2-data-preprocessing.py
python 4-check-dataset.py
```

脚本完成以下工作：

1. 下载 Roboflow version 3 的 COCO 数据集。
2. 转换成 Ultralytics YOLO 检测标签。
3. 固定类别顺序为 `good、broke、lose、uncovered、circle`。
4. 生成 `Manhole-Cover-5Class-3/data.yaml`。
5. 检查图片、标签、类别编号、归一化坐标和越界框。

数据目录应为：

```text
model_train/Manhole-Cover-5Class-3/
├── data.yaml
├── train/images + train/labels
├── valid/images + valid/labels
└── test/images  + test/labels
```

边界框轻微越过图片边界需要人工确认；类别越界、非数字、负宽高和归一化值超过 `[0,1]` 必须修复后再训练。

### 1.3 训练

```bash
python 5-train.py
```

当前训练方案为 YOLO11s、`640 x 640`、最多 300 epochs、AdamW、初始学习率 `0.0003`、batch 8，并启用早停、AMP 和周期权重保存。

主要产物：

```text
model_train/runs/manhole-cover-yolo11s-production/weights/best.pt
model_train/runs/manhole-cover-yolo11s-production/weights/last.pt
model_train/runs/manhole-cover-yolo11s-production/weights/epoch*.pt
```

`best.pt` 用于导出；`last.pt` 和 `epoch*.pt` 用于断点续训、回滚和对比。

### 1.4 PyTorch 模型评估与测试

```bash
python 6-evaluate.py
python 7-test.py
```

结果保存到：

```text
model_train/runs/eval-val/
model_train/runs/test-predict/
```

必须记录 `best.pt` 的 precision、recall、mAP50 和 mAP50-95，作为 ONNX 与 AXModel 的精度基线。

## 2. 导出 ONNX

### 2.1 安装导出依赖

```bash
conda activate lyg_manhover_detection_train_export
pip install "onnx>=1.12,<2" "onnxruntime-gpu>=1.17,<2" "onnxslim>=0.1.82"
```

### 2.2 导出并验证

```bash
cd /home/milo/workspace/LYG_manhover_detection_workflow/model_export
python 1-export.py
python 2-verify.py
```

导出参数固定为 batch 1、640、FP32、opset 12、静态输入、不包含 NMS。模型保存到：

```text
model_export/model/manhole-cover-yolo11s-production.onnx
```

验证结果保存到：

```text
model_export/runs/onnx-val/
```

ONNX 的 mAP50 和 mAP50-95 应与 `best.pt` 基本一致，建议绝对下降不超过 `0.01`。

## 3. 转换 AXModel

### 3.1 准备 Pulsar2

Pulsar2 4.0：<https://huggingface.co/AXERA-TECH/Pulsar2/resolve/main/4.0/ax_pulsar2_4.0.tar.gz>

```bash
cd /home/milo/workspace/LYG_manhover_detection_workflow/model_convert
docker load -i ax_pulsar2_4.0.tar.gz
docker image inspect pulsar2:4.0 --format '{{.RepoTags}} {{.Id}}'
```

复制待转换模型：

```bash
cp ../model_export/model/manhole-cover-yolo11s-production.onnx \
  models/manhole-cover-yolo11s-production.onnx
```

启动容器，当前 `model_convert` 目录会挂载到 `/workflow`：

```bash
docker run -it --net host --rm \
  -v "$(pwd):/workflow" -w /workflow pulsar2:4.0
```

进入容器后确认版本：

```bash
python --version
pulsar2 version
```

### 3.2 准备量化图片

将具有代表性的真实业务图片放入 `model_convert/dataset/calib_images/`。量化图片不能使用最终验证集，且应覆盖五个类别、光照、距离、角度和背景变化。

```bash
tar -cf dataset/manhover.tar -C dataset/calib_images .
tar -tf dataset/manhover.tar
```

当前示例使用 72 张量化图片，因此后续显式指定 `calibration_size=72`。图片数量变化后必须同步修改该参数。

### 3.3 检查 ONNX

以下命令在 Pulsar2 容器中执行：

```bash
python show.py \
  --onnx_model models/manhole-cover-yolo11s-production.onnx \
  --format json \
  --output models/manhole-cover-yolo11s-production.onnx.json \
  --check
```

控制台应输出 `ONNX checker: PASS`，生成的 JSON 报告应包含：

```text
ONNX checker: PASS
input : images  FLOAT [1, 3, 640, 640]
output: output0 FLOAT [1, 9, 8400]
opset : 12
```

### 3.4 生成 Pulsar2 配置

```bash
python build.py \
  --onnx_model models/manhole-cover-yolo11s-production.onnx \
  --output_config config/manhole-cover-yolo11s-production.onnx.build_config.json \
  --calibration_dataset ./dataset/manhover.tar \
  --calibration_size 72 \
  --npu_mode NPU1 \
  --overwrite
```

配置显式使用：

```text
预处理: U8 NHWC RGB -> RGB / 255
量化方法: MinMax
网络精度: U16
精度分析: EndToEnd
输出 Tensor: output0
NPU 模式: NPU1
```

### 3.5 编译 AXModel

```bash
mkdir -p output/manhole-cover-yolo11s-production

pulsar2 build \
  --config config/manhole-cover-yolo11s-production.onnx.build_config.json \
  --input models/manhole-cover-yolo11s-production.onnx \
  --output_dir output/manhole-cover-yolo11s-production \
  --output_name manhole-cover-yolo11s-production.axmodel \
  --target_hardware AX650 \
  --npu_mode NPU1 \
  --compiler.check 3 \
  --compiler.check_mode CheckOutput \
  --compiler.check_cosine_simularity 0.999
```

成功产物：

```text
model_convert/output/manhole-cover-yolo11s-production/manhole-cover-yolo11s-production.axmodel
```

编译必须成功，输出不得包含 NaN/Inf，编译器输出余弦相似度不得低于 `0.999`。

### 3.6 关于模型裁剪

正常转换成功时不运行 `cut.py`。它只用于定位 Pulsar2 不支持的尾部算子或导出中间 Tensor。

不得只裁剪到 `/model.23/Concat_output_0`，该 Tensor 的形状是 `[1,64,8400]`，只有框回归分支，没有五分类分数。任何裁剪方案都必须同时保留完整框数据和类别数据，并同步修改量化配置、输出解码和验证程序。

## 4. Pulsar2 容器仿真

以下命令仍在 `/workflow` 中执行：

```bash
cp output/manhole-cover-yolo11s-production/manhole-cover-yolo11s-production.axmodel \
  pulsar2_sim/models/manhole-cover-yolo11s-production.axmodel

cd pulsar2_sim
mkdir -p sim_inputs/1 sim_outputs
echo 1 > list.txt

python cli_detect_manhover.py \
  --pre_processing \
  --image_path sim_images/1.jpg \
  --axmodel_path models/manhole-cover-yolo11s-production.axmodel \
  --intermediate_path sim_inputs/1

pulsar2 run \
  --model models/manhole-cover-yolo11s-production.axmodel \
  --input_dir sim_inputs \
  --output_dir sim_outputs \
  --list list.txt

python cli_detect_manhover.py \
  --post_processing \
  --image_path sim_images/1.jpg \
  --axmodel_path models/manhole-cover-yolo11s-production.axmodel \
  --intermediate_path sim_outputs/1 \
  --output_image 1_result.jpg \
  --conf_thres 0.25 \
  --iou_thres 0.45
```

检查 `1_result.jpg` 中类别、置信度和检测框是否合理。单张图片仿真只验证转换链和后处理可以运行，不能代替完整精度验证。

## 5. ONNX 与 AXModel 同口径验证

### 5.1 准备独立验证集

验证集不能参与训练和量化。图片与 YOLO 标签必须同名：

```text
model_val/
├── images/*.jpg
├── labels/*.txt
└── model/
    ├── manhole-cover-yolo11s-production.onnx
    └── manhole-cover-yolo11s-production.axmodel
```

可以先使用数据集的 `test` 划分跑通流程：

```bash
cd /home/milo/workspace/LYG_manhover_detection_workflow/model_val
cp -a ../model_train/Manhole-Cover-5Class-3/test/images/. images/
cp -a ../model_train/Manhole-Cover-5Class-3/test/labels/. labels/
```

当前 `test` 划分只有 11 张图片，只能用于冒烟测试。上线验收必须补充更大的独立业务测试集，并保证五个类别和主要困难场景都有足够样本。

复制模型：

```bash
cd /home/milo/workspace/LYG_manhover_detection_workflow/model_val

cp ../model_export/model/manhole-cover-yolo11s-production.onnx \
  model/manhole-cover-yolo11s-production.onnx
cp ../model_convert/output/manhole-cover-yolo11s-production/manhole-cover-yolo11s-production.axmodel \
  model/manhole-cover-yolo11s-production.axmodel
```

两端必须使用完全相同的 `images/`、`labels/`、类别顺序和阈值。

### 5.2 ONNX Runtime 验证

在 x86 主机或训练环境执行：

```bash
pip install -r requirements_onnx.txt

python src_gpu/val_detect_manhover_onnx.py \
  --onnx_model model/manhole-cover-yolo11s-production.onnx \
  --data data_gpu.yaml \
  --provider auto \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --max-det 300 \
  --save-path runs/manhole-cover-yolo11s-production_onnx.txt \
  --prediction-path runs/manhole-cover-yolo11s-production_onnx_predictions.jsonl \
  --image-dir runs/manhole-cover-yolo11s-production_onnx_images
```

使用 `--provider cuda` 强制 CUDA，使用 `--provider cpu --limit 10` 可进行快速冒烟测试。

### 5.3 AX650N 实板验证

该步骤只能在 AX650N 板端执行：

```bash
pip3 install -r requirements_npu.txt
pip3 install ./axengine-0.1.3-py3-none-any.whl

python3 src_npu/val_detect_manhover_npu.py \
  --axmodel model/manhole-cover-yolo11s-production.axmodel \
  --data data_npu.yaml \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --max-det 300 \
  --save-path runs/manhole-cover-yolo11s-production_axmodel.txt \
  --prediction-path runs/manhole-cover-yolo11s-production_axmodel_predictions.jsonl \
  --image-dir runs/manhole-cover-yolo11s-production_axmodel_images
```

终端会逐张显示标注数和检测数；两个 `*_predictions.jsonl` 文件记录详细检测结果，两个 `*_images/` 目录保存完成后处理并绘制检测框的图片，便于定位 ONNX 与 AXModel 的逐图差异。

### 5.4 上线验收标准

1. PyTorch、ONNX 和 AXModel 的类别顺序一致。
2. ONNX checker、Pulsar2 编译和单图仿真全部通过。
3. ONNX 与 AXModel 在同一独立验证集上计算 precision、recall、mAP50 和 mAP50-95。
4. AXModel 相对 ONNX 的 mAP50、mAP50-95 绝对下降建议不超过 `0.01`。
5. 人工检查至少 20 张困难样本，包括小目标、遮挡、强光、阴影、倾斜和远距离目标。
6. 实板连续推理至少 100 次，记录平均延迟、P95 延迟、峰值内存和异常次数。
7. 上线时的 RGB/BGR、NHWC/NCHW、letterbox、归一化、置信度阈值和 NMS 必须与本工作流一致。

## 6. 产物与版本控制

以下内容由 `.gitignore` 排除，不提交到 Git：

```text
下载的数据集和量化图片
PyTorch、ONNX、AXModel 模型文件
训练、导出、转换、仿真和验证结果
Pulsar2 Docker 镜像压缩包
Python 缓存、日志和本地环境
```

仓库只保留可复现流程需要的 Python 脚本、YAML/JSON 配置、依赖清单和文档。
