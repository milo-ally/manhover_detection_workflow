# AX650N 完整工作流

本文档是仓库根目录下的 AX650N 复现入口，覆盖训练、ONNX 导出、AXModel 转换、仿真、精度验证和板端部署。

## 0. 固定约定

```text
模型名:       manhole-cover-yolo11s-production
类别顺序:     0 good, 1 broke, 2 lose, 3 uncovered, 4 circle
输入尺寸:     640 x 640
ONNX 输入:    images [1,3,640,640] FP32 NCHW RGB / 255
AXModel 输入: images [1,640,640,3] U8 NHWC RGB
输出:         output0 [1,9,8400]
输出解释:     0..3 为 cx/cy/w/h，4..8 为五个类别分数
```

AX650 链路的转换/验证/部署文件统一使用 `yolo11s-manhole-detection`（ONNX 从
`model_export` 复制时重命名，转换产物为 `yolo11s-manhole-detection.axmodel`）；
`manhole-cover-yolo11s-production` 是训练/导出的模型名。

仓库中的训练和导出脚本包含工作区绝对路径 `/home/milo/workspace/LYG_Internship/Code/LYG_manhover_detection_workflow`。仓库移动到其他位置后，需要先修改 `model_train/` 和 `model_export/` 中的路径。

平台目录不要混用：

```text
AX650 转换:   model_convert/ax650/
AX650 验证:   model_val/ax650/
AX650 部署:   model_deployment/ax650/
```

## 1. 训练模型

### 1.1 创建环境

建议使用 Python 3.10、CUDA 12.1 和独立 Conda 环境：

```bash
conda create -n lyg_manhover_detection_train_export python=3.10 -y
conda activate lyg_manhover_detection_train_export
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics roboflow PyYAML
```

检查 GPU：

```bash
cd /home/milo/workspace/LYG_Internship/Code/LYG_manhover_detection_workflow/model_train
python 3-check_environment.py
```

### 1.2 下载和检查数据

数据源：<https://universe.roboflow.com/liujunxiang/manhole-cover-zsmly/dataset/3>

```bash
cd /home/milo/workspace/LYG_Internship/Code/LYG_manhover_detection_workflow/model_train
python 1-download_dataset.py
python 2-data-preprocessing.py
python 4-check-dataset.py
```

预期目录：

```text
model_train/Manhole-Cover-5Class-3/
├── data.yaml
├── train/images + train/labels
├── valid/images + valid/labels
└── test/images  + test/labels
```

类别编号、归一化坐标、空标签和越界框必须在训练前检查通过。

### 1.3 训练、评估和测试

```bash
python 5-train.py
python 6-evaluate.py
python 7-test.py
```

当前训练脚本使用 YOLO11s、640、batch 8、最多 300 epochs，产物为：

```text
model_train/runs/manhole-cover-yolo11s-production/weights/best.pt
model_train/runs/manhole-cover-yolo11s-production/weights/last.pt
```

使用 `best.pt` 继续导出。记录训练验证集的 Precision、Recall、mAP50 和 mAP50-95，作为后续基线。

## 2. 导出和验证 ONNX

在训练环境安装导出依赖：

```bash
conda activate lyg_manhover_detection_train_export
pip install "onnx>=1.12,<2" "onnxruntime-gpu>=1.17,<2" "onnxslim>=0.1.82"
```

执行：

```bash
cd /home/milo/workspace/LYG_Internship/Code/LYG_manhover_detection_workflow/model_export
python 1-export.py
python 2-verify.py
```

产物：

```text
model_export/model/manhole-cover-yolo11s-production.onnx
model_export/runs/onnx-val/
```

导出固定为 batch 1、640、FP32、opset 12、静态输入、不包含 NMS。ONNX checker 和 ONNX 精度验证通过后再进入转换。

## 3. 转换 AXModel

### 3.1 准备 Pulsar2

Pulsar2 4.0 下载地址：<https://huggingface.co/AXERA-TECH/Pulsar2/resolve/main/4.0/ax_pulsar2_4.0.tar.gz>

当前转换工程已放在 `model_convert/ax650/`，其中包含 `show.py`、`build.py`、`cut.py` 和
`models/`（含 `show.py` 生成的 `yolo11s-manhole-detection.onnx.json` 报告）。
`ax_pulsar2_4.0.tar.gz`、`dataset/`、`pulsar2_sim/`、`output/` 均不提交 Git，需按
`model_convert/ax650/SOP.md` 准备。

```bash
cd model_convert/ax650
docker load -i ax_pulsar2_4.0.tar.gz
docker image inspect pulsar2:4.0 --format '{{.RepoTags}} {{.Id}}'
cp ../../model_export/model/manhole-cover-yolo11s-production.onnx models/yolo11s-manhole-detection.onnx
docker run -it --net host --rm \
  -v "$(pwd):/workflow" -w /workflow pulsar2:4.0
```

以下命令均在容器内 `/workflow` 执行：

```bash
python --version
pulsar2 version
```

### 3.2 准备量化集

当前工程已复制 130 张校准图片到 `model_convert/ax650/dataset/calib_images/`（`dataset/`
不提交 Git，需自行放置）。量化集不能使用最终验证集，应覆盖五个类别及主要光照、距离、
角度和背景。

```bash
tar -cf dataset/manhole_cover.tar -C dataset/calib_images .
tar -tf dataset/manhole_cover.tar | grep -Ei '\.(jpg|jpeg|png)$' | wc -l
```

数量应为 `130`。图片数量变化后，`--calibration_size` 必须同步修改。

### 3.3 检查、生成配置并编译

```bash
python show.py \
  --onnx_model models/yolo11s-manhole-detection.onnx \
  --format json \
  --output models/yolo11s-manhole-detection.onnx.json \
  --check

python build.py \
  --onnx_model models/yolo11s-manhole-detection.onnx \
  --output_config config/yolo11s-manhole-detection.onnx.build_config.json \
  --calibration_dataset ./dataset/manhole_cover.tar \
  --calibration_size 130 \
  --npu_mode NPU1 \
  --overwrite

mkdir -p output/yolo11s-manhole-detection
pulsar2 build \
  --config config/yolo11s-manhole-detection.onnx.build_config.json \
  --input models/yolo11s-manhole-detection.onnx \
  --output_dir output/yolo11s-manhole-detection \
  --output_name yolo11s-manhole-detection.axmodel \
  --target_hardware AX650 \
  --npu_mode NPU1 \
  --compiler.check 3 \
  --compiler.check_mode CheckOutput \
  --compiler.check_cosine_simularity 0.999
```

真实产物路径：

```text
model_convert/ax650/output/yolo11s-manhole-detection/
└── yolo11s-manhole-detection.axmodel
```

检查要求：Pulsar2 成功、输出无 NaN/Inf、余弦相似度不低于 `0.999`，并且 `output0 [1,9,8400]` 保持可用。

正常转换成功时不运行 `cut.py`。裁剪只用于定位不支持算子或调试中间 Tensor；裁剪输出必须同时保留框和五分类数据，并重新生成配置、重新编译和重新验证。

## 4. Pulsar2 仿真

`pulsar2_sim/` 不提交 Git（`cli_detect_manhover.py` 与 `sim_images/` 需按
`model_convert/ax650/SOP.md` 恢复）：

```bash
cd /workflow/pulsar2_sim
mkdir -p models sim_inputs/1 sim_outputs
cp ../output/yolo11s-manhole-detection/yolo11s-manhole-detection.axmodel models/
echo 1 > list.txt

python cli_detect_manhover.py \
  --pre_processing \
  --image_path sim_images/1.jpg \
  --axmodel_path models/yolo11s-manhole-detection.axmodel \
  --intermediate_path sim_inputs/1

pulsar2 run \
  --model models/yolo11s-manhole-detection.axmodel \
  --input_dir sim_inputs \
  --output_dir sim_outputs \
  --list list.txt

python cli_detect_manhover.py \
  --post_processing \
  --image_path sim_images/1.jpg \
  --axmodel_path models/yolo11s-manhole-detection.axmodel \
  --intermediate_path sim_outputs/1 \
  --output_image 1_result.jpg \
  --conf_thres 0.25 \
  --iou_thres 0.45
```

仿真只验证模型输入、推理和后处理链路能运行，不能替代 AX650N 实板精度和性能验证。

## 5. AX650N 精度验证

验证工程完全位于 `model_val/ax650/`，本地已有 11 张验证图、标签、ONNX 和 AXModel 副本。两种后端使用同一套 `validation_common.py`、letterbox、解码、分类别 NMS 和 mAP 口径。

### 5.1 ONNX

验证数据和模型目录约定见 `model_val/ax650/README.md`（图片在 `images/val/`、标签在
`labels/val/`、模型在 `model/`，均不提交 Git，需自行放置）。

```bash
cd model_val/ax650
python3 -m venv .venv-onnx
source .venv-onnx/bin/activate
pip install -r requirements_onnx.txt
python src_gpu/val_detect_manhover_onnx.py \
  --onnx_model model/yolo11s-manhole-detection.onnx \
  --data data_gpu.yaml \
  --device cpu \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --max-det 300 \
  --metrics-json runs/yolo11s-manhole-detection_onnx_metrics.json
```

选择推理设备时使用 `--device`：`cpu` 或指定 GPU `cuda0` / `cuda1` / `cuda2` 等（需 `onnxruntime-gpu`）。

当前已实际得到：`mAP50=0.7397`，`mAP50-95=0.3838`（11 张图片、19 个目标）。`conf=0.001` 用于指标验收，人工查看可使用 `conf=0.25`。

### 5.2 AXModel 板端

将 `model_val/ax650/` 完整复制到 AX650N，安装本目录提供的 `axengine-0.1.3-py3-none-any.whl`
（wheel 不提交 Git，需自行放到本目录）和 `requirements_npu.txt`：

```bash
cd model_val/ax650
pip3 install ./axengine-0.1.3-py3-none-any.whl
pip3 install -r requirements_npu.txt
python3 src_npu/val_detect_manhover_npu.py \
  --axmodel model/yolo11s-manhole-detection.axmodel \
  --data data_npu.yaml \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --max-det 300 \
  --metrics-json runs/yolo11s-manhole-detection_axmodel_metrics.json
```

最终比较 ONNX 与 AXModel 的 mAP50、mAP50-95，建议绝对下降不超过 `0.01`；同时检查困难样本、平均/P95 延迟、峰值内存，并连续推理至少 100 次。

## 6. AX650N 部署

### 6.1 最小插件工程

独立的最小工程位于：

```text
model_deployment/ax650/manhole_cover_detection/
```

它包含 `include/`、`plugins/`、`src/`、`msp_sdk/`，构建 `libmanhole_plugin.so` 和 `demo`。
编译环境需要 AArch64 工具链、AX650N SDK、`libax_engine.so`、`libax_sys.so` 和 OpenCV。

```bash
cd model_deployment/ax650/manhole_cover_detection
cmake -S . -B build \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
cmake --build build -j$(nproc)
cmake --install build
```

离线视频验证（把 `config/streams_config.json` 的 `input_source` 指向本地 H.264 文件，
模型放到 `models/yolo11s-manhole-detection.axmodel` 并同步修改 `models[].path`）：

```bash
cd bin
export LD_LIBRARY_PATH="$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH"
./demo -c ../config/streams_config.json -m offline -o /tmp/output_boxed.mp4
```

当前插件按 `output0 [1,9,8400]` 解码，输出 `AI_RESULT_T` 归一化框；`demo` 的主链路
（ConfigService/VideoStreamManager/VideoDemux/IVPS/OSD/VENC/RTP）负责视频读取、
画框和输出。模型文件不提交 Git，需从
`model_convert/ax650/output/yolo11s-manhole-detection/yolo11s-manhole-detection.axmodel`
复制。

### 6.2 在线流部署

在线流式链路（RTSP 输入、AX 视频链路、插件加载、OSD 和 RTP 推流）也由同一个
`model_deployment/ax650/manhole_cover_detection` 工程完成，井盖插件注册、配置和编译
说明见 [model_deployment/ax650/README.md](model_deployment/ax650/README.md)，
基础推流链路见 [model_deployment/ax650/FLOW1.md](model_deployment/ax650/FLOW1.md) 和
[model_deployment/ax650/FLOW2.md](model_deployment/ax650/FLOW2.md)。

当前推流链路使用 SSH 双向端口转发，不要求主机和 AX650 互相 ping 通。
按 [model_deployment/ax650/FLOW1.md](model_deployment/ax650/FLOW1.md) 建立隧道：

```powershell
ssh -R 8556:127.0.0.1:554 -L 8557:127.0.0.1:8554 root@172.19.30.3
```

AX650 端读取隧道输入：

```text
rtsp://127.0.0.1:8556/src_in
```

AX650 本地 MediaMTX 输出到：

```text
rtsp://127.0.0.1:8554/ai_out
```

主机通过 SSH `-L` 映射后的地址查看：

```text
rtsp://127.0.0.1:8557/ai_out
```

先在 AX650 执行 `ffprobe -rtsp_transport tcp` 验证 `8556/src_in`，再启动板端程序。
SSH、主机 MediaMTX、主机推流 FFmpeg 和板端 MediaMTX 窗口都不能关闭。

## 7. 交付检查

```text
[ ] 训练 best.pt、ONNX checker 和 ONNX 指标完成
[ ] AX 校准集为独立的 72 张图片，Pulsar2 配置 calibration_size=72
[ ] AXModel 编译成功，output0 [1,9,8400]，无 NaN/Inf
[ ] Pulsar2 仿真结果框和类别合理
[ ] AX650N 实板 mAP、延迟、内存和连续运行完成
[ ] 最小插件或 device_side 视频链路完成
```
