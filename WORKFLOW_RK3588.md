# RK3588 完整工作流

本文档是 RK3588 的独立复现入口，覆盖 ONNX 准备、RKNN 转换、验证和 C/C++ 离线视频部署。

## 0. 固定约定

```text
模型名:       manhole-cover-yolo11s-production
类别顺序:     0 good, 1 broke, 2 lose, 3 uncovered, 4 circle
输入尺寸:     640 x 640
ONNX 输入:    images [1,3,640,640] FP32 NCHW RGB / 255
RKNN 输入:    由 Runtime 查询实际格式，当前程序兼容 NCHW/NHWC
输出:         output0 [1,9,8400]
输出解释:     0..3 为 cx/cy/w/h，4..8 为五个类别分数
```

RK3588 的文件、数据集、模型、Runtime、文档和代码均在各自目录中独立保存：

```text
RKNN 转换:   model_convert/rk3588/
RKNN 验证:   model_val/rk3588/
C/C++ 部署:  model_deployment/rk3588/manhole_cover_detection/
```

不要从其他平台目录加载模型或量化/验证数据。当前 RK3588 目录已经复制了自己的 ONNX、72 张校准图片、11 张验证图片、标签和 RKNN 模型。

## 1. 训练和 ONNX 来源

训练和导出使用仓库根目录的通用流程：

```bash
cd /home/milo/workspace/LYG_Internship/Code/LYG_manhover_detection_workflow/model_train
python 1-download_dataset.py
python 2-data-preprocessing.py
python 4-check-dataset.py
python 5-train.py
python 6-evaluate.py
python 7-test.py

cd ../model_export
python 1-export.py
python 2-verify.py
```

导出产物为：

```text
model_export/model/manhole-cover-yolo11s-production.onnx
```

RK3588 转换目录中的 ONNX 是独立副本：

```text
model_convert/rk3588/models/manhole-cover-yolo11s-production.onnx
```

重新导出模型后，应明确复制新的 ONNX 到 RK3588 的 `models/`，然后重新生成校准列表、转换和验证，不要在脚本中跨目录动态加载。

## 2. RKNN 转换

### 2.1 官方工具和版本

使用 Rockchip 官方仓库：

- RKNN-Toolkit2：<https://github.com/airockchip/rknn-toolkit2.git>
- RKNN Model Zoo：<https://github.com/airockchip/rknn_model_zoo.git>

当前实际版本：

```text
RKNN-Toolkit2: 2.3.2
rknn-toolkit2 commit: 59a913d172e7f5ff03c9076e2ec7b1b1288ffd08
```

依赖仓库放在 `model_convert/rk3588/third_party/`，已被 `.gitignore` 忽略；重新准备时必须按 [model_convert/rk3588/SOP.md](model_convert/rk3588/SOP.md) 记录的仓库和版本执行。

### 2.2 主机环境

当前转换主机为 x86_64、Python 3.12：

```bash
cd model_convert/rk3588
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu \
  --trusted-host download.pytorch.org torch==2.4.0+cpu
pip install --no-deps packages/rknn_toolkit2-2.3.2-cp312-cp312-linux_x86_64.whl
```

关键版本：`onnx==1.16.1`、`protobuf==4.25.4`、`numpy==1.26.4`。

### 2.3 校准集、检查和转换

校准图片已经独立复制到：

```text
model_convert/rk3588/dataset/calib_images/
```

从 `model_convert/rk3588/` 执行：

```bash
.venv/bin/python scripts/make_calibration_list.py
.venv/bin/python scripts/inspect_onnx.py \
  models/manhole-cover-yolo11s-production.onnx
.venv/bin/python scripts/convert_rknn.py \
  --onnx models/manhole-cover-yolo11s-production.onnx \
  --dataset dataset/calibration.txt \
  --output output/manhole-cover-yolo11s-production.rknn \
  --dtype fp
```

校准列表应包含 72 张本地图片，ONNX 检查应显示：

```text
input images [1, 3, 640, 640]
output output0 [1, 9, 8400]
onnx checker: OK
```

真实转换产物：

```text
model_convert/rk3588/output/manhole-cover-yolo11s-production.rknn
```

转换完成后复制到验证目录：

```bash
cp output/manhole-cover-yolo11s-production.rknn \
  ../../model_val/rk3588/models/
```

部署使用 FP RKNN，避免坐标和类别分数共用 INT8 输出量化范围。C/C++ 程序仍必须查询 tensor 属性，并通过 `want_float=1` 获取浮点输出。

## 3. RKNN 精度验证

### 3.1 主机 ONNX 基线

验证数据是 `model_val/rk3588/images/` 和 `model_val/rk3588/labels/` 的本地副本，ONNX 也位于 `model_val/rk3588/models/`（`images/`、`labels/`、`models/` 均不提交 Git，需按 `model_val/rk3588/SOP.md` 放置）。

```bash
cd model_val/rk3588
../../model_convert/rk3588/.venv/bin/python \
  src/val_detect_manhole_onnx.py \
  --onnx models/manhole-cover-yolo11s-production.onnx \
  --data data_rknn.yaml \
  --limit 0
```

当前已实际通过 11 张图片、19 个目标的 ONNX 基线：

```text
mAP50:    0.7397
mAP50-95: 0.3838
```

### 3.2 RKNN Lite 板端验证

板端需要 Python、与 Python 版本匹配的 `rknn_toolkit_lite2` wheel 和 `requirements_rknn.txt`。本工程使用的 Lite wheel 来自官方 Toolkit2 2.3.2，aarch64 文件位于 `model_val/rk3588/packages/`（不提交 Git，需自行放置）。

将整个 `model_val/rk3588/` 复制到 RK3588 板端后执行：

```bash
cd model_val/rk3588
pip3 install rknn_toolkit_lite2-2.3.2-*.whl
pip3 install -r requirements_rknn.txt
python3 src/val_detect_manhole_rknn.py \
  --rknn models/manhole-cover-yolo11s-production.rknn \
  --data data_rknn.yaml \
  --conf-thres 0.001 \
  --iou-thres 0.7
```

结果保存到 `runs/`。将 RKNN 指标与 ONNX 基线比较，建议 mAP50 和 mAP50-95 的绝对下降均不超过 `0.01`，并人工检查困难样本。

主机没有 RK3588 NPU，因此主机只能完成 ONNX 验证和 RKNN 编译；不能把主机结果称为 NPU 精度验证。

## 4. C/C++ 离线视频部署

### 4.1 工程结构

```text
model_deployment/rk3588/manhole_cover_detection/
├── CMakeLists.txt
├── config/streams_config.json         # 字段与 AX650 streams_config.json 完全一致
├── include/
│   ├── ai_interface.h                 # 平台无关插件 ABI（IAIModel / AI_RESULT_T / AI_FRAME_T）
│   ├── rknpu_manhole.hpp
│   ├── osd_renderer_interface.h       # IOSDRenderer / DefaultOSDRenderer
│   └── manager/                       # config_service / ai_processor / video_stream /
│                                      # video_stream_manager / inference_engine /
│                                      # inference_manager / osd_renderer / ai_pipeline_config
├── plugins/model_manhole_cover.cpp    # -> libmanhole_plugin.so（IAIModel 实现，内部 RKNN）
├── src/
│   ├── main.cpp                       # -c/-m offline|stream/-o/--mediamtx*/--enable-raw
│   ├── osd_renderer_interface.cpp
│   └── manager/*.cpp
├── utilities/                         # json.hpp / sample_log.h
├── models/manhole-cover-yolo11s-production.rknn
└── rknpu2/                    # RKNN 运行时（随仓库提交：rknn_api.h + aarch64 librknnrt.so）
    ├── include/rknn_api.h
    └── lib/librknnrt.so
```

`rknpu2/` 已随仓库提交，`CMakeLists.txt` 默认从 `rknpu2/include` 和 `rknpu2/lib` 查找
`rknn_api.h` 与 `librknnrt.so`，克隆后可直接编译。如需核对/重新获取（Rockchip 官方
`rknn_model_zoo` 提交 `bad6c7334531becaf90a561988519b7bec34d0ab`，curl 下载地址见
`model_deployment/rk3588/manhole_cover_detection/SOP.md` §1 与 `readme.txt`）。

该工程与 AX650 板端工程同构（配置驱动 + dlopen 插件 ABI + 多流管理），推理后端为
RKNPU2。当前模型是单输出 `[1,9,8400]`，因此插件后处理按当前五分类输出实现，
不能直接套用三分支 DFL 后处理。

### 4.2 编译

编译环境需要 AArch64 C++ 编译器、CMake、OpenCV `core/imgproc/videoio` 和 RKNN Runtime：

```bash
cd model_deployment/rk3588/manhole_cover_detection
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
cmake --install build
file rknpu2/lib/librknnrt.so
```

Runtime 应为 ARM aarch64。若使用交叉编译器，可通过 CMake toolchain 文件指定编译器和 sysroot。
产物：`bin/demo` 和 `bin/libmanhole_plugin.so`。

### 4.3 离线视频推理和保存

把 `config/streams_config.json` 的 `input_source` 指向本地视频后：

```bash
cd bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m offline -o /tmp/output_manhole.mp4
```

程序逐帧读取视频，插件完成 RGB letterbox、RKNN 推理、解码/NMS，CPU 绘制检测框并
保存输出视频（OpenCV `mp4v`）。阈值由配置 `conf_thres/nms_thres`（或 `models[]` 的
`conf_threshold/nms_threshold`）下发。

后续接入告警、推流或插件时，在 `VideoStream::runLoop()` 的 `processFrame` 返回后
使用 `AI_RESULT_T`；先保持离线视频链路稳定，再替换输入输出模块。

## 5. SSH-TUNNEL 实时推流部署

实时推流按照 [model_deployment/rk3588/FLOW1.md](model_deployment/rk3588/FLOW1.md)
和 [model_deployment/rk3588/FLOW2.md](model_deployment/rk3588/FLOW2.md) 执行。
当前链路不使用局域网直连，也不要求主机和 RK3588 互相 ping 通。

### 5.1 端口约定

```text
主机 MediaMTX 输入: 554
RK3588 隧道输入:    127.0.0.1:8556/src_in
RK3588 输出:        127.0.0.1:8554/ai_out
主机查看输出:       127.0.0.1:8557/ai_out
```

### 5.2 启动顺序

主机启动 MediaMTX，配置 `protocols: [tcp]`、`rtspAddress: :554`，再推送测试视频：

```powershell
ffmpeg -re -stream_loop -1 -i .\test.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:554/src_in
```

新开 PowerShell 窗口建立并保持 SSH 隧道：

```powershell
ssh -R 8556:127.0.0.1:554 -L 8557:127.0.0.1:8554 root@172.19.30.3
```

RK3588 启动本地 MediaMTX 后，先验证输入：

```bash
ffprobe -v error -rtsp_transport tcp rtsp://127.0.0.1:8556/src_in
```

### 5.3 启动 RK3588 AI 流

把 `config/streams_config.json` 的 `input_source` 改为 `rtsp://127.0.0.1:8556/src_in`
后：

```bash
cd bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m stream
```

在线输出使用 RK3588 板端 `h264_rkmpp` 编码器，默认发布到
`rtsp://127.0.0.1:8554/ai_out`（可在配置 `rtsp_output_url` 覆盖）。主机查看或录制
必须使用 SSH 映射端口 `8557`：

```bash
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8557/ai_out
ffmpeg -y -rtsp_transport tcp -i rtsp://127.0.0.1:8557/ai_out -c copy ai_result.mp4
```

隧道窗口关闭、主机 MediaMTX 停止或板端 MediaMTX 停止，相关 RTSP 端口都会失效。

## 6. 交付检查

```text
[ ] RK3588 ONNX、校准集、验证集和模型均为本目录独立副本
[ ] 校准列表为本地 72 张图片
[ ] RKNN Toolkit2 转换成功，output0 [1,9,8400]
[ ] 主机 ONNX 基线完成
[ ] RK3588 板端 RKNN Lite 精度验证完成
[ ] C++ 程序在板端成功编译
[ ] 离线输入视频成功生成带框输出视频
[ ] SSH 隧道输入 `8556/src_in` 可被 RK3588 读取
[ ] 主机通过 `8557/ai_out` 可查看和录制 AI 输出
[ ] 记录平均/P95 推理耗时、内存和连续运行稳定性
```
