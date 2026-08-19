# RK3588 C++ 部署 SOP（AX650 同构架构）

## 1. 准备文件

确认以下文件存在：

```bash
ls -lh \
  models/manhole-cover-yolo11s-production.rknn \
  rknpu2/include/rknn_api.h \
  rknpu2/lib/librknnrt.so
```

`rknpu2/`（`include/rknn_api.h` + `lib/librknnrt.so`）**已随仓库提交**，`CMakeLists.txt`
默认在 `rknpu2/include` 和 `rknpu2/lib` 查找这两个文件，克隆后无需额外准备即可编译。
如需核对或重新获取（本工程使用的版本来自 Rockchip 官方 `rknn_model_zoo`，提交
`bad6c7334531becaf90a561988519b7bec34d0ab`），可复现流程：

```bash
cd model_deployment/rk3588/manhole_cover_detection
mkdir -p rknpu2/include rknpu2/lib
# ① rknn_api.h（头文件，与架构无关）
curl -L -o rknpu2/include/rknn_api.h \
  "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/bad6c7334531becaf90a561988519b7bec34d0ab/3rdparty/rknpu2/include/rknn_api.h"
# ② librknnrt.so（aarch64 运行库；仓库内路径 3rdparty/rknpu2/Linux/aarch64/librknnrt.so）
curl -L -o rknpu2/lib/librknnrt.so \
  "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/bad6c7334531becaf90a561988519b7bec34d0ab/3rdparty/rknpu2/Linux/aarch64/librknnrt.so"
file rknpu2/lib/librknnrt.so   # 必须显示 ARM aarch64
```

（备选：`git clone https://github.com/airockchip/rknn_model_zoo.git /tmp/rknn_model_zoo`
后 `git checkout bad6c7334531becaf90a561988519b7bec34d0ab`，再复制
`3rdparty/rknpu2/include/rknn_api.h` 与 `3rdparty/rknpu2/Linux/aarch64/librknnrt.so`。）

准备一个板端 OpenCV 可以读取的视频，例如 `input.mp4`。

## 2. 编译

板端先安装编译依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev
pkg-config --modversion opencv4
```

如果 `libopencv-dev` 因板端 `rkmpp` FFmpeg 版本冲突无法安装，运行本目录的安装脚本。脚本只下载并解包开发文件，不会替换系统运行库：

```bash
chmod +x install_opencv.sh
./install_opencv.sh
```

使用脚本输出的实际目录配置 `OpenCV_DIR`，例如：

```bash
cmake -S . -B build \
  -DOpenCV_DIR="$HOME/opt/opencv-dev/usr/lib/aarch64-linux-gnu/cmake/opencv4"
```

该方法只解包 headers 和 CMake 配置，不会替换板端已有的 `rkmpp` FFmpeg 运行库。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
cmake --install build
```

工程只查找实际使用的 OpenCV `core`、`imgproc` 和 `videoio` 模块，不要求完整的 OpenCV 模块集合。如果 OpenCV 是自定义安装，使用：

```bash
cmake -S . -B build \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
```

检查 Runtime 动态库架构：

```bash
file rknpu2/lib/librknnrt.so
```

输出应为 ARM aarch64。若系统 OpenCV 缺少视频编解码支持，CMake 可能成功但 `VideoCapture` 或 `VideoWriter` 会打开失败，需要更换板端 OpenCV 或编码器。

产物：

```text
bin/demo
bin/libmanhole_plugin.so
```

## 3. 运行

### 3.1 离线视频（MP4）

把 `config/streams_config.json` 的 `input_source` 指向本地视频文件：

```bash
cd bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m offline -o /tmp/output_manhole.mp4
```

启动时应打印模型输入输出属性。程序默认使用配置中的阈值：

```text
conf_threshold = 0.25
iou_threshold  = 0.45
```

输出视频包含检测框、类别名和置信度（OpenCV `mp4v` 编码）。推理过程中的实时日志包含
`[ManholeCover]` 检测数量和置信度。

### 3.2 SSH 隧道 RTSP 流

按上级 `FLOW1.md` 建立 SSH 隧道，把 `streams_config.json` 的 `input_source` 改为
`rtsp://127.0.0.1:8556/src_in`：

```bash
cd bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m stream
```

输出使用板端 `h264_rkmpp` 编码器发布到 `rtsp://127.0.0.1:8554/ai_out`
（可在配置 `rtsp_output_url` 覆盖）；主机通过 SSH `-L` 映射后的
`rtsp://127.0.0.1:8557/ai_out` 查看。

### 3.3 配置热更新

`demo` 固定监控 `/dev/shm/ai_config.json`，文件变化时动态下发阈值/模型更新：

```bash
cp config/streams_config.json /dev/shm/ai_config.json
```

## 4. 架构说明（与 AX650 完全同构，仅底层硬件不同）

工程与 `model_deployment/ax650/manhole_cover_detection` 的类名/方法名/配置格式一致：

```text
src/main.cpp
   -c/-m offline|stream/-o/--mediamtx*/--enable-raw 参数解析，加载 streams_config.json，
   ConfigService + VideoStreamManager

src/manager/config_service.cpp
   监控 /dev/shm/ai_config.json 热更新，解析 streams/models/global_settings 配置

src/manager/video_stream_manager.cpp
   根据配置创建多路 VideoStream；OSDAssociatedModel/initializeOSDForAIStream 与 ax650 一致

src/manager/video_stream.cpp
   OpenCV 解码 -> 推理（单模型直连或 InferenceManager 多模型调度）
   -> OSDRenderer 画框 -> MP4 / h264_rkmpp RTSP 输出

src/manager/ai_processor.cpp
   根据配置 plugin 字段或默认 ./libmanhole_plugin.so dlopen 插件，
   调用 CreateAIModel/Init/Inference/Deinit，阈值透传环境变量

src/manager/inference_engine.cpp / inference_manager.cpp
   单模型引擎适配 + Single/Parallel/Serial(ROI) 多模型调度（与 ax650 同语义）

src/manager/osd_renderer.cpp / src/osd_renderer_interface.cpp
   OSDRenderer + IOSDRenderer/DefaultOSDRenderer（AX650 走 IVPS RGN，RK3588 绘制到 BGR 帧）

include/ai_interface.h
   平台无关插件 ABI：IAIModel / AI_RESULT_T / AI_FRAME_T（AX650 用 AX_VIDEO_FRAME_T）

plugins/model_manhole_cover.cpp
   -> libmanhole_plugin.so：RKNN 推理 + letterbox + 解码/NMS + AI_RESULT_T
```

`streams_config.json` 字段与 AX650 完全一致（`output_width/output_height/fps`、
`ai_output_width/ai_output_height/ai_fps`、`enable_raw_stream`、`plugin`、`models[]`
含 `roi_from_previous/independent/params`、`global_settings` 含
`mediamtx_host/mediamtx_port/default_*/enable_raw_stream`）。

后续接入告警、业务回调或替换输入输出模块时，在 `VideoStream::runLoop()` 的
推理返回后使用 `AI_RESULT_T`；先保持离线视频链路跑通，再替换输入输出模块。
