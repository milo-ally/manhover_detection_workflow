# RK3588 AI 流

本文档在 `FLOW1.md` 的主机 MediaMTX 和输入推流基础上，使用 RK3588 板端
C++/RKNPU2 完成完整的 AI 流。工程与 AX650 板端工程同构：配置驱动
（`config/streams_config.json` + `/dev/shm/ai_config.json` 热更新）、dlopen 插件 ABI
（`libmanhole_plugin.so` + `IAIModel`）、多流管理，`-c/-m offline|stream/-o` 命令行。

```text
主机 test.mp4
    |
    | FFmpeg 推送到主机 554/src_in
    v
主机 MediaMTX:554
    |
    | RTSP/TCP
    v
RK3588 demo（-m stream，OpenCV VideoCapture）
    |
    | BGR 帧 -> AI 插件（RGB letterbox -> RKNPU2 RKNN 推理）
    v
AI_RESULT_T -> CPU 画框（类别、置信度）
    |
    | BGR rawvideo 管道 -> FFmpeg h264_rkmpp
    v
RK3588 本地 MediaMTX:8554/ai_out
    |
    +--> SSH -L 8557 -> 主机 ffplay 查看
    +--> SSH -L 8557 -> 主机 ffmpeg 录制结果文件
```

## 1. 前置条件

主机和 RK3588 通过 SSH 隧道连接，不要求互相 `ping`，也不使用主机局域网 IP。
主机侧的 MediaMTX、输入推流和 SSH 命令按照本目录的 `FLOW1.md` 执行。

本文后续统一使用：

```text
INPUT_URL=rtsp://127.0.0.1:8556/src_in
OUTPUT_URL=rtsp://127.0.0.1:8554/ai_out
```

`8556` 是 SSH `-R` 映射到 RK3588 的主机输入端口，`8554` 是 RK3588 本地
MediaMTX 输出端口。主机通过 SSH `-L` 的 `8557` 查看输出。

## 2. 先跑通 FLOW1 的输入流

在主机上启动 MediaMTX，并确认监听 `554`：

```bash
./mediamtx mediamtx.yml
```

另开主机终端推送测试视频：

```bash
ffmpeg -re -stream_loop -1 -i ./test.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:554/src_in
```

另开主机终端建立双向 SSH 隧道，并保持窗口运行：

```powershell
ssh -R 8556:127.0.0.1:554 -L 8557:127.0.0.1:8554 root@172.19.30.3
```

在 RK3588 上检查隧道输入流：

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=index,codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8556/src_in
```

预期可以看到视频流，例如：

```text
index=0
codec_name=hevc
width=1138
height=720
```

输入流不能读取时，不要继续排查 RKNN。先确认主机 MediaMTX、FFmpeg 推流和
SSH 隧道仍在运行，并确认隧道没有 `remote port forwarding failed`。

## 3. RK3588 环境检查

在 RK3588 上安装运行和构建所需工具：

```bash
sudo apt update
```

```bash
sudo apt install -y ffmpeg cmake build-essential pkg-config
```

检查 FFmpeg 是否包含 Rockchip 编码器：

```bash
ffmpeg -encoders | grep -E 'h264_rkmpp|hevc_rkmpp'
```

本程序使用 `h264_rkmpp` 输出 H.264 RTSP。如果没有该编码器，需要使用板端
已有的 Rockchip multimedia FFmpeg，不能直接使用只包含软件编码器的另一个
FFmpeg 二进制。

检查 RKNPU2 Runtime：

```bash
file rknpu2/lib/librknnrt.so
```

注意：`rknpu2/` 目录不随仓库提交（已被 `.gitignore` 忽略），`CMakeLists.txt` 默认从
`rknpu2/include` 和 `rknpu2/lib` 查找 `rknn_api.h` 与 `librknnrt.so`，编译前必须先按
`manhole_cover_detection/SOP.md` 从官方 `rknn_model_zoo` 把 Runtime 放回本目录。

RK3588 应显示 `ARM aarch64`。模型和 Runtime 必须匹配板端架构。

## 4. 模型要求

当前部署使用 FP RKNN：

```text
models/manhole-cover-yolo11s-production.rknn
```

模型输入和输出约定：

```text
输入：RGB uint8，640x640，letterbox，padding=114
输出：output0 [1,9,8400]
通道 0..3：cx/cy/w/h，坐标位于 640x640 letterbox 图像
通道 4..8：good/broke/lose/uncovered/circle 分数
```

当前 INT8 产物的 `output0` 量化范围会把类别分数压成 0，不能用于本流程。
如果重新转换模型，应在 `model_convert/rk3588` 执行：

```bash
.venv/bin/python scripts/convert_rknn.py --onnx models/manhole-cover-yolo11s-production.onnx --dataset dataset/calibration.txt --output output/manhole-cover-yolo11s-production.rknn --dtype fp
```

然后把 FP 模型复制到部署目录，再复制整个部署目录到 RK3588。

## 5. 程序说明（与 AX650 同构）

### 5.1 架构

```text
src/main.cpp
   -c/-m offline|stream/-o/--mediamtx*/--enable-raw 参数解析，加载 streams_config.json，
   ConfigService + VideoStreamManager

src/manager/config_service.cpp
   监控 /dev/shm/ai_config.json 热更新，解析 streams/models/global_settings 配置

src/manager/video_stream_manager.cpp
   根据配置创建多路 VideoStream（输入来自配置的 input_source）

src/manager/video_stream.cpp
   OpenCV VideoCapture 解码 -> 推理（单模型或 InferenceManager 多模型调度）
   -> OSDRenderer 画框 -> OpenCV MP4（offline）或 FFmpeg h264_rkmpp RTSP（stream）

src/manager/ai_processor.cpp
   dlopen 插件（配置 plugin 字段或默认 ./libmanhole_plugin.so），
   调用 CreateAIModel/Init/Inference/Deinit，阈值透传环境变量

src/manager/inference_engine.cpp / inference_manager.cpp
   单模型引擎 + Single/Parallel/Serial(ROI) 多模型调度（与 ax650 同语义）

src/manager/osd_renderer.cpp / src/osd_renderer_interface.cpp
   OSDRenderer + IOSDRenderer/DefaultOSDRenderer（绘制到 BGR 帧）

include/ai_interface.h
   平台无关插件 ABI：IAIModel / AI_RESULT_T / AI_FRAME_T

plugins/model_manhole_cover.cpp -> libmanhole_plugin.so
   RKNN 推理 + RGB letterbox(114) + output0 解码 + 分类别 NMS + AI_RESULT_T
```

### 5.2 参数全部使用选项名

```text
-c <file>       streams 配置 JSON（默认 config/streams_config.json）
-m <offline|stream>   运行模式
-o <file>       offline 输出 MP4 路径（offline only）
--mediamtx IP:PORT       MediaMTX 端点（兼容字段，RTSP 输出在配置 rtsp 相关字段）
--mediamtx-host H / --mediamtx-port P  兼容字段
--enable-raw            端云比对 raw 流（RK3588 暂不支持，忽略）
-h              帮助
```

输入源不通过位置参数传入，一律在 `streams_config.json` 的 `input_source` 里配置，
可以是本地文件路径或 RTSP URL。`streams_config.json` 字段与 AX650 完全一致。

### 5.3 阈值透传

配置中每路流的 `conf_thres` / `nms_thres`，以及 `models[]` 里的
`conf_threshold` / `nms_threshold`，由 `AIProcessor::applyModelParamsToEnv()` 写入
环境变量 `MANHOLE_CONF_THRESH` / `MANHOLE_NMS_THRESH`（并回退 `MODEL_*`）。
插件 `Init()` 读取这两个环境变量（默认 0.25 / 0.45），因此配置阈值会真正影响
RKNN 后处理，而不是只显示在配置中。

### 5.4 输入预处理必须与验证一致

每一帧进入 RKNPU2 前执行：

1. 按原图比例缩放到 640x640 内。
2. 使用 `(114,114,114)` 填充剩余区域。
3. BGR 转 RGB。
4. 按 RKNN 输入属性选择 NHWC 或 NCHW 排列。
5. 以 `uint8` 送入 RKNN。

检测框从 640x640 letterbox 坐标减去 padding，再除以缩放比例，恢复到原始
视频尺寸。

### 5.5 输出分支

offline 模式：OpenCV `VideoWriter`（`mp4v`）保存 MP4，路径由 `-o` 指定。

stream 模式：程序启动 FFmpeg 子进程，把绘制后的 BGR 原始帧写入 stdin：

```text
ffmpeg -loglevel warning -f rawvideo -pix_fmt bgr24 -s WIDTHxHEIGHT -r FPS -i pipe:0 \
  -an -c:v h264_rkmpp -pix_fmt yuv420p -f rtsp \
  -rtsp_transport tcp rtsp://127.0.0.1:8554/ai_out
```

这里的 `-s`、`-r` 由输入视频实际属性生成。输出 RTSP 地址默认
`rtsp://127.0.0.1:8554/ai_out`，多路时自动加流号，也可在配置 `rtsp_output_url`
覆盖。当前 AI 流默认不转发音频，因为程序只从 OpenCV 读取视频帧。

### 5.6 推理和绘制顺序

每一帧必须遵守：

```text
VideoCapture.read -> AIProcessor.processFrame（插件推理 -> AI_RESULT_T）
    -> 检测框绘制在 BGR 帧上 -> 写入 VideoWriter 或 FFmpeg stdin
```

如果推理正确但查看的流没有检测框，先确认写入输出的是绘制后的帧。

## 6. 编译 RK3588 程序

进入 RK3588 部署目录：

```bash
cd ~/Desktop/manhole_cover_detection
```

如果板端没有 OpenCV 开发头文件，先执行本目录的离线准备脚本：

```bash
./install_opencv.sh
```

清理并配置：

```bash
rm -rf build
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$HOME/opt/opencv-dev/usr/lib/aarch64-linux-gnu/cmake/opencv4"
```

编译并安装：

```bash
cmake --build build -j2
cmake --install build
```

产物：`bin/demo` 和 `bin/libmanhole_plugin.so`。

如果 CMake 因为完整 OpenCV 配置文件引用不存在的可选模块失败，应使用当前
仓库版本的 `CMakeLists.txt`。它只查找程序需要的 `core`、`imgproc` 和
`videoio`，不要求 `videostab` 等无关模块。

## 7. 启动完整 AI 流

确认主机上的 MediaMTX 和 `src_in` 推流仍在运行后，把 `config/streams_config.json`
的 `input_source` 改为：

```text
rtsp://127.0.0.1:8556/src_in
```

进入程序目录并设置运行库：

```bash
cd bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
```

启动在线 AI 流：

```bash
./demo -c ../config/streams_config.json -m stream
```

正常启动时应看到：

```text
[Main] Loading streams from config file: ../config/streams_config.json
[AIProcessor] Loading plugin: ./libmanhole_plugin.so
[AIProcessor] Model initialized successfully
[ManholeCover] Loading model: ...
[ManholeCover] thresholds: conf=0.250 nms=0.450
[VideoStream] Stream 1: output_mode=rtsp command=ffmpeg ... h264_rkmpp ... rtsp://127.0.0.1:8554/ai_out
output decode: floats=... channels=9 anchors=8400 layout=[channels,anchors]
[ManholeCover] detections=... confidence: ...
```

`detections` 大于 0 时，框已经绘制到发送给 FFmpeg 的视频帧上。

## 8. 主机查看 AI 流

主机上使用 `ffplay` 查看带框的 AI 流：

```bash
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8557/ai_out
```

如果主机没有 `ffplay`，可先安装：

```bash
sudo apt install -y ffmpeg
```

无图形界面时检查 AI 输出流：

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=index,codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8557/ai_out
```

预期至少看到：

```text
index=0
codec_name=h264
width=1138
height=720
```

也可以把 AI 流保存为最终结果文件：

```bash
ffmpeg -y -rtsp_transport tcp -i rtsp://127.0.0.1:8557/ai_out -c copy ai_result.mp4
```

按 `Ctrl+C` 停止录制。录制终端停止不会停止 RK3588 推理程序。

## 9. 离线验证

把 `config/streams_config.json` 的 `input_source` 指向本地视频文件：

```bash
cd bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m offline -o /tmp/output_manhole.mp4
```

预期日志：

```text
[Main] Loading streams from config file: ../config/streams_config.json
[ManholeCover] Loading model: ...
[ManholeCover] thresholds: conf=0.250 nms=0.450
[VideoStream] Stream 1: output_mode=file /tmp/output_manhole.mp4
[Main] Offline input completed
```

检查结果文件：

```bash
ffprobe -v error -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 /tmp/output_manhole.mp4
```

## 10. 排错顺序

### 10.1 没有 `output_mode=rtsp`

确认 `-m stream` 模式，并确认配置的 `input_source` 已改为 RTSP URL；确认使用
更新后的 `src/main.cpp` 重新编译出的 `bin/demo`。

### 10.2 FFmpeg 管道立即退出

在 RK3588 检查：

```bash
ffmpeg -encoders | grep h264_rkmpp
```

如果没有输出，当前 FFmpeg 不能执行文档中的硬件编码命令。先恢复板端
Rockchip multimedia FFmpeg，或明确选择板端实际存在的 H.264 编码器后再修改
`src/manager/video_stream.cpp`。

### 10.3 `ai_out` 没有视频

同时检查三处：

```bash
ffprobe -v error -rtsp_transport tcp rtsp://127.0.0.1:8556/src_in
```

```bash
ffprobe -v error -rtsp_transport tcp rtsp://127.0.0.1:8557/ai_out
```

```bash
ps aux | grep demo
```

输入流正常但输出为空时，查看 RK3588 终端中的 FFmpeg warning 和
`ffmpeg RTSP output pipe closed`。

### 10.4 输出有视频但没有框

检查 RK3588 终端：

```text
output decode: ...
[ManholeCover] detections=... confidence: ...
```

如果 `sample_scores` 全部为 0，使用 FP RKNN 替换当前 INT8 模型；如果
`detections` 大于 0 但画面没有框，确认主机查看的是 `ai_out`，不是
`src_in`，并确认程序调用顺序是“推理 -> 绘制 -> 写入 FFmpeg”。

### 10.5 输入流读取失败

确认 SSH 隧道仍在运行，并且 RK3588 输入使用
`rtsp://127.0.0.1:8556/src_in`，输出使用
`rtsp://127.0.0.1:8554/ai_out`；不要使用主机局域网 IP。

### 10.6 找不到插件

默认插件路径是当前目录下的 `./libmanhole_plugin.so`（从 `bin/` 启动）。如果不在
当前目录，在配置的 `plugin` 字段（stream 级或 `models[]` 级）里显式写完整路径。

## 11. 验收标准

完成以下检查后，才算 RK3588 AI 流跑通：

```text
[ ] RK3588 可以 ffprobe 读取 src_in
[ ] demo 成功加载 FP RKNN 和 libmanhole_plugin.so
[ ] 离线模式生成 output_manhole.mp4 且能看到检测框
[ ] 每帧出现 [ManholeCover] detections 日志
[ ] 输出 FFmpeg 使用 h264_rkmpp 且没有立即退出
[ ] MediaMTX 中出现 ai_out
[ ] 主机 ffplay 可以看到 ai_out
[ ] ai_out 画面中出现类别、置信度和检测框
[ ] ffmpeg 可以把 ai_out 保存为 ai_result.mp4
```
