# RK3588 AI 流

本文档在 `FLOW1.md` 的主机 MediaMTX 和输入推流基础上，使用 RK3588 板端
C++/RKNPU2 完成完整的 AI 流。工程与 AX650 板端工程同构：配置驱动
（`config/streams_config.json` + `/dev/shm/ai_config.json` 热更新）、dlopen 插件 ABI
（`libmanhole_plugin.so` + `IAIModel`）、多流管理，`-c/-m offline|stream/-o` 命令行。
仅底层硬件不同：RK3588 用 MPP（解码/编码）+ RGA/OpenCV（缩放/格式转换）+ RKNPU2。

```text
主机 test.mp4
    |
    | FFmpeg 推送到主机 554/src_in（或 RK3588 直接用本地文件）
    v
输入源（RTSP/本地文件）
    |
    | H264Demux（FFmpeg libavformat，h264_mp4toannexb）
    v
RK3588 demo（-m stream）
    ├── 主码流：MPP 解码 -> RGA(OpenCV) 缩放/格式转换 -> CPU 画框
    │              -> MPP 编码(H.264) -> ffmpeg -c copy -> RTSP 推流
    └── AI 流：FrameBroker 取帧 -> RGA 640 -> RKNN 推理 -> SharedAIResult
            （画框读取 AI 结果叠加到主码流帧上）
    |
    | RTSP（demo 作为 publisher）
    v
RK3588 本地 MediaMTX（rtspAddress :8554）→ ai_out
    |
    +--> SSH -L 8557 -> 主机 ffplay 查看
    +--> SSH -L 8557 -> 主机 ffmpeg 录制结果文件
```

在线输出走 `MPP 编码 -> ffmpeg -c copy -> RTSP -> MediaMTX`（demo 作为 RTSP
publisher）。保留 MPP 硬件编码，只把传输层从 `rtp_pusher(RTP/UDP)` 换成 RTSP——
因为 v1.10.0 MediaMTX 收不了 `rtp_pusher` 的裸 RTP。不要把旧文档中的 `h264_rkmpp`
命令或 `udp+rtp://` 配置复制到这里。

## 1. 前置条件

主机和 RK3588 通过 SSH 隧道连接，不要求互相 `ping`，也不使用主机局域网 IP。
主机侧的 MediaMTX、输入推流和 SSH 命令按照本目录的 `FLOW1.md` 执行。

本文后续统一使用：

```text
INPUT_URL=rtsp://127.0.0.1:8556/src_in
OUTPUT_URL=rtsp://127.0.0.1:8554/ai_out
```

`8556` 是 SSH `-R` 映射到 RK3588 的主机输入端口，`8554` 是 RK3588 本地
MediaMTX RTSP 输出端口。主机通过 SSH `-L` 的 `8557` 查看输出。

### 1.1 在线推流最小步骤（本地文件输入，不依赖 SSH 隧道）

如果只想快速验证推流（本地 `test.mp4` 作输入、输出 `ai_out`），可以完全在
板端本地完成，不需要主机 MediaMTX / SSH 隧道：

```bash
# ① 下载 aarch64 版 MediaMTX（仓库里的 mediamtx 是 x86-64，仅主机可用）
wget https://github.com/bluenviron/mediamtx/releases/download/v1.10.0/mediamtx_v1.10.0_linux_arm64v8.tar.gz
tar -xvzf mediamtx_v1.10.0_linux_arm64v8.tar.gz
file ./mediamtx     # 必须显示 ARM aarch64
```

```bash
# ② 配置板端 mediamtx.yml：作为 RTSP publisher 接收本工程推流。
#    本工程 stream 模式 = MPP 编码 + ffmpeg -c copy 推 RTSP（不是裸 RTP/UDP）。
#    注意：v1.10.0 不支持 udp+rtp:// source（那是 v1.14+ 才有的），不要用。
cat > mediamtx.yml <<'EOF'
logLevel: info
protocols: [tcp]
rtspAddress: :8554
paths:
  ai_out: {}
EOF
./mediamtx mediamtx.yml    # 保持此窗口运行
```

> `ai_out` 用默认 publisher：本工程 demo 作为 RTSP publisher 发布到
> `rtsp://127.0.0.1:8554/ai_out`，MediaMTX 转发给读取端。漏开 RTSP 或端口不对，
> 推流会失败、`ai_out` 没有画面。

```bash
# ③ 更新 streams_config.json 的 input_source 指向本地 test.mp4
#    （RTSP 输出地址默认 rtsp://<host>:8554/ai_out，可用 rtsp_output_url 覆盖）
# ④ 重新编译并启动（见 §6、§7）
```

```bash
# ⑤ 板端另开终端查看（或从主机经 SSH 隧道看）
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/ai_out
# 只验证流参数
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8554/ai_out
```

> 本工程 stream 模式输出 = `MPP 编码 -> ffmpeg -c copy -> RTSP -> MediaMTX`，
> 由 demo 作为 RTSP publisher 发布到 `rtsp://127.0.0.1:8554/ai_out`。板端
> MediaMTX 只要开 RTSP（`rtspAddress`）、`ai_out` 用默认 publisher 即可接收，
> **不需要 `udp+rtp://` source**（v1.10.0 不支持，那是 v1.14+ 才有的）。
>
> 在线模式（`-m stream`）输入文件播完即结束，不循环。test.mp4 只有几秒，
> 看完即断属正常。想持续看，用 `ffmpeg -stream_loop -1` 推的 RTSP 源作输入，
> 或改用循环解码。

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
codec_name=h264
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

检查板端系统库与 RKNPU2 Runtime：

```bash
ls /usr/lib/librockchip_mpp*.so /usr/lib/librga.so   # MPP/RGA 运行库
```

```bash
file rknpu2/lib/librknnrt.so
```

档位2 使用 MPP 硬件解码/编码与 RGA/OpenCV 2D 处理，需要 MPP/RGA 开发头文件
（见 `manhole_cover_detection/SOP.md` §1.1，含官方下载 URL）与 FFmpeg 开发库
（libavformat/libavcodec/libavutil，H264Demux 用）。板端系统库在
`/lib/aarch64-linux-gnu`，pkg-config 版本 libavformat 60.16.100。

注意：`rknpu2/`（`include/rknn_api.h` + `lib/librknnrt.so`，aarch64）**已随仓库提交**，
`CMakeLists.txt` 默认从 `rknpu2/include` 和 `rknpu2/lib` 查找，克隆后可直接编译。
如需核对/重新获取（Rockchip 官方 `rknn_model_zoo` 提交
`bad6c7334531becaf90a561988519b7bec34d0ab`，详见 `manhole_cover_detection/SOP.md` §1）：

```bash
cd manhole_cover_detection
mkdir -p rknpu2/include rknpu2/lib
# ① rknn_api.h
curl -L -o rknpu2/include/rknn_api.h \
  "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/bad6c7334531becaf90a561988519b7bec34d0ab/3rdparty/rknpu2/include/rknn_api.h"
# ② librknnrt.so（aarch64）
curl -L -o rknpu2/lib/librknnrt.so \
  "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/bad6c7334531becaf90a561988519b7bec34d0ab/3rdparty/rknpu2/Linux/aarch64/librknnrt.so"
file rknpu2/lib/librknnrt.so   # 必须显示 ARM aarch64
```

缺这两个文件时 `demo` 能编译但 `libmanhole_plugin.so` 编译/链接失败，运行时报
`dlopen failed`。RK3588 应显示 `ARM aarch64`，模型和 Runtime 必须匹配板端架构。

### 3.1 板端 MediaMTX（arm64）

仓库内 `mediamtx` 与 `mediamtx.exe` 均为 x86-64（主机/Windows 用），板端需另装
aarch64 版：

```bash
wget https://github.com/bluenviron/mediamtx/releases/download/v1.10.0/mediamtx_v1.10.0_linux_arm64v8.tar.gz
tar -xvzf mediamtx_v1.10.0_linux_arm64v8.tar.gz
```

板端 `mediamtx.yml` 至少配置（作为 RTSP publisher 接收本工程推流，见 §1.1）：

```yml
protocols: [tcp]
rtspAddress: :8554
paths:
  ai_out: {}
```

启动并保持运行：

```bash
./mediamtx mediamtx.yml
```

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
   ConfigService + VideoStreamManager；offline 模式结束后 ffmpeg 封装 MP4

src/manager/config_service.cpp
   监控 /dev/shm/ai_config.json 热更新，解析 streams/models/global_settings 配置

src/manager/video_stream_manager.cpp
   根据配置创建多路 VideoStream（每路输入拆主码流 + AI 流）

src/manager/video_stream.cpp
   主码流：H264Demux -> MPP 解码 -> RGA(OpenCV) -> CPU 画框 -> MPP 编码
            -> ffmpeg -c copy RTSP 或文件
   AI 流：FrameBroker 取帧 -> RGA 640 -> 推理 -> SharedAIResult

common/rk_media.{h,cpp}
   档位2 硬件层：MPP mpi_dec（对应 VDEC）/ mpp_enc（对应 VENC）/ RGA(OpenCV)
   （对应 IVPS，降级为 CPU 路径）

common/h264_demux.{h,cpp}
   FFmpeg 解封装（对应 VideoDemux）

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
--mediamtx IP:PORT       MediaMTX RTP 端点（兼容字段）
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
视频尺寸（插件以归一化 `x/y/w/h` 输出，画框时按输出帧尺寸还原）。

### 5.5 输出分支

offline 模式：`MPP 编码 -> raw H.264 文件 -> ffmpeg 封装 MP4`（路径由 `-o` 指定，
以 `.mp4` 结尾时先写 `.tmp.h264`，结束后自动封装，与 AX650 一致）。

stream 模式：`MPP 编码 -> ffmpeg -c copy -> RTSP -> MediaMTX`。demo 作为 RTSP
publisher 发布到 `rtsp://<host>:8554/ai_out`，板端 MediaMTX 开 RTSP（`rtspAddress`）、
`ai_out` 用默认 publisher 即可接收。RTSP 地址默认 `rtsp://<host>:8554/ai_out`，可由
配置 `rtsp_output_url` 覆盖（不依赖 `mediamtx_host/mediamtx_port`，那是兼容字段）。
**v1.10.0 不支持 `udp+rtp://` source（v1.14+ 才有），不要用**。当前不转发音频。

### 5.6 推理和绘制顺序

每一帧必须遵守：

```text
MPP 解码帧 -> 主码流缩放/格式转换 -> AIProcessor.processFrame（插件推理 -> AI_RESULT_T）
    -> OSDRenderer 检测框绘制在即将编码的 BGR 帧上 -> MPP 编码 -> 推流
```

AI 流把 `AI_RESULT_T` 写入 `SharedAIResult`，主码流编码线程读取后在当前输出帧
上叠加画框（跨流叠加，与 AX650 语义一致）。如果推理正确但查看的流没有检测框，
先确认写入输出的是绘制后的帧，并确认该帧 `nObjSize > 0`。

## 6. 编译 RK3588 程序

进入 RK3588 部署目录：

```bash
cd ~/Desktop/rk3588/manhole_cover_detection
git pull
```

如果板端没有 OpenCV 开发头文件，先执行本目录的离线准备脚本：

```bash
./install_opencv.sh
```

清理并配置：

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$HOME/opt/opencv-dev/usr/lib/aarch64-linux-gnu/cmake/opencv4"
```

编译：

```bash
cmake --build build -j2
```

产物：`bin/demo` 和 `bin/libmanhole_plugin.so`。

如果 CMake 因为完整 OpenCV 配置文件引用不存在的可选模块失败，应使用当前
仓库版本的 `CMakeLists.txt`。它只查找程序需要的 `core`、`imgproc` 模块，
不要求 `videostab` 等无关模块。

## 7. 启动完整 AI 流

先确认 `config/streams_config.json` 的 `input_source`：

- 在线推流：本地文件（如 `/home/cat/Desktop/rk3588/test.mp4`）或
  RTSP URL（如 `rtsp://127.0.0.1:8556/src_in`）。
- RTSP 输出默认 `rtsp://<host>:8554/ai_out`（可用配置 `rtsp_output_url` 覆盖）。

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
[H264Demux] opened ...: H.264 video stream idx=0 1138x720 ...
[RkMedia] decoder info-change: 1138x720 ...
[VideoStream] Stream 1: output_mode=rtsp rtsp://127.0.0.1:8554/ai_out (ffmpeg -c copy)
output decode: floats=... channels=9 anchors=8400 layout=[channels,anchors]
[ManholeCover] detections=... confidence: ...
```

`detections` 大于 0 时，框已经绘制到发送给编码器的视频帧上。

## 8. 主机查看 AI 流

主机上使用 `ffplay` 查看带框的 AI 流（经 SSH `-L 8557` 隧道）：

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
width=1920
height=1080
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
[VideoStream] Stream 1: output_mode=file /tmp/output_manhole.mp4.tmp.h264
[VideoStream] Stream 1: wrote encoder header 38 bytes
[Main] Offline input completed
[Main] Muxing offline H.264 to MP4: /tmp/output_manhole.mp4
[Main] Offline MP4 generated: /tmp/output_manhole.mp4
```

检查结果文件：

```bash
ffprobe -v error -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 /tmp/output_manhole.mp4
```

## 10. 排错顺序

### 10.1 没有 `output_mode=rtsp` / `output_mode=file`

确认 `-m stream` 或 `-m offline` 模式正确，并确认 `config/streams_config.json`
的 `input_source` 已配置；确认使用更新后的 `src/main.cpp` 重新编译出的 `bin/demo`。

### 10.2 `ai_out` 没有视频

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

输入流正常但输出为空时：

- 确认板端 mediamtx.yml 开了 RTSP（`rtspAddress: :8554`），`ai_out` 用默认
  publisher（不要配 `udp+rtp://`，v1.10.0 不支持）。
- 确认 `demo` 日志出现 `output_mode=rtsp rtsp://127.0.0.1:8554/ai_out (ffmpeg -c copy)`，
  且板端 ffmpeg 子进程没有立即报错退出（`ffmpeg` 需要能连上 MediaMTX RTSP）。
- 确认是 `-m stream` 而非 `-m offline`。

### 10.3 输出有视频但没有框

检查 RK3588 终端：

```text
output decode: ...
[ManholeCover] detections=... confidence: ...
```

如果 `sample_scores` 全部为 0，使用 FP RKNN 替换当前 INT8 模型；如果
`detections` 大于 0 但画面没有框，确认查看的是 `ai_out` 而不是 `src_in`，
并确认 `OSDRenderer` 已 `init()`（主码流启动时打印，未 init 时不画框）。

### 10.4 输入流读取失败

确认 SSH 隧道仍在运行，并且 RK3588 输入使用
`rtsp://127.0.0.1:8556/src_in`，输出使用
`rtsp://127.0.0.1:8554/ai_out`；不要使用主机局域网 IP。

### 10.5 找不到插件

默认插件路径是当前目录下的 `./libmanhole_plugin.so`（从 `bin/` 启动）。如果不在
当前目录，在配置的 `plugin` 字段（stream 级或 `models[]` 级）里显式写完整路径。

### 10.6 offline MP4 无画面 / ffmpeg 封装失败

- 确认 `getHeader()` 写出 SPS/PPS（日志 `wrote encoder header N bytes`，N>0），
  且输出包用 `mpp_packet_get_pos/get_length`（实际编码长度），不能用
  `get_size`（缓冲区容量，会把垃圾写入流，导致 `non-existing PPS`）。
- 若 MP4 已生成但画面异常，用 `ffprobe` 检查码流参数（宽度/高度应为配置输出
  尺寸，如 1920x1080）。

## 11. 验收标准

完成以下检查后，才算 RK3588 AI 流跑通：

```text
[ ] RK3588 可以 ffprobe 读取 src_in（或本地文件输入能解出 H.264 流）
[ ] demo 成功加载 FP RKNN 和 libmanhole_plugin.so
[ ] 离线模式生成 output_manhole.mp4 且能看到检测框
[ ] 每帧出现 [ManholeCover] detections 日志
[ ] stream 模式出现 output_mode=rtsp rtsp://127.0.0.1:8554/ai_out (ffmpeg -c copy)
[ ] MediaMTX 中出现 ai_out
[ ] 主机 ffplay 可以看到 ai_out
[ ] ai_out 画面中出现类别、置信度和检测框
[ ] ffmpeg 可以把 ai_out 保存为 ai_result.mp4
```
