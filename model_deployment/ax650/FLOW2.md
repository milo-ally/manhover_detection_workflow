# AX650 井盖检测 AI 流

本文档只针对：

```text
model_deployment/ax650/manhole_cover_detection
```

这是 AX650 井盖检测独立小工程，直接通过 AX Engine 加载 `.axmodel`，不依赖
RK3588 的 RKNPU2、`h264_rkmpp` 或 `.rknn` 文件。它复用了原 device_side 主工程的
完整链路：ConfigService、VideoStreamManager、VideoDemux、IVPS/OSD、VENC/RTP。

本工程支持两种模式：

```text
离线模式：本地 H.264 文件 -> VDEC/IVPS -> AX Engine -> OSD 画框 -> H.264 -> MP4
在线模式：RTSP/文件 -> VDEC/IVPS -> AX Engine -> OSD 画框 -> H.264/RTP -> MediaMTX
```

## 1. 流程总览

```text
主机 test.mp4
    |
     | FFmpeg 推送 rtsp://127.0.0.1:554/src_in（主机 MediaMTX）
    v
AX650 demo（-m stream）
    |
    | VideoDemux 读取 -> VDEC 解码 -> IVPS 缩放
    v
NV12 -> RGB letterbox -> AXModel 推理
    |
    | AI_RESULT_T（归一化坐标、类别、置信度）
    v
OSD（DefaultOSDRenderer，AX_IVPS_RGN_Update）叠加框和标签
    |
    | VENC H.264 -> RTP
    v
AX650 本地 MediaMTX（RTP 端口 8000，路径按现场配置）
    |
    +--> SSH -L 8557 -> 主机 ffplay 查看
    +--> SSH -L 8557 -> 主机 ffmpeg 录制
```

在线输出使用板端原生 VENC/RTP 推流，不经过 FFmpeg 管道，也不要复制 RK3588 文档中
的 `h264_rkmpp` 命令到这里。

## 2. 固定路径和模型

工程目录：

```bash
cd model_deployment/ax650/manhole_cover_detection
```

模型文件不提交到 Git（`*.axmodel` 被 .gitignore 忽略）。将已经验证通过的 AXModel
放到工程 `models/` 目录，例如：

```text
models/yolo11s-manhole-detection.axmodel
```

模型约定：

```text
输入：640x640，U8 NHWC RGB，letterbox padding=114
输出：output0 [1,9,8400] FP32
0..3：cx/cy/w/h
4..8：good/broke/lose/uncovered/circle 分数
```

插件文件：

```text
plugins/model_manhole_cover.cpp
```

主程序通过 `AIProcessor` 的 `dlopen` 加载：

```text
libmanhole_plugin.so
```

（未配置 `plugin` 字段时默认 `./libmanhole_plugin.so`，失败回退 `./bin/libmanhole_plugin.so`。）

## 3. 主机准备 MediaMTX、输入流和 SSH 隧道

按照 `FLOW1.md` 在主机安装并启动 MediaMTX、推送输入视频、建立双向 SSH 隧道：

```powershell
ssh -R 8556:127.0.0.1:554 -L 8557:127.0.0.1:8554 root@172.19.30.3
```

```bash
ffmpeg -re -stream_loop -1 -i ./test.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:554/src_in
```

AX650 上验证输入（只检查 `127.0.0.1:8556`，不要用主机局域网 IP，也不要 ping）：

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=index,codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8556/src_in
```

只有 AX650 能通过 `127.0.0.1:8556` 读取 `src_in` 后，才继续运行 AI 程序。
注意本工程只支持 H.264 输入；H.265/HEVC 输入不支持，需要先把源视频转成 H.264。

## 4. AX650 编译环境

编译需要：

```text
AArch64 交叉编译器
AX650 SDK 头文件：msp_sdk/include
AX650 运行库：板端通常位于 /soc/lib
OpenCV core、imgproc、videoio 开发文件
```

检查交叉编译器：

```bash
aarch64-linux-gnu-g++ --version
```

检查 SDK 头文件：

```bash
test -f msp_sdk/include/ax_engine_api.h && echo AX_SDK_OK
```

`CMakeLists.txt` 默认使用 `aarch64-linux-gnu-gcc` / `aarch64-linux-gnu-g++`，且要求
`msp_sdk/` 已放入工程目录（`CMakeLists.txt` 检查 `msp_sdk/include/ax_sys_api.h`，找不到会
直接报错）。如果交叉编译器前缀不同，在 CMake 配置时显式传入编译器路径。

## 5. 编译小工程

在工程目录执行：

```bash
cd model_deployment/ax650/manhole_cover_detection
```

清理旧构建：

```bash
rm -rf build bin
```

配置：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

如果 CMake 找不到交叉编译器，使用：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
```

编译并安装：

```bash
cmake --build build -j$(nproc)
cmake --install build
```

产物：

```text
bin/demo
bin/libmanhole_plugin.so
```

将 `bin/`、`models/` 和配置文件复制到 AX650，例如：

```text
/tmp/manhole_cover_detection/bin/demo
/tmp/manhole_cover_detection/bin/libmanhole_plugin.so
/tmp/manhole_cover_detection/models/yolo11s-manhole-detection.axmodel
/tmp/manhole_cover_detection/config/streams_config.json
```

## 6. 程序说明

### 6.1 参数全部使用选项名

主程序支持：

```text
-c <file>       streams 配置 JSON（默认 config/streams_config.json）
-m <offline|stream>   运行模式
-o <file>       offline 输出：.mp4 结尾自动封装 MP4，否则输出裸 H.264
-h              帮助
```

输入源不通过位置参数传入，一律在 `streams_config.json` 的 `input_source` 里配置。
`input_source` 可以是本地文件路径，也可以是 RTSP URL。

### 6.2 阈值透传

配置文件中每路流的 `conf_thres` / `nms_thres`，以及 `models[]` 里的
`conf_threshold` / `nms_threshold`，由 `AIProcessor::applyModelParamsToEnv()` 写入
环境变量：

```text
MODEL_CONF_THRESH
MODEL_NMS_THRESH
```

插件 `model_manhole_cover.cpp` 在 `Init()` 中优先读取 `MANHOLE_CONF_THRESH` /
`MANHOLE_NMS_THRESH`，未设置时回退到 `MODEL_CONF_THRESH` / `MODEL_NMS_THRESH`，
默认 0.25 / 0.45。因此配置阈值会真正影响 AXModel 后处理，而不是只显示在配置中。

### 6.3 离线输出分支（-m offline）

```text
本地 H.264 文件 -> VDEC 解码 -> IVPS 缩放 -> AI 推理 + OSD
   -> VENC H.264 -> raw H.264
   -> -o 以 .mp4 结尾时自动调用 ffmpeg 封装为 MP4
```

此模式用于先确认模型、预处理、后处理和画框正确，不需要 MediaMTX。
只支持 H.264 输入；H.265/HEVC 输入不支持。

### 6.4 在线输出分支（-m stream）

```text
VDEC/IVPS -> AI 推理 + OSD -> VENC H.264 -> RTP -> MediaMTX
```

主码流通过原生 VENC/RTP 链路把 H.264 推到 MediaMTX 的 RTP 端口（默认
`127.0.0.1:8000`，可用 `--mediamtx` 或 `--mediamtx-host`/`--mediamtx-port` 覆盖）。
程序启动时会打印提示：`Note: Configure MediaMTX with: source: udp+rtp://0.0.0.0:8000`，
需要在板端 `mediamtx.yml` 中把对应路径的 source 指向该 RTP 端口。当前不转发音频。

### 6.5 推理和绘制顺序

每一帧必须遵守：

```text
VDEC 解码帧 -> NV12 -> 插件内 RGB letterbox -> AXModel Inference
    -> AI_RESULT_T -> OSD（AX_IVPS_RGN_Update）叠加框
    -> VENC/RTP 或离线 VENC 写入
```

OSD 叠加发生在 AI 分支，画框结果出现在主输出流上。如果推理正确但查看的流没有检测框，
先确认查看的是主输出流，而不是原始输入流。

## 7. 离线模式验收

将 `test.mp4`（H.264）和 AXModel 放到板端，把 `streams_config.json` 的
`input_source` 指向该文件后执行：

```bash
cd /tmp/manhole_cover_detection/bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m offline -o /tmp/output_boxed.mp4
```

预期日志：

```text
[Main] Loading streams from config file: ../config/streams_config.json
[ManholeCover] Loading model: /tmp/manhole_cover_detection/models/yolo11s-manhole-detection.axmodel
[ManholeCover] thresholds: conf=0.250 nms=0.450
[Main] Offline input completed
[Main] Offline MP4 generated: /tmp/output_boxed.mp4
```

检查结果文件：

```bash
ffprobe -v error -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 /tmp/output_boxed.mp4
```

将 `/tmp/output_boxed.mp4` 拷回主机查看，或在板端使用支持文件播放的工具检查。

## 8. 在线 AI 流运行

先在 AX650 上确认主机 MediaMTX、输入推流和 SSH 隧道仍在运行，输入可读：

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=index,codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8556/src_in
```

确认板端 MediaMTX 正在运行，且对应路径已配置 `source: udp+rtp://0.0.0.0:8000`
（或与 `--mediamtx` 指定的端口一致）。

把 `streams_config.json` 的 `input_source` 改为：

```text
rtsp://127.0.0.1:8556/src_in
```

进入程序目录并设置运行库：

```bash
cd /tmp/manhole_cover_detection/bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
```

启动在线 AI 流：

```bash
./demo -c ../config/streams_config.json -m stream
```

预期日志：

```text
[Main] Loading streams from config file: ../config/streams_config.json
[AIProcessor] Loading plugin: ./libmanhole_plugin.so
[AIProcessor] Model initialized successfully
[ManholeCover] Loading model: ...
[ManholeCover] thresholds: conf=0.250 nms=0.450
[OSD] Using default OSD renderer for AI stream 1
[OSD] Initialized OSD management for AI stream 1, N pipelines
```

程序终端和主机的 MediaMTX、输入 FFmpeg 终端都不要关闭。

## 9. 主机查看和录制

主机上查看输出 AI 流：

```bash
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8557/ai_out
```

如果主机没有 `ffplay`，安装完整 FFmpeg：

```bash
sudo apt install -y ffmpeg
```

无图形界面时检查输出流：

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=index,codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8557/ai_out
```

录制带框结果：

```bash
ffmpeg -y -rtsp_transport tcp -i rtsp://127.0.0.1:8557/ai_out -c copy ax650_ai_result.mp4
```

停止录制不会自动停止 AX650 推理程序。

## 10. 排错顺序

### 10.1 `src_in` 读取失败

在 AX650 执行（先确认 SSH 隧道仍在运行）：

```bash
ffprobe -v error -rtsp_transport tcp rtsp://127.0.0.1:8556/src_in
```

不要在 AX650 上 ping 主机或使用主机局域网 IP；本方案只检查 `127.0.0.1:8556` 的
TCP RTSP 流。

### 10.2 找不到 AX 运行库

```bash
find /soc/lib /usr/lib -name 'libax_engine.so*' -o -name 'libax_sys.so*'
```

并确认：

```bash
echo "$LD_LIBRARY_PATH"
```

### 10.3 找不到插件

```bash
ls -l ./demo ./libmanhole_plugin.so
```

默认插件路径是当前目录下的 `./libmanhole_plugin.so`。如果不在当前目录，在配置文件的
`plugin` 字段（stream 级或 `models[]` 级）里显式写完整路径。

### 10.4 模型初始化失败

```bash
ls -lh /tmp/manhole_cover_detection/models/yolo11s-manhole-detection.axmodel
```

确认模型是 AX650 对应的 `.axmodel`，不是 RK3588 的 `.rknn`。

### 10.5 输出 RTSP 没有画面

先确认板端 MediaMTX 正在运行，且路径 source 指向程序实际推送的 RTP 端口
（默认 `udp+rtp://0.0.0.0:8000`，可用 `--mediamtx IP:PORT` 修改端点）。再查看程序
日志中的 MediaMTX endpoint 提示和 RTP 推送日志。

主机通过 SSH `-L 8557` 查看时使用：

```text
rtsp://127.0.0.1:8557/ai_out
```

### 10.6 输出有视频但没有框

确认程序日志中的 OSD 链路：

```text
[OSD] Updated AI result for stream ...: nObjSize=...
[OSD] AX_IVPS_RGN_Update success: ...
```

如果 `nObjSize=0`，先降低配置中的 `conf_threshold` 验证（例如 0.05）。如果仍然为 0，
检查 AXModel、输入色彩顺序、letterbox 和模型输出；如果日志有检测但画面没有框，确认
查看的是主输出流而不是 `src_in`，并确认没有设置 `AX_DISABLE_OSD=1`。

## 11. 验收标准

```text
[ ] AX650 可以 ffprobe 读取 src_in
[ ] demo 成功加载 AXModel 和 libmanhole_plugin.so
[ ] 离线模式生成 output_boxed.mp4
[ ] 离线视频中能看到检测框、类别和置信度
[ ] 在线模式通过 VENC/RTP 推送到 MediaMTX
[ ] MediaMTX 中出现 AI 输出流
[ ] 主机 ffplay 可以看到 AI 输出流
[ ] AI 输出流中检测框与离线模式一致
[ ] ffmpeg 可以把 AI 输出流录制成 ax650_ai_result.mp4
```
