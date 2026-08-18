# AX650 井盖检测 AI 流

本文档只针对：

```text
model_deployment/ax650/manhole_cover_detection
```

这是 AX650 井盖检测独立小工程，不依赖 `device_side` 主工程，也不使用
RK3588 的 RKNPU2、`h264_rkmpp` 或 `.rknn` 文件。

本工程支持两种模式：

```text
离线模式：MP4 -> OpenCV -> AX Engine -> 画框 -> MP4
在线模式：RTSP -> OpenCV -> AX Engine -> 画框 -> FFmpeg/libx264 -> RTSP/MediaMTX
```

## 1. 流程总览

```text
主机 test.mp4
    |
    | FFmpeg 推送 rtsp://HOST_IP:8554/src_in
    v
主机 MediaMTX
    |
    | RTSP/TCP
    v
AX650 OpenCV VideoCapture
    |
    | BGR -> NV12 -> AXModel 推理
    v
AI_RESULT_T
    |
    | 还原归一化坐标并绘制框、类别、置信度
    v
绘制后的 BGR 帧
    |
    | FFmpeg stdin，libx264，RTSP/TCP
    v
主机 MediaMTX:8554/ai_out
    |
    +--> 主机 ffplay 查看
    +--> 主机 ffmpeg 录制
```

AX650 小工程的在线输出使用 FFmpeg 软件编码器 `libx264`。不要把 RK3588
文档中的 `h264_rkmpp` 命令复制到这里。

## 2. 固定路径和模型

工程目录：

```bash
cd model_deployment/ax650/manhole_cover_detection
```

模型文件不提交到 Git。将已经验证通过的 AXModel 放到 AX650 板端，例如：

```text
/tmp/manhole-cover-yolo11s-production.axmodel
```

模型约定：

```text
输入：640x640，U8 NHWC RGB
输出：output0 [1,9,8400] FP32
0..3：cx/cy/w/h
4..8：good/broke/lose/uncovered/circle
```

插件文件：

```text
plugins/model_manhole_cover.cpp
```

主程序通过 `dlopen` 加载：

```text
libmanhole_plugin.so
```

## 3. 主机准备 MediaMTX 和输入流

主机安装：

```bash
sudo apt update
```

```bash
sudo apt install -y ffmpeg
```

检查主机 FFmpeg：

```bash
ffmpeg --version
```

获取主机局域网 IP：

```bash
hostname -I
```

例如主机输出：

```text
192.168.0.129 172.17.0.1
```

使用 `192.168.0.129`，不要使用 `172.17.0.1` 或 `127.0.0.1` 作为 AX650
连接主机时的地址。

按照 `FLOW1.md` 下载并启动 MediaMTX：

```bash
./mediamtx mediamtx.yml
```

另开主机终端推送输入测试视频：

```bash
ffmpeg -re -stream_loop -1 -i ./test.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://192.168.0.129:8554/src_in
```

输入流验证：

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=index,codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8554/src_in
```

AX650 上验证：

```bash
ping 192.168.0.129
```

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=index,codec_name,width,height -of default=noprint_wrappers=1 rtsp://192.168.0.129:8554/src_in
```

只有 AX650 能读取 `src_in` 后，才继续运行 AI 程序。

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

当前 `CMakeLists.txt` 默认使用：

```text
aarch64-linux-gnu-gcc
aarch64-linux-gnu-g++
```

如果交叉编译器前缀不同，在 CMake 配置时显式传入编译器路径。

## 5. 编译小工程

在仓库根目录执行：

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

如果 CMake 找不到交叉编译器或 SDK，使用：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DAX_SDK_DIR=/path/to/ax650n_sdk
```

编译：

```bash
cmake --build build -j$(nproc)
```

产物：

```text
bin/debug_demo
bin/libmanhole_plugin.so
```

将 `bin/` 和 AXModel 复制到 AX650，例如：

```text
/tmp/manhole_cover_detection/bin/debug_demo
/tmp/manhole_cover_detection/bin/libmanhole_plugin.so
/tmp/manhole-cover-yolo11s-production.axmodel
```

## 6. 程序改造说明

### 6.1 参数全部使用 `--`

主程序支持：

```text
--input       本地 MP4 或 RTSP URL
--output      本地 MP4 路径或 RTSP URL
--model       AXModel 路径
--plugin      插件路径，默认 ./libmanhole_plugin.so
--conf-thres  置信度阈值，默认 0.25
--iou-thres   NMS IoU 阈值，默认 0.45
--encoder     RTSP 输出编码器，默认 libx264
```

不再支持位置参数。输入、输出和模型都必须使用选项名。

### 6.2 阈值透传

主程序在加载插件前设置：

```text
MANHOLE_CONF_THRESH
MANHOLE_NMS_THRESH
```

插件 `model_manhole_cover.cpp` 在 `Init()` 中读取这两个环境变量。因此命令行
阈值会真正影响 AXModel 后处理，而不是只显示在命令行中。

### 6.3 离线输出分支

当 `--output` 不是 RTSP URL 时，程序使用 OpenCV `VideoWriter`：

```text
OpenCV BGR frame -> MP4
```

此模式用于先确认模型、预处理、后处理和画框正确，不需要 MediaMTX。

### 6.4 在线输出分支

当 `--output` 以 `rtsp://` 或 `rtsps://` 开头时，程序启动：

```text
ffmpeg -f rawvideo -pix_fmt bgr24 -s WIDTHxHEIGHT -r FPS -i pipe:0 \
  -an -c:v libx264 -preset ultrafast -tune zerolatency \
  -pix_fmt yuv420p -f rtsp -rtsp_transport tcp rtsp://HOST_IP:8554/ai_out
```

程序将“已经完成推理并绘制检测框”的 BGR 帧写入 FFmpeg stdin。当前不转发
音频，输出是视频 AI 流。

### 6.5 推理和绘制顺序

每一帧必须遵守：

```text
VideoCapture.read
    -> BGR 转 NV12
    -> AXModel Inference
    -> AI_RESULT_T 归一化坐标转像素坐标
    -> drawResult
    -> VideoWriter 或 FFmpeg stdin
```

如果把原始 `bgr` 写入输出，就算推理正确，查看的流也不会有检测框。

## 7. 离线模式验收

将 `test.mp4` 和 AXModel 放到板端后执行：

```bash
cd /tmp/manhole_cover_detection/bin
```

```bash
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
```

```bash
./debug_demo --input /tmp/test.mp4 --output /tmp/output_boxed.mp4 --model /tmp/manhole-cover-yolo11s-production.axmodel --conf-thres 0.25 --iou-thres 0.45
```

预期日志：

```text
[INFO] output_mode=file
[ManholeCover] Loading model: /tmp/manhole-cover-yolo11s-production.axmodel
[ManholeCover] thresholds: conf=0.250 nms=0.450
[INFO] frame=1 detections=...
[INFO] output video: /tmp/output_boxed.mp4, frames=...
```

检查结果文件：

```bash
ffprobe -v error -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 /tmp/output_boxed.mp4
```

将 `/tmp/output_boxed.mp4` 拷回主机查看，或在板端使用支持文件播放的工具检查。

## 8. 在线 AI 流运行

先确认 AX650 上的 FFmpeg 有 `libx264`：

```bash
ffmpeg -encoders | grep libx264
```

进入程序目录并设置运行库：

```bash
cd /tmp/manhole_cover_detection/bin
```

```bash
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
```

启动在线 AI 流：

```bash
./debug_demo --input rtsp://192.168.0.129:8554/src_in --output rtsp://192.168.0.129:8554/ai_out --model /tmp/manhole-cover-yolo11s-production.axmodel --conf-thres 0.25 --iou-thres 0.45 --encoder libx264
```

预期日志：

```text
[INFO] output_mode=rtsp encoder=libx264
[INFO] ffmpeg command: ffmpeg ... rtsp://192.168.0.129:8554/ai_out
[ManholeCover] thresholds: conf=0.250 nms=0.450
[INFO] frame=1 detections=...
```

程序终端和主机的 MediaMTX、输入 FFmpeg 终端都不要关闭。

## 9. 主机查看和录制

主机上查看输出 AI 流：

```bash
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/ai_out
```

如果主机没有 `ffplay`，安装完整 FFmpeg：

```bash
sudo apt install -y ffmpeg
```

无图形界面时检查输出流：

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=index,codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8554/ai_out
```

录制带框结果：

```bash
ffmpeg -y -rtsp_transport tcp -i rtsp://127.0.0.1:8554/ai_out -c copy ax650_ai_result.mp4
```

停止录制不会自动停止 AX650 推理程序。

## 10. 排错顺序

### 10.1 `src_in` 读取失败

在 AX650 执行：

```bash
ping 192.168.0.129
```

```bash
ffprobe -v error -rtsp_transport tcp rtsp://192.168.0.129:8554/src_in
```

确认使用的是主机局域网 IP，不是 `127.0.0.1`、`172.17.0.1`。

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
ls -l ./debug_demo ./libmanhole_plugin.so
```

默认插件路径是当前目录下的 `./libmanhole_plugin.so`。如果不在当前目录，
显式传入：

```bash
--plugin /tmp/manhole_cover_detection/bin/libmanhole_plugin.so
```

### 10.4 模型初始化失败

```bash
ls -lh /tmp/manhole-cover-yolo11s-production.axmodel
```

确认模型是 AX650 对应 `.axmodel`，不是 RK3588 的 `.rknn`。

### 10.5 输出 RTSP 没有画面

先检查 AX650：

```bash
ffmpeg -encoders | grep libx264
```

再查看程序打印的完整 FFmpeg 命令和 warning。确认主机 MediaMTX 正在运行，
并且输出地址是：

```text
rtsp://192.168.0.129:8554/ai_out
```

主机本地查看时才使用：

```text
rtsp://127.0.0.1:8554/ai_out
```

### 10.6 输出有视频但没有框

确认程序日志中的：

```text
[ManholeCover] detections=...
[INFO] frame=... detections=...
```

如果 `detections=0`，先降低阈值验证：

```bash
--conf-thres 0.05 --iou-thres 0.45
```

如果仍然为 0，检查 AXModel、输入色彩顺序、letterbox 和模型输出；如果日志
有检测但画面没有框，确认查看的是 `ai_out` 而不是 `src_in`，并确认绘制发生
在写入 FFmpeg 之前。

## 11. 验收标准

```text
[ ] AX650 可以 ffprobe 读取 src_in
[ ] debug_demo 成功加载 AXModel 和 libmanhole_plugin.so
[ ] 离线模式生成 output_boxed.mp4
[ ] 离线视频中能看到检测框、类别和置信度
[ ] 在线模式启动 FFmpeg/libx264
[ ] MediaMTX 中出现 ai_out
[ ] 主机 ffplay 可以看到 ai_out
[ ] ai_out 中检测框与离线模式一致
[ ] ffmpeg 可以把 ai_out 录制成 ax650_ai_result.mp4
```
