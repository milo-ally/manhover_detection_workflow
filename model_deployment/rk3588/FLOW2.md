# RK3588 AI 流

本文档在 `FLOW1.md` 的主机 MediaMTX 和输入推流基础上，使用 RK3588 板端
C++/RKNPU2 完成完整的 AI 流：

```text
主机 test.mp4
    |
    | FFmpeg 推送到主机 554/src_in
    v
主机 MediaMTX:554
    |
    | RTSP/TCP
    v
RK3588 OpenCV VideoCapture
    |
    | BGR 帧 -> RGB letterbox -> RKNPU2 RKNN 推理
    v
绘制检测框、类别和置信度
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

## 5. C++ 程序改造点

### 5.1 输入从 MP4 扩展为 RTSP

`src/main.cpp` 使用：

```cpp
cv::VideoCapture capture(input_path);
```

因此 `--input` 可以是本地文件，也可以是 RTSP URL。AI 流运行时使用：

```text
    --input rtsp://127.0.0.1:8556/src_in
```

程序保持逐帧读取：

```cpp
while (capture.read(frame)) {
    detector.infer(frame, detections, conf_threshold, iou_threshold, max_det);
    draw_detections(frame, detections);
}
```

### 5.2 输入预处理必须与验证一致

每一帧进入 RKNPU2 前执行：

1. 按原图比例缩放到 640x640 内。
2. 使用 `(114,114,114)` 填充剩余区域。
3. BGR 转 RGB。
4. 按 RKNN 输入属性选择 NHWC 或 NCHW 排列。
5. 以 `uint8` 送入 RKNN。

检测框从 640x640 letterbox 坐标减去 padding，再除以缩放比例，恢复到原始
视频尺寸。

### 5.3 RKNPU2 推理和后处理

初始化阶段查询 input/output tensor 属性，并确认输出是 9 个通道和 8400 个
候选框。推理阶段调用：

```cpp
rknn_inputs_set(context_, 1, &input);
rknn_run(context_, nullptr);
output.want_float = 1;
rknn_outputs_get(context_, 1, &output, nullptr);
```

后处理选择每个候选框的最高类别分数，执行置信度过滤和按类别 NMS，然后把
结果转换成原始视频坐标。

### 5.4 绘制 AI 结果

`draw_detections()` 在推流之前调用：

```cpp
draw_detections(frame, detections);
```

它在 BGR 帧上绘制矩形框、类别名和置信度。必须确认推流写入的是绘制后的
`frame`，不能把原始帧写入 FFmpeg 管道。

### 5.5 输出分支

当 `--output` 是本地文件时，使用 OpenCV `VideoWriter` 保存 MP4：

```text
--output output_manhole.mp4
```

当 `--output` 是 `rtsp://` 或 `rtsps://` URL 时，程序启动 FFmpeg 子进程，
把绘制后的 BGR 原始帧写入 stdin：

```text
ffmpeg -f rawvideo -pix_fmt bgr24 -s WIDTHxHEIGHT -r FPS -i pipe:0 \
  -an -c:v h264_rkmpp -pix_fmt yuv420p -f rtsp \
  -rtsp_transport tcp rtsp://127.0.0.1:8554/ai_out
```

这里的 `-s`、`-r` 由输入视频实际属性生成。当前 AI 流默认不转发音频，
因为程序只从 OpenCV 读取视频帧；如需音频，需要额外设计音视频同步和音频
转发链路，不能仅在当前 BGR 管道中增加参数。

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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="/home/cat/opt/opencv-dev/usr/lib/aarch64-linux-gnu/cmake/opencv4"
```

编译：

```bash
cmake --build build -j2
```

如果 CMake 因为完整 OpenCV 配置文件引用不存在的可选模块失败，应使用当前
仓库版本的 `CMakeLists.txt`。它只查找程序需要的 `core`、`imgproc` 和
`videoio`，不要求 `videostab` 等无关模块。

## 7. 启动完整 AI 流

确认主机上的 MediaMTX 和 `src_in` 推流仍在运行后，在 RK3588 执行：

```bash
./bin/debug_demo --model models/manhole-cover-yolo11s-production.rknn --input rtsp://127.0.0.1:8556/src_in --output rtsp://127.0.0.1:8554/ai_out --conf-thres 0.25 --iou-thres 0.45 --max-det 100
```

正常启动时应看到：

```text
input: ...
output: ...
video=1138x720 fps=...
output_mode=rtsp command=ffmpeg ... h264_rkmpp ... rtsp://127.0.0.1:8554/ai_out
output decode: ...
frame=0 detections=... inference_ms=...
```

程序会逐帧打印检测数量和推理耗时。`detections` 大于 0 时，框已经绘制到
发送给 FFmpeg 的视频帧上。

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

## 9. 参数说明

所有运行参数必须使用 `--`：

```text
--model       RKNN 模型路径
--input       本地视频或输入 RTSP URL
--output      本地 MP4 路径或输出 RTSP URL
--conf-thres  置信度阈值，默认 0.25
--iou-thres   NMS IoU 阈值，默认 0.45
--max-det     单帧最多保留的检测框数量，默认 100
```

离线验证：

```bash
./bin/debug_demo --model models/manhole-cover-yolo11s-production.rknn --input test.mp4 --output output_manhole.mp4 --conf-thres 0.25 --iou-thres 0.45 --max-det 100
```

实时 AI 流：

```bash
./bin/debug_demo --model models/manhole-cover-yolo11s-production.rknn --input rtsp://127.0.0.1:8556/src_in --output rtsp://127.0.0.1:8554/ai_out --conf-thres 0.25 --iou-thres 0.45 --max-det 100
```

## 10. 排错顺序

### 10.1 没有 `output_mode=rtsp`

确认 `--output` 是完整的 `rtsp://...` URL，并且使用的是更新后的
`src/main.cpp` 重新编译出的程序。

### 10.2 FFmpeg 管道立即退出

在 RK3588 检查：

```bash
ffmpeg -encoders | grep h264_rkmpp
```

如果没有输出，当前 FFmpeg 不能执行文档中的硬件编码命令。先恢复板端
Rockchip multimedia FFmpeg，或明确选择板端实际存在的 H.264 编码器后再修改
`src/main.cpp`。

### 10.3 `ai_out` 没有视频

同时检查三处：

```bash
ffprobe -v error -rtsp_transport tcp rtsp://127.0.0.1:8556/src_in
```

```bash
ffprobe -v error -rtsp_transport tcp rtsp://127.0.0.1:8557/ai_out
```

```bash
ps aux | grep debug_demo
```

输入流正常但输出为空时，查看 RK3588 终端中的 FFmpeg warning 和
`ffmpeg RTSP output pipe closed`。

### 10.4 输出有视频但没有框

检查 RK3588 终端：

```text
output: ...
output decode: ...
frame=... detections=...
```

如果 `sample_scores` 全部为 0，使用 FP RKNN 替换当前 INT8 模型；如果
`detections` 大于 0 但画面没有框，确认主机查看的是 `ai_out`，不是
`src_in`，并确认程序调用顺序是“推理 -> 绘制 -> 写入 FFmpeg”。

### 10.5 输入流读取失败

确认 SSH 隧道仍在运行，并且 RK3588 输入使用
`rtsp://127.0.0.1:8556/src_in`，输出使用
`rtsp://127.0.0.1:8554/ai_out`；不要使用主机局域网 IP。

## 11. 验收标准

完成以下检查后，才算 RK3588 AI 流跑通：

```text
[ ] RK3588 可以 ffprobe 读取 src_in
[ ] C++ 程序成功加载 FP RKNN
[ ] 每帧出现 inference_ms 日志
[ ] 输出 FFmpeg 使用 h264_rkmpp 且没有立即退出
[ ] MediaMTX 中出现 ai_out
[ ] 主机 ffplay 可以看到 ai_out
[ ] ai_out 画面中出现类别、置信度和检测框
[ ] ffmpeg 可以把 ai_out 保存为 ai_result.mp4
```
