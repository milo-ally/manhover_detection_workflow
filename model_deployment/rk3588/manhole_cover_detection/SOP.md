# RK3588 C++ 离线视频推理 SOP

## 1. 准备文件

确认以下文件存在：

```bash
ls -lh \
  models/manhole-cover-yolo11s-production.rknn \
  rknpu2/include/rknn_api.h \
  rknpu2/lib/librknnrt.so
```

注意：`rknpu2/` 目录不随仓库提交（已被 `.gitignore` 忽略），`CMakeLists.txt` 默认在
`rknpu2/include` 和 `rknpu2/lib` 查找 `rknn_api.h` 与 `librknnrt.so`，编译前必须先把
Runtime 放回本目录。本工程使用的 `rknn_api.h` 和 `librknnrt.so` 来自 Rockchip 官方 `rknn_model_zoo`，版本提交为：
`bad6c7334531becaf90a561988519b7bec34d0ab`。如果需要重新准备 Runtime，应从官方仓库的 `3rdparty/rknpu2` 取对应 RK3588/aarch64 文件。

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

## 3. 运行

```bash
./bin/debug_demo \
  --model models/manhole-cover-yolo11s-production.rknn \
  --input input.mp4 \
  --output output_manhole.mp4 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --max-det 100
```

启动时应打印模型输入输出属性。程序默认使用：

```text
conf_threshold = 0.25
iou_threshold  = 0.45
max_det        = 100
```

输出视频包含检测框、类别名和置信度。推理过程中的实时日志包含帧号、检测数量和单帧推理耗时。

## 4. 后续插件接入点

当前 `src/main.cpp` 的处理顺序是：

```text
读取 frame
 -> detector.infer(frame, detections, ...)
 -> draw_detections(frame, detections)
 -> writer.write(frame)
```

后续插件可以在 `infer` 返回后使用 `frame` 和 `detections`，完成告警、业务回调或推流；先保持离线视频链路跑通，再替换输入输出模块。
