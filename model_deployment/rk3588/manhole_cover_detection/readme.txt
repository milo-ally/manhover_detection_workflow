# RK3588 C++ 视频推理 Demo

本目录是 RK3588 的独立板端部署工程，使用 Rockchip RKNN Runtime 和 OpenCV：

```text
输入视频 -> OpenCV 解码 -> RGB letterbox -> RKNN 推理
         -> 五分类后处理/NMS -> 绘制检测框 -> OpenCV 保存输出视频
```

当前模型输出是单个 `output0 [1,9,8400]`，不是 Model Zoo YOLO11 示例中的三分支 DFL 输出，因此本工程只复用示例的 RKNN API 调用和输入处理方式，后处理按当前模型单独实现。

类别顺序：

```text
0 good
1 broke
2 lose
3 uncovered
4 circle
```

## 目录

```text
model_deployment/rk3588/manhole_cover_detection/
├── CMakeLists.txt
├── include/rknpu_manhole.hpp
├── plugins/model_manhole_cover.cpp
├── src/main.cpp
├── models/manhole-cover-yolo11s-production.rknn
└── rknpu2/
    ├── include/rknn_api.h
    └── lib/librknnrt.so
```

## 板端编译

板端需要安装 C++ 编译器、CMake、pkg-config 和带 `core/imgproc/videoio` 的 OpenCV 开发包：

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

然后使用脚本输出的 `OpenCV_DIR` 配置 CMake，例如：

```bash
cmake -S . -B build \
  -DOpenCV_DIR="$HOME/opt/opencv-dev/usr/lib/aarch64-linux-gnu/cmake/opencv4"
```

这种方式不会替换已经安装的 `rkmpp` FFmpeg 运行库。

将本目录完整复制到板端后执行：

```bash
cd model_deployment/rk3588/manhole_cover_detection
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

如果 OpenCV 不在系统默认路径，补充它的 CMake 路径：

```bash
cmake -S . -B build -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
```

## 离线视频运行

```bash
./bin/debug_demo \
  --model models/manhole-cover-yolo11s-production.rknn \
  --input input.mp4 \
  --output output_manhole.mp4 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --max-det 100
```

参数依次为：模型、输入视频、输出视频、置信度阈值、NMS IoU 阈值、最大检测数。程序会每帧输出检测数量和推理耗时，结束后保存完整结果视频。

如果板端 OpenCV 没有 MP4 编码器，先使用系统支持的输入格式，或在 `src/main.cpp` 将 `mp4v` 替换为板端可用的编码器并使用对应扩展名。

## 实现注意事项

- 程序启动时查询 RKNN 输入输出属性，兼容 NCHW/NHWC 输入。
- 输入使用 RGB `uint8`，与当前转换配置的 `/255` 预处理对应。
- 部署模型使用 FP RKNN；输出通过 `want_float = 1` 请求 Runtime 返回 FP32。
- 不要直接使用当前 INT8 产物：该检测头的坐标和类别分数共用输出量化范围，可能导致类别分数全部为 0。
- `src/main.cpp` 中的 `draw_detections` 是后续接入插件、告警或推流前的处理位置；当前先把结果写入输出视频。
- RKNN Runtime 必须与板端架构匹配。本目录提供的是 `aarch64` 的 `librknnrt.so`。
