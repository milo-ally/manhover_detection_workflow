# RK3588 C++ 离线视频推理 SOP

## 1. 准备文件

确认以下文件存在：

```bash
ls -lh \
  models/manhole-cover-yolo11s-production.rknn \
  rknpu2/include/rknn_api.h \
  rknpu2/lib/librknnrt.so
```

准备一个板端 OpenCV 可以读取的视频，例如 `input.mp4`。

本目录的 `rknn_api.h` 和 `librknnrt.so` 来自 Rockchip 官方 `rknn_model_zoo`，版本提交为：
`bad6c7334531becaf90a561988519b7bec34d0ab`。如果需要重新准备 Runtime，应从官方仓库的 `3rdparty/rknpu2` 取对应 RK3588/aarch64 文件。

## 2. 编译

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

检查 Runtime 动态库架构：

```bash
file rknpu2/lib/librknnrt.so
```

输出应为 ARM aarch64。若系统 OpenCV 缺少视频编解码支持，CMake 可能成功但 `VideoCapture` 或 `VideoWriter` 会打开失败，需要更换板端 OpenCV 或编码器。

## 3. 运行

```bash
./bin/debug_demo \
  models/manhole-cover-yolo11s-production.rknn \
  input.mp4 \
  output_manhole.mp4 \
  0.25 0.45 100
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
