AX650N 井盖检测最小练习工程

一、目录结构

manhole_cover_detection/
  include/ai_interface.h
    插件与主程序共用的 AI 模型接口和检测结果结构。
  plugins/letterbox_utils.hpp
    RGB 图像 letterbox 预处理和坐标还原工具。
  plugins/model_manhole_cover.cpp
    井盖模型插件，实现 Init、Inference、Deinit 和插件导出函数。
  src/main.cpp
    独立 MP4 推理程序，动态加载井盖插件并输出打框视频。
  msp_sdk/include/
    编译需要的 AX650N SDK 头文件。
  CMakeLists.txt
    AArch64 交叉编译配置。

二、构建环境

需要：

  AArch64 交叉编译器
  AX650N SDK 头文件
  AX650N 运行库：libax_engine.so、libax_sys.so
  OpenCV core、imgproc、videoio 模块

SDK 头文件已经复制到 msp_sdk/include。AX650N 动态库通常位于板端 /soc/lib，CMakeLists.txt 已加入 /soc/lib 和 /usr/lib 搜索路径。

三、编译

在 AX650N 交叉编译环境中执行：

  cd model_deployment/manhole_cover_detection
  cmake -S . -B build
  cmake --build build -j$(nproc)
  cmake --install build

如果交叉编译器名称或 SDK 位置不同，可以指定：

  cmake -S . -B build \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DAX_SDK_DIR=/path/to/ax650n_sdk

构建产物：

  bin/libmanhole_plugin.so
  bin/debug_demo

四、运行

将输入视频和模型复制到板端，例如：

  /tmp/input.mp4
  /tmp/manhole-cover-yolo11s-production.axmodel

进入插件和可执行文件所在目录：

  cd bin
  export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
  ./debug_demo \
    --input /tmp/input.mp4 \
    --output /tmp/output_boxed.mp4 \
    --model /tmp/manhole-cover-yolo11s-production.axmodel

检测到目标时，插件只输出一行摘要：

  [ManholeCover] detections=2 confidence: 0.923 0.681

处理完成时，主程序输出保存位置：

  [INFO] output video: /tmp/output_boxed.mp4, frames: 300

也支持原来的位置参数：

  debug_demo 输入视频 输出视频 AXModel 文件

可选插件参数：

  --plugin ./libmanhole_plugin.so

五、处理流程

  1. dlopen 加载 libmanhole_plugin.so。
  2. dlsym 获取 CreateAIModel 和 DestroyAIModel。
  3. 初始化 AX_SYS 和 AX_ENGINE。
  4. OpenCV 读取 MP4 的 BGR 帧。
  5. BGR 转连续 NV12 内存。
  6. AX_SYS_MemAlloc 分配 AX_VIDEO_FRAME_T 使用的设备内存。
  7. 调用插件 Inference。
  8. 将 AI_RESULT_T 的归一化坐标转换为视频像素坐标。
  9. 使用 OpenCV 绘制矩形框、类别和置信度。
 10. 写出新的 MP4 文件。
 11. 释放 AX 内存、模型、插件、AX_ENGINE 和 AX_SYS。

六、输入要求

视频宽度和高度必须是正偶数，例如 1920x1080、1280x720、640x480。程序保持输入视频分辨率和帧率，输出视频不保留音频。

输入视频编码需要当前板端 OpenCV videoio 支持，建议使用常见的 H.264 MP4。

七、接口约定

AI_OBJ_T 中的 x、y、w、h 是 0 到 1 的归一化坐标，表示左上角坐标和宽高。model_manhole_cover.cpp 负责将模型输出还原到原始帧并归一化；src/main.cpp 只负责乘以输出视频宽高后绘制。

井盖模型输出约定为：

  output0 [1,9,8400]
  0..3：cx、cy、w、h
  4..8：good、broke、lose、uncovered、circle

八、常见问题

1. 找不到插件

确认运行目录中存在 libmanhole_plugin.so，并且 LD_LIBRARY_PATH 包含 AX650N 运行库路径。

2. 模型初始化失败

确认模型路径正确，并且 AXModel 与插件的输入输出约定一致。

3. 输出视频没有检测框

查看模型是否返回 nObjSize 大于 0。nObjSize 为 0 表示当前帧没有超过置信度阈值的检测结果。

4. 输出视频无法打开

确认 OpenCV videoio 构建时包含 FFmpeg 或板端支持的 MP4 编码器。

5. 程序直接退出

必须检查 stderr。程序现在会打印参数错误、插件加载错误、模型初始化错误、AX 内存分配错误和推理错误。常见原因是没有传入 --input、--output、--model，或者运行目录不是 bin 导致找不到 libmanhole_plugin.so。
