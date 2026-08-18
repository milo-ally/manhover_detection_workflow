# AX650N 板端部署工作流

本目录只负责把已经转换并验证通过的 `.axmodel` 接入 `device_side` 板端工程，不在这里重新训练、导出或转换模型。

当前井盖五分类模型约定：

```text
classes        good, broke, lose, uncovered, circle
ONNX input     images [1,3,640,640] FP32 NCHW RGB / 255
ONNX output    output0 [1,9,8400] FP32
AXModel input  images [1,640,640,3] U8 NHWC RGB
AXModel output output0 [1,9,8400] FP32
```

类别 ID 固定为：`0=good`、`1=broke`、`2=lose`、`3=uncovered`、`4=circle`。

## 1. 前置工作和交付信息

进入 `model_deployment` 前，`model_convert` 和 `model_val` 必须已经完成。

需要从 `model_convert` 提供：

```text
model_convert/output/<model-name>/<model-name>.axmodel
model_convert/pulsar2_sim/cli_detect_manhover.py 中确认过的预处理和后处理口径
```

需要从 `model_val` 提供：

```text
model_val/runs/*_onnx.txt
model_val/runs/*_axmodel.txt
model_val/runs/*_onnx_predictions.jsonl
model_val/runs/*_axmodel_predictions.jsonl
```

最后部署步骤必须明确拿到这些信息：

```text
模型文件名       例如 manhole-cover-yolo11s-production.axmodel
输入规格         U8 NHWC RGB, 640x640, letterbox padding=114
输出规格         output0 [1,9,8400]
类别顺序         good,broke,lose,uncovered,circle
后处理           cx/cy/w/h + 5 类分数，按类别 NMS
上线阈值         建议 conf=0.25, nms=0.45；mAP 验证阈值另按 model_val
验收结论         AXModel 相对 ONNX 的 mAP50/mAP50-95 下降是否可接受
```

只有验收通过的 `.axmodel` 才复制到 `device_side/models/`：

```bash
cd model_deployment/device_side
cp ../../model_convert/output/manhole-cover-yolo11s-production/manhole-cover-yolo11s-production.axmodel \
  models/manhole-cover-yolo11s-production.axmodel
```

## 2. device_side 代码链路

`device_side` 主程序当前直接通过 AX Engine 加载 `.axmodel`，不是必须先做 SDK-GEN `model.bin`。

核心链路：

```text
src/main.cpp
  初始化 AX 系统资源、AX_ENGINE、ConfigService、VideoStreamManager

src/manager/config_service.cpp
  解析 /dev/shm/ai_config.json 或 -c 指定的 streams_config.json
  将 model name/path/conf/nms/params 下发给每路流

src/manager/video_stream_manager.cpp
  根据配置创建 VideoStream 和 AIProcessor

src/manager/ai_processor.cpp
  根据 modelName/modelPath 选择插件 .so
  dlopen 插件，调用 CreateAIModel、Init、Inference、Deinit

include/ai_interface.h
  定义插件 ABI：IAIModel、AI_RESULT_T、AI_OBJ_T

plugins/*.cpp
  每个模型自己的 AX Engine 初始化、预处理、推理和后处理
```

## 3. 示例

### 3.1 写井盖插件

**新增文件：`device_side/plugins/model_manhole_cover.cpp`**

插件必须实现 `IAIModel`：

```cpp
class ManholeCoverModel : public IAIModel {
public:
    int Init(const char* model_path) override;
    void GetInputSize(int* w, int* h) override;
    int Inference(const AX_VIDEO_FRAME_T* pFrame, AI_RESULT_T* pResult) override;
    int Deinit() override;
};

extern "C" {
    IAIModel* CreateAIModel() { return new ManholeCoverModel(); }
    void DestroyAIModel(IAIModel* p) { delete p; }
}
```

`Init()` 做：

```text
读取 .axmodel
AX_ENGINE_CreateHandle
AX_ENGINE_GetIOInfo
分配输入 buffer：640 * 640 * 3, U8 NHWC RGB
按 IOInfo 分配 output buffer
读取 MANHOLE_CONF_THRESH / MANHOLE_NMS_THRESH，默认 0.25 / 0.45
```

`Inference()` 做：

```text
AX_VIDEO_FRAME_T NV12 -> RGB
RGB letterbox 到 640x640，padding=114
拷贝 U8 NHWC RGB 到输入 buffer，并 flush cache
AX_ENGINE_RunSync
读取 output0 [1,9,8400]
转成 [8400,9]
前 4 维按 cx/cy/w/h 解码，后 5 维取最大类分数
按 conf 过滤，按类别 NMS
去 letterbox padding，除 gain，还原原图坐标
写入 AI_RESULT_T，坐标为归一化 x/y/w/h，label 为五分类名称
```

不要直接复制 `model_human_detection.cpp` 或 `model_smoke_fire.cpp` 的 DFL 多头解码逻辑；井盖模型输出不是 `4*REG_MAX + class_num` 多头格式。

`Deinit()` 释放：

```text
AX_ENGINE handle
AX_SYS_MemAlloc 申请的输入/输出内存
pStride 数组
```

### 3.2 改编译文件

**修改 `device_side/CMakeLists.txt`，加入插件库和安装目标：**

```cmake
# 井盖检测插件
add_library(manhole_cover_plugin SHARED plugins/model_manhole_cover.cpp)
target_link_libraries(manhole_cover_plugin axdl_lib ByteTrack ${OpenCV_LIBS})
```

确保井盖插件出现在 `install` 中, 主程序目标已改成通用名 `device_ai_demo`：

```cmake
add_executable(device_ai_demo ${SRC_APP} ${SRC_LIST_LIBS})
...
install(TARGETS manhole_cover_plugin DESTINATION bin) # make sure
install(TARGETS device_ai_demo DESTINATION bin) # fixed
```

对应链接目标：

```cmake
target_link_libraries(device_ai_demo
    axdl_lib
    ByteTrack
    ${AX_LIBS}
    ${DEMUX_LIBS}
    ${OpenCV_LIBS}
    ${SYS_LIBS}
) # make sure
```

如果增加了专用 OSD Renderer，`src/osd_renderers/*.cpp` 已被 glob 收集，但仍需要在 `ai_processor.cpp` include 对应头文件并创建 Renderer。

## 5. 改插件注册逻辑

修改 `device_side/src/manager/ai_processor.cpp`。

在 `AIProcessor::loadModel()` 的插件选择分支中加入：

```cpp
} else if (pluginHint.find("manhole") != std::string::npos ||
           pluginHint.find("cover") != std::string::npos) {
    pluginPath = "./libmanhole_cover_plugin.so";
    ALOGN("[AIProcessor] Using manhole cover plugin for model: %s", modelPath.c_str());
```

在 `AIProcessor::applyModelParamsToEnv()` 中加入阈值透传：

```cpp
} else if (lowerHint.find("manhole") != std::string::npos ||
           lowerHint.find("cover") != std::string::npos) {
    setFloat("conf_threshold", "MANHOLE_CONF_THRESH");
    setFloat("nms_threshold", "MANHOLE_NMS_THRESH");
```

如果希望配置只写 `"name": "manhole_cover"`，再改 `device_side/src/manager/config_service.cpp` 的 `getModelPath()`：

```cpp
if (modelPath.find("manhole") != std::string::npos ||
    modelPath.find("cover") != std::string::npos) {
    return "../models/manhole-cover-yolo11s-production.axmodel";
}
```

也可以不改 `getModelPath()`，在配置里直接写完整 `path`。

## 6. 配置运行

推荐使用 `models` 数组配置，写入 `device_side/config/streams_config.json`：

```json
{
  "streams": [
    {
      "stream_id": 1,
      "input_source": "rtsp://172.19.31.71:8554/lygstream1",
      "conf_thres": 0.25,
      "nms_thres": 0.45,
      "output_width": 1920,
      "output_height": 1080,
      "fps": 30,
      "enable_ai": true,
      "ai_output_width": 640,
      "ai_output_height": 640,
      "ai_fps": 30,
      "models": [
        {
          "name": "manhole_cover",
          "path": "../models/manhole-cover-yolo11s-production.axmodel",
          "conf_threshold": 0.25,
          "nms_threshold": 0.45
        }
      ]
    }
  ],
  "global_settings": {
    "mediamtx_host": "127.0.0.1",
    "mediamtx_port": "8000",
    "default_model": "manhole_cover",
    "default_conf_thres": 0.25,
    "default_nms_thres": 0.45,
    "enable_raw_stream": false
  }
}
```

运行时热更新文件是 `/dev/shm/ai_config.json`：

```bash
cp config/streams_config_manhole.json /dev/shm/ai_config.json
```

## 7. 编译和启动

在 AX650N 板端或对应交叉编译环境中：

```bash
cd /root/device_side
mkdir -p build
cd build
cmake ..
make -j$(nproc)
make install
```

前置条件：

```text
device_side/msp_sdk 已放入 AX650N BSP SDK 输出
msp_sdk/include/ax_sys_api.h 存在
OpenCV 和 AX SDK 运行库可链接
新插件已加入 CMakeLists.txt
```

启动：

```bash
cd /root/device_side/bin
export LD_LIBRARY_PATH=$PWD:/root/device_side/bin:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./device_ai_demo -c ../config/streams_config_manhole.json
```

如果从源码目录运行，注意配置中的 `../models/*.axmodel` 是相对当前工作目录解析的，通常应从 `bin/` 启动。

## 8. 检测框和 OSD 显示说明

井盖插件本身不直接在视频帧上画框。它在 `Inference()` 中完成推理和后处理，将检测结果写入 `AI_RESULT_T`，包括归一化的 `x/y/w/h`、类别和置信度。

真实板端运行时，`VideoStreamManager` 为每路输入创建两条逻辑链路：

```text
输入 RTSP
   ├── AI 分支：送入 AIProcessor，得到 AI_RESULT_T
   └── 主输出分支：送入 MediaMTX
                    ↑
          VideoStreamManager 将 AI_RESULT_T
          交给 DefaultOSDRenderer
          通过 AX_IVPS_RGN_Update() 叠加矩形框和标签
```

因此，检测框应当在主 MediaMTX 输出流中查看，而不是在 AI 推理分支或原始输入流中查看。配置中的 `enable_ai: true` 只表示启用推理；要看到框，还必须完成 OSD 初始化，并且当前帧至少检测到一个目标（`AI_RESULT_T.nObjSize > 0`）。井盖模型当前没有专用 OSD Renderer，会使用通用 `DefaultOSDRenderer` 绘制矩形框、类别和置信度。

`run_demo.py` 不能用于验证画框。它只是读取 `/dev/shm/ai_config.json` 并模拟配置处理，不加载 `.axmodel`、不执行 AX Engine 推理，也不启动视频 OSD 或 MediaMTX 输出。画框验证必须使用真实的 `device_ai_demo` 和板端运行库。

启动前确认：

```text
device_side/models/manhole-cover-yolo11s-production.axmodel 存在
device_side/bin/device_ai_demo 存在
device_side/bin/libmanhole_cover_plugin.so 存在
LD_LIBRARY_PATH 包含 bin、/soc/lib 和 /usr/lib
未设置 AX_DISABLE_OSD=1
```

建议按以下日志确认画框链路：

```text
Using manhole cover plugin
Model initialized successfully
Using default OSD renderer
Initialized OSD management
Updated AI result
AX_IVPS_RGN_Update success
```

其中，只有看到 `nObjSize > 0` 的检测结果时才会实际更新目标框；如果模型运行正常但没有检测到目标，视频中不会出现框，这不代表 OSD 链路失效。

## 9. 跑通标准

日志必须看到：

```text
Using manhole cover plugin
Model initialized successfully
```

不得出现：

```text
dlopen failed
Model file does not exist
AX_ENGINE_RunSync failed
output0 contains NaN/Inf
```

验收顺序：

```text
1. 单张图片：model_convert/pulsar2_sim 结果正常
2. 精度：model_val 中 ONNX 和 AXModel mAP 对比通过
3. 板端：device_ai_demo 能加载 .axmodel 和插件
4. 实流：在主 MediaMTX 输出流中确认 OSD 框、类别、置信度和仿真/验证结果基本一致
5. 稳定性：连续推理至少 100 次，记录平均延迟、P95 延迟、峰值内存和异常次数
```

如果实流中看不到框，按以下顺序排查：

```text
1. 确认查看的是主 MediaMTX 输出流，而不是原始输入流或 AI 分支
2. 确认日志中出现 OSD 初始化和 AX_IVPS_RGN_Update success
3. 确认 AI_RESULT_T.nObjSize > 0，排除当前画面确实没有达到阈值的目标
4. 确认没有设置 AX_DISABLE_OSD=1
5. 确认模型路径、插件路径和 LD_LIBRARY_PATH 正确
```

## 10. 最短 SOP

已写入的代码和作用：

```text
device_side/plugins/model_manhole_cover.cpp
  新增 ManholeCoverModel : IAIModel。
  Init(): 读取 .axmodel，AX_ENGINE_CreateHandle，AX_ENGINE_GetIOInfo，分配 640*640*3 U8 NHWC RGB 输入 buffer 和输出 buffer。
  Inference(): NV12->RGB，letterbox(640,pad=114)，AX_ENGINE_RunSync，解码 output0 [1,9,8400]。
  后处理: output0 按 [channels,anchors] 解析；0..3=cx/cy/w/h，4..8=good/broke/lose/uncovered/circle 分数；按类别 NMS；还原 letterbox；写 AI_RESULT_T。
  Deinit(): 释放 AX_ENGINE handle、AX_SYS_MemAlloc 内存和 pStride。
  导出: CreateAIModel() / DestroyAIModel()，供 AIProcessor dlopen/dlsym 调用。

device_side/CMakeLists.txt
  新增 add_library(manhole_cover_plugin SHARED plugins/model_manhole_cover.cpp)。
  新增 target_link_libraries(manhole_cover_plugin axdl_lib ByteTrack ${OpenCV_LIBS})。
  新增 install(TARGETS manhole_cover_plugin DESTINATION bin)。
  主程序目标从 demo_helmet 改为 device_ai_demo，并同步修改 target_link_libraries / install。

device_side/src/manager/ai_processor.cpp
  loadModel(): modelName/modelPath 包含 manhole 或 cover 时加载 ./libmanhole_cover_plugin.so。
  applyModelParamsToEnv(): 将 conf_threshold/nms_threshold 写入 MANHOLE_CONF_THRESH/MANHOLE_NMS_THRESH。

device_side/src/manager/config_service.cpp
  getModelPath(): name=manhole_cover 或包含 manhole/cover 时默认映射到 ../models/manhole-cover-yolo11s-production.axmodel。

device_side/config/streams_config_manhole.json
  新增井盖模型单路运行配置；models[0].name=manhole_cover，path 指向 ../models/manhole-cover-yolo11s-production.axmodel。
```

需要放置的文件：

```bash
cd model_deployment/device_side
cp ../../model_convert/output/manhole-cover-yolo11s-production/manhole-cover-yolo11s-production.axmodel \
  models/manhole-cover-yolo11s-production.axmodel
cp config/streams_config_manhole.json /dev/shm/ai_config.json
```

编译：

```bash
cd model_deployment/device_side
cmake -S . -B build
cmake --build build -j$(nproc)
cmake --install build
```

运行：

```bash
cd model_deployment/device_side/bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./device_ai_demo -c ../config/streams_config_manhole.json
```

预期产物：

```text
bin/device_ai_demo
bin/libmanhole_cover_plugin.so
bin/libaxdl_lib.so
bin/libByteTrack.so
```

## 11. 临时 MP4 文件推理模式（考核用）

`device_ai_demo` 额外提供了一个临时的 MP4 离线推理模式。该模式读取 MP4 文件，调用现有井盖检测插件，将 `AI_RESULT_T` 中的检测框、类别和置信度绘制到视频帧，然后输出新的 MP4 文件。它不启动 RTSP、MediaMTX 或实时视频流。

README 中对应的临时代码均使用以下注释标记，考核完成后可按标记整体删除：

```cpp
// ===== TEMP_MP4_INFERENCE_BEGIN
// ===== TEMP_MP4_INFERENCE_END
```

使用前需要重新编译 `device_ai_demo`，并确认输入 MP4、模型文件和输出目录在板端可访问：

```bash
cd /root/device_side
cmake -S . -B build
cmake --build build -j$(nproc)
cmake --install build

cd bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./device_ai_demo --mp4-in test.mp4 --mp4-out test_boxed.mp4 --mp4-model ../models/yolo11s-manhole-detection.axmodel
```

处理完成后检查：

```text
/tmp/manhole_output_boxed.mp4
```

日志中应看到：

```text
[TEMP_MP4] processed frame=30 detections=...
[TEMP_MP4] completed: frames=... output=/tmp/manhole_output_boxed.mp4
```

如果 `detections=0`，输出视频会生成但不会出现检测框；此时应先查看井盖插件的 `DEBUG` 日志，确认模型是否加载成功以及 `AI_RESULT_T.nObjSize` 是否大于 0。
