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

## 3. 写井盖插件

新增文件：

```text
device_side/plugins/model_manhole_cover.cpp
```

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

## 4. 改编译文件

修改 `device_side/CMakeLists.txt`，加入插件库和安装目标：

```cmake
add_library(manhole_cover_plugin SHARED plugins/model_manhole_cover.cpp)
target_link_libraries(manhole_cover_plugin axdl_lib ByteTrack ${OpenCV_LIBS})
install(TARGETS manhole_cover_plugin DESTINATION bin)
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
cp config/streams_config.json /dev/shm/ai_config.json
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
./demo_helmet -c ../config/streams_config.json
```

如果从源码目录运行，注意配置中的 `../models/*.axmodel` 是相对当前工作目录解析的，通常应从 `bin/` 启动。

## 8. 跑通标准

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
3. 板端：demo_helmet 能加载 .axmodel 和插件
4. 实流：OSD 框、类别、置信度和仿真/验证结果基本一致
5. 稳定性：连续推理至少 100 次，记录平均延迟、P95 延迟、峰值内存和异常次数
```
