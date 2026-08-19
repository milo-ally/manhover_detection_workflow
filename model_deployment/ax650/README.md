# AX650N 板端部署工作流

本目录只负责把已经转换并验证通过的 `.axmodel` 接入板端工程
`model_deployment/ax650/manhole_cover_detection`，不在这里重新训练、导出或转换模型。

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

进入 `model_deployment/ax650` 前，`model_convert` 和 `model_val` 必须已经完成。

需要从 `model_convert` 提供：

```text
model_convert/ax650/output/yolo11s-manhole-detection/yolo11s-manhole-detection.axmodel
model_convert/ax650/pulsar2_sim/cli_detect_manhover.py 中确认过的预处理和后处理口径
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
模型文件名       例如 yolo11s-manhole-detection.axmodel
输入规格         U8 NHWC RGB, 640x640, letterbox padding=114
输出规格         output0 [1,9,8400]
类别顺序         good,broke,lose,uncovered,circle
后处理           cx/cy/w/h + 5 类分数，按类别 NMS
上线阈值         建议 conf=0.25, nms=0.45；mAP 验证阈值另按 model_val
验收结论         AXModel 相对 ONNX 的 mAP50/mAP50-95 下降是否可接受
```

只有验收通过的 `.axmodel` 才复制到部署目录 `models/`（模型文件不提交 Git，`*.axmodel` 已被 .gitignore 忽略）：

```bash
mkdir -p model_deployment/ax650/manhole_cover_detection/models
cp model_convert/ax650/output/yolo11s-manhole-detection/yolo11s-manhole-detection.axmodel \
  model_deployment/ax650/manhole_cover_detection/models/
```

## 2. 板端代码链路

板端工程 `manhole_cover_detection` 直接通过 AX Engine 加载 `.axmodel`，不是必须先做
SDK-GEN `model.bin`。它复用了原 device_side 主工程的完整链路：ConfigService、
VideoStreamManager、VideoDemux、IVPS/OSD 和 VENC/RTP，本仓库只编译井盖插件和主程序。

### 2.1 数据流总览（主码流 + AI 流）

```text
输入 RTSP / 本地 H.264
   │
   ▼
VideoDemux（ffmpeg 解封装，AVCC→AnnexB）
   │  latest-frame 缓冲 + 每源专属 worker
   ▼
VDEC（硬件解码 H.264 → NV12；同一输入源共享一个 VDEC 组）
   │
   ├──────────────────────────────┬──────────────────────────────┐
   ▼                              ▼                              ▼
【主码流：输出流】            【AI 流：推理流】               【raw 流（可选）】
 IVPS 1920×1080                IVPS 640×640                   IVPS 1920×1080
 n_osd_rgn=1（挂 OSD 区域）    n_fifo_count=1（回调）           enableAI=false
 po_mediamtx_h264              po_buff_nv12                    （无 OSD，
 （离线: po_venc_h264）        output_func=aiInferenceCallback   端云比对用）
   │                              │ 复制帧后立即返回（不阻塞 IVPS 线程）
   │                              ▼
   │                          AIWorker（每 AI 流一个固定线程）
   │                              │ InferenceManager::run（单/并行/串行 ROI）
   │                              ▼
   │                          插件 Inference()：
   │                            NV12→RGB letterbox(114)
   │                            → AX_ENGINE_RunSync
   │                            → output0 [1,9,8400] 解码 + 分类别 NMS
   │                            → AI_RESULT_T（归一化坐标）
   │                              │
   │                              ▼
   │                      deliverWorkerInferenceResult
   │                              │ updateAIResult(aiStreamId, &result)
   │                              ▼
   │                      OSDAssociatedModel.latestResult
   │                              │ osdUpdateThread 唤醒
   │                              ▼
   │                      DefaultOSDRenderer.render
   │                              │ AX_IVPS_RGN_Update(主码流 region)
   │                              ▼
   │◄──────── 主码流 IVPS 硬件合成检测框（编码前零拷贝）──────────────┘
   ▼
VENC H.264 ──► RTP pusher（UDP/RTP）──► MediaMTX
                                           │
                             主机经 SSH -L 8557 查看
```

要点：主码流与 AI 流是两条独立硬件流水线（共享 VDEC 组、独立 IVPS 组）；AI 结果经
OSD 管理线程（`OSDAssociatedModel`）+ `AX_IVPS_RGN_Update` **跨流叠加**到主码流
IVPS 的 OSD region，在编码前由硬件完成画框合成。

### 2.2 核心链路

```text
src/main.cpp
   解析命令行参数，加载 streams_config.json（-c 指定或默认 config/streams_config.json），
   初始化 AX 系统资源、AX_ENGINE、ConfigService、VideoStreamManager

src/manager/config_service.cpp
   固定监控 /dev/shm/ai_config.json 的热更新；-c 指定的文件用于初始加载 streams 配置
   将 model name/path/conf/nms/plugin 下发给每路流

src/manager/video_stream_manager.cpp
   根据配置创建 VideoStream 和 AIProcessor

src/manager/ai_processor.cpp
   根据配置的 plugin 字段或默认 ./libmanhole_plugin.so 选择插件
   dlopen 插件，调用 CreateAIModel、Init、Inference、Deinit

include/ai_interface.h
   定义插件 ABI：IAIModel、AI_RESULT_T、AI_OBJ_T

plugins/model_manhole_cover.cpp
   井盖模型自己的 AX Engine 初始化、预处理、推理和后处理
```

编译产物为 `bin/demo` 和 `bin/libmanhole_plugin.so`，详见本目录 `FLOW2.md`
和工程内 `readme.txt`。

## 3. 井盖插件

### 3.1 插件实现 `plugins/model_manhole_cover.cpp`

插件实现 `IAIModel`：

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
读取 MANHOLE_CONF_THRESH，未设置时回退 MODEL_CONF_THRESH，默认 0.25
读取 MANHOLE_NMS_THRESH，未设置时回退 MODEL_NMS_THRESH，默认 0.45
读取 .axmodel 文件内容
AX_ENGINE_CreateHandle
AX_ENGINE_GetIOInfo
AX_SYS_MemAlloc 分配 640 * 640 * 3（U8 NHWC RGB）输入 buffer 和按 IOInfo 分配输出 buffer
```

`Inference()` 做：

```text
AX_VIDEO_FRAME_T NV12 -> RGB
RGB letterbox 到 640x640，padding=114（与 pulsar2_sim / model_val 口径一致）
拷贝 U8 NHWC RGB 到输入 buffer，并 flush cache
AX_ENGINE_RunSync
读取 output0 [1,9,8400]
按 [channels, anchors] 解析：0..3=cx/cy/w/h，4..8=good/broke/lose/uncovered/circle 分数
按 conf 过滤，按类别 NMS
去 letterbox padding，除 gain，还原原图坐标
写入 AI_RESULT_T，坐标为归一化 x/y/w/h，label 为五分类名称
```

不要直接复制 `model_human_detection.cpp` 或 `model_smoke_fire.cpp` 的 DFL 多头解码逻辑；
井盖模型输出不是 `4*REG_MAX + class_num` 多头格式。

`Deinit()` 释放：

```text
AX_ENGINE handle
AX_SYS_MemAlloc 申请的输入/输出内存
pStride 数组
```

### 3.2 编译文件 `CMakeLists.txt`

工程只编译井盖插件和主程序两个目标：

```cmake
add_library(manhole_plugin SHARED plugins/model_manhole_cover.cpp)
target_link_libraries(manhole_plugin ax_engine ax_sys ${OpenCV_LIBS})

add_executable(demo ${SRC_APP} ${SRC_LIST_LIBS})
set(AX_LIBS ax_sys ax_vdec ax_ivps ax_venc ax_vo ax_mipi ax_3a ax_proton ax_nt_stream ax_nt_ctrl ax_engine ax_interpreter)
set(DEMUX_LIBS avformat avcodec avutil swresample)
set(SYS_LIBS dl pthread m rt stdc++fs)
target_link_libraries(demo ${AX_LIBS} ${DEMUX_LIBS} ${OpenCV_LIBS} ${SYS_LIBS})

install(TARGETS demo DESTINATION bin)
install(TARGETS manhole_plugin DESTINATION bin)
```

产物：`bin/demo`（主程序）和 `bin/libmanhole_plugin.so`（井盖插件）。

### 3.3 插件加载逻辑（ai_processor.cpp）

`AIProcessor::loadModel()` 不再按 modelName 关键字分支到各平台插件库，当前实现：

```text
优先使用配置中 models[].plugin 或 stream 级 plugin 字段指定的插件路径
未配置 plugin 字段时，默认加载 ./libmanhole_plugin.so
dlopen 失败时回退尝试 ./bin/libmanhole_plugin.so
```

`AIProcessor::applyModelParamsToEnv()` 把配置阈值透传为环境变量：

```text
conf_threshold -> MODEL_CONF_THRESH
nms_threshold  -> MODEL_NMS_THRESH
```

插件侧 `Init()` 优先读 `MANHOLE_CONF_THRESH` / `MANHOLE_NMS_THRESH`，未设置时回退到
`MODEL_CONF_THRESH` / `MODEL_NMS_THRESH`，因此配置文件中的阈值会真实作用于 AXModel
后处理，而不是只显示在配置里。

### 3.4 模型路径解析（config_service.cpp）

`getModelPath()` 当前实现：

```text
模型名包含 "/" 或 "\\" 时按完整路径返回
模型名以 .axmodel 结尾时返回 ../models/<模型名>
其他逻辑名统一映射到 ../models/yolo11s-manhole-detection.axmodel
```

也可以不改代码，在 `streams_config.json` 的 `models[].path` 里直接写完整路径。

## 4. 配置运行

推荐使用 `models` 数组配置，写入 `config/streams_config.json`：

```json
{
  "streams": [
    {
      "stream_id": 1,
      "input_source": "/root/kaohe_6076/test.mp4",
      "input_codec": "h264",
      "conf_thres": 0.25,
      "nms_thres": 0.45,
      "output_width": 1920,
      "output_height": 1080,
      "fps": 30,
      "enable_ai": true,
      "ai_output_width": 640,
      "ai_output_height": 640,
      "ai_fps": 15,
      "plugin": "/root/kaohe_6076/device_side/bin/libmanhole_plugin.so",
      "models": [
        {
          "name": "manhole_cover",
          "path": "/root/kaohe_6076/models/yolo11s-manhole-detection.axmodel",
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

支持的关键字段：

```text
stream 级：
  input_source      输入文件路径或 RTSP URL
  input_codec       h264（当前只支持 H.264 输入，H.265/HEVC 不支持）
  conf_thres/nms_thres          流级默认阈值
  enable_ai         是否启用 AI（false 时主码流不挂 OSD）
  output_width/output_height/fps    主码流 IVPS 输出尺寸和帧率
  ai_output_width/ai_output_height/ai_fps   AI 流 IVPS 输出尺寸和帧率
  plugin            插件 .so 路径（可放在 models[] 里按模型指定）
  models[]          每项：name/path/conf_threshold/nms_threshold/plugin/params

global_settings 级：
  mediamtx_host/mediamtx_port    MediaMTX RTP 端点
  default_model/default_conf_thres/default_nms_thres
  enable_raw_stream              端云精度比对的 raw 主码流开关（默认 false）
```

运行时热更新文件固定是 `/dev/shm/ai_config.json`：

```bash
cp config/streams_config.json /dev/shm/ai_config.json
```

ConfigService 监控 `/dev/shm/ai_config.json`，文件变化时动态下发配置。

## 5. 编译和启动

在 AX650N 板端或对应交叉编译环境中：

```bash
cd model_deployment/ax650/manhole_cover_detection
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cmake --install build
```

前置条件：

```text
msp_sdk 已放入工程目录（包含 include/ax_sys_api.h 等 AX650N BSP SDK 输出）
OpenCV 和 AX SDK 运行库可链接
models/yolo11s-manhole-detection.axmodel 已放置（模型不提交 Git）
```

在线模式（RTSP/文件输入，H.264/RTP 推送到 MediaMTX）：

```bash
cd model_deployment/ax650/manhole_cover_detection/bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m stream
```

离线模式（输入为本地文件，输出 MP4 或裸 H.264）：

```bash
./demo -c ../config/streams_config.json -m offline -o /tmp/output.mp4
```

如果从源码目录运行，注意配置中的相对路径是相对当前工作目录解析的，通常应从
`bin/` 启动，或把 `models[].path`、`plugin` 写成完整路径。

## 6. 运行模式说明

主程序只支持 `-m offline` 和 `-m stream` 两种模式，参数全部使用选项名：

```text
-c <file>       streams 配置 JSON（默认 config/streams_config.json）
-m <mode>       offline 或 stream
-o <file>       offline 输出路径，.mp4 结尾自动封装 MP4，否则输出裸 H.264
-h              帮助
```

### 6.1 离线模式（-m offline）

```text
本地文件 -> VDEC 解码 -> IVPS 缩放 -> AI 推理（NV12 -> RGB -> letterbox -> AXModel）
   -> OSD 叠加检测框 -> VENC H.264 -> 写入 raw H.264
   -> 输出以 .mp4 结尾时自动调用 ffmpeg 封装为 MP4
```

只支持 H.264 输入；H.265/HEVC 输入不支持，需先转成 H.264。

### 6.2 在线模式（-m stream）

```text
RTSP/文件输入 -> VDEC -> IVPS -> AI 推理 -> OSD 叠加检测框
   -> VENC H.264 -> RTP 推送到 MediaMTX（默认 127.0.0.1:8000）
```

输出走板端原生 VENC/RTP 链路，不经过 FFmpeg 管道。程序启动时会提示：
`Note: Configure MediaMTX with: source: udp+rtp://0.0.0.0:8000`，需要在板端 MediaMTX
配置中把对应路径的 source 指到该 RTP 端口（或用 `--mediamtx` / `--mediamtx-host`
`--mediamtx-port` 指定实际端点）。不要把 RK3588 文档中的 `h264_rkmpp` 命令复制到这里。

## 7. 检测框和 OSD 显示说明

井盖插件本身不直接在视频帧上画框。它在 `Inference()` 中完成推理和后处理，将检测结果
写入 `AI_RESULT_T`，包括归一化的 `x/y/w/h`、类别和置信度。

板端运行时，`VideoStreamManager` 为每路输入创建两条逻辑链路：

```text
输入 RTSP/文件
   ├── AI 分支：送入 AIProcessor，得到 AI_RESULT_T
   └── 主输出分支：VENC H.264 -> RTP -> MediaMTX
                    ↑
          VideoStreamManager 将 AI_RESULT_T
          交给 DefaultOSDRenderer
          通过 AX_IVPS_RGN_Update() 叠加矩形框和标签
```

因此，检测框应当在主 MediaMTX 输出流中查看，而不是在 AI 推理分支或原始输入流中查看。
配置中的 `enable_ai: true` 只表示启用推理；要看到框，还必须完成 OSD 初始化，并且当前
帧至少检测到一个目标（`AI_RESULT_T.nObjSize > 0`）。井盖模型当前没有专用 OSD Renderer
（`AIProcessor::getOSDRenderer()` 返回 nullptr），会使用通用 `DefaultOSDRenderer` 绘制
矩形框、类别和置信度。

启动前确认：

```text
manhole_cover_detection/models/yolo11s-manhole-detection.axmodel 存在
bin/demo 存在
bin/libmanhole_plugin.so 存在
LD_LIBRARY_PATH 包含 bin、/soc/lib 和 /usr/lib
未设置 AX_DISABLE_OSD=1（该环境变量可强制关闭 OSD）
```

建议按以下日志确认画框链路：

```text
[AIProcessor] Loading plugin: ./libmanhole_plugin.so
[AIProcessor] Model initialized successfully
[OSD] Using default OSD renderer for AI stream 1
[OSD] Initialized OSD management for AI stream 1, N pipelines
[OSD] Updated AI result for stream 1: nObjSize=...
[OSD] AX_IVPS_RGN_Update success: pipeid=...
```

其中，只有看到 `nObjSize > 0` 的检测结果时才会实际更新目标框；如果模型运行正常但
没有检测到目标，视频中不会出现框，这不代表 OSD 链路失效。

## 8. 跑通标准

日志必须看到：

```text
[AIProcessor] Loading plugin: ./libmanhole_plugin.so
[AIProcessor] Model initialized successfully
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
1. 单张图片：model_convert/ax650/pulsar2_sim 结果正常
2. 精度：model_val 中 ONNX 和 AXModel mAP 对比通过
3. 板端：demo 能加载 .axmodel 和 libmanhole_plugin.so
4. 实流：在主 MediaMTX 输出流中确认 OSD 框、类别、置信度和仿真/验证结果基本一致
5. 稳定性：连续推理至少 100 次，记录平均延迟、P95 延迟、峰值内存和异常次数
```

如果实流中看不到框，按以下顺序排查：

```text
1. 确认查看的是主 MediaMTX 输出流，而不是原始输入流或 AI 分支
2. 确认日志中出现 [OSD] 初始化和 AX_IVPS_RGN_Update success
3. 确认 AI_RESULT_T.nObjSize > 0，排除当前画面确实没有达到阈值的目标
4. 确认没有设置 AX_DISABLE_OSD=1
5. 确认模型路径、插件路径和 LD_LIBRARY_PATH 正确
```

## 9. 最短 SOP

已写入的代码和作用：

```text
plugins/model_manhole_cover.cpp
  新增 ManholeCoverModel : IAIModel。
  Init(): 读取 MANHOLE_CONF_THRESH/MANHOLE_NMS_THRESH（回退 MODEL_*，默认 0.25/0.45），
  AX_ENGINE_CreateHandle，AX_ENGINE_GetIOInfo，分配 640*640*3 U8 NHWC RGB 输入 buffer 和输出 buffer。
  Inference(): NV12->RGB，letterbox(640,pad=114)，AX_ENGINE_RunSync，解码 output0 [1,9,8400]。
  后处理: output0 按 [channels,anchors] 解析；0..3=cx/cy/w/h，4..8=good/broke/lose/uncovered/circle 分数；
  按类别 NMS；还原 letterbox；写 AI_RESULT_T（归一化坐标 + 类别名 + 置信度）。
  Deinit(): 释放 AX_ENGINE handle、AX_SYS_MemAlloc 内存和 pStride。
  导出: CreateAIModel() / DestroyAIModel()，供 AIProcessor dlopen/dlsym 调用。

CMakeLists.txt
  新增 add_library(manhole_plugin SHARED plugins/model_manhole_cover.cpp)。
  新增 target_link_libraries(manhole_plugin ax_engine ax_sys ${OpenCV_LIBS})。
  主程序目标为 demo，install 安装 demo 和 manhole_plugin 到 bin/。

src/manager/ai_processor.cpp
  loadModel(): 配置 plugin 字段优先，未配置默认 ./libmanhole_plugin.so，失败回退 ./bin/libmanhole_plugin.so。
  applyModelParamsToEnv(): 将 conf_threshold/nms_threshold 写入 MODEL_CONF_THRESH/MODEL_NMS_THRESH。

src/manager/config_service.cpp
  getModelPath(): 非路径模型名统一映射到 ../models/yolo11s-manhole-detection.axmodel。

config/streams_config.json
  井盖模型运行配置；models[0].name=manhole_cover，path 指向 yolo11s-manhole-detection.axmodel。
```

需要放置的文件：

```bash
mkdir -p model_deployment/ax650/manhole_cover_detection/models
cp model_convert/ax650/output/yolo11s-manhole-detection/yolo11s-manhole-detection.axmodel \
  model_deployment/ax650/manhole_cover_detection/models/
```

编译：

```bash
cd model_deployment/ax650/manhole_cover_detection
cmake -S . -B build
cmake --build build -j$(nproc)
cmake --install build
```

运行：

```bash
cd model_deployment/ax650/manhole_cover_detection/bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m stream
```

预期产物：

```text
bin/demo
bin/libmanhole_plugin.so
```

## 10. 离线 MP4 验证（考核用）

`demo` 的离线模式可以直接生成带框 MP4，用于先确认模型、预处理、后处理和画框链路，
不需要 MediaMTX：

```bash
cd model_deployment/ax650/manhole_cover_detection/bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m offline -o /tmp/output_boxed.mp4
```

把 `streams_config.json` 的 `input_source` 指向本地 H.264 视频（例如
`/root/kaohe_6076/test.mp4`）。`-o` 以 `.mp4` 结尾时，程序先写裸 H.264 再自动调用
FFmpeg 封装为 MP4；`-o` 以 `.h264` 结尾时直接输出裸 H.264 码流。

处理完成后检查：

```bash
ffprobe -v error -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 /tmp/output_boxed.mp4
```

日志中应看到：

```text
[Main] Offline MP4 generated: /tmp/output_boxed.mp4
[ManholeCover] detections=... confidence: ...
```

如果 `detections=0`，输出视频会生成但不会出现检测框；此时应先查看井盖插件的
`[ManholeCover]` 日志，确认模型是否加载成功以及 `AI_RESULT_T.nObjSize` 是否大于 0，
或先降低配置里的 `conf_threshold` 验证。
