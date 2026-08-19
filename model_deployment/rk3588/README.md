# RK3588 板端部署工作流

本目录只负责把已经转换并验证通过的 `.rknn` 接入板端工程
`model_deployment/rk3588/manhole_cover_detection`，不在这里重新训练、导出或转换模型。

当前井盖五分类模型约定：

```text
classes        good, broke, lose, uncovered, circle
ONNX input     images [1,3,640,640] FP32 NCHW RGB / 255
ONNX output    output0 [1,9,8400] FP32
RKNN input     RGB uint8, 640x640, letterbox padding=114（NCHW/NHWC 由运行时查询）
RKNN output    output0 [1,9,8400] FP32（want_float=1）
```

类别 ID 固定为：`0=good`、`1=broke`、`2=lose`、`3=uncovered`、`4=circle`。

工程与 AX650 板端工程（`model_deployment/ax650/manhole_cover_detection`）**架构完全同构**：
类名/方法名一致、`streams_config.json` 字段一致、CLI 一致、插件 ABI 一致；仅底层硬件实现
不同——AX650 走 VDEC/IVPS/VENC 硬件流水线，RK3588 用 OpenCV 解码 + FFmpeg
`h264_rkmpp` 输出 + RKNN 推理。

## 1. 前置工作和交付信息

进入 `model_deployment/rk3588` 前，`model_convert` 和 `model_val` 必须已经完成。

需要从 `model_convert` 提供：

```text
model_convert/rk3588/output/manhole-cover-yolo11s-production.rknn
```

需要从 `model_val` 提供：

```text
model_val/rk3588/runs/*_onnx.txt
model_val/rk3588/runs/*_rknn.txt
model_val/rk3588/runs/*_onnx_predictions.jsonl
model_val/rk3588/runs/*_rknn_predictions.jsonl
```

最后部署步骤必须明确拿到这些信息：

```text
模型文件名       例如 manhole-cover-yolo11s-production.rknn
输入规格         RGB uint8, 640x640, letterbox padding=114
输出规格         output0 [1,9,8400]
类别顺序         good,broke,lose,uncovered,circle
后处理           cx/cy/w/h + 5 类分数，按类别 NMS
上线阈值         建议 conf=0.25, nms=0.45；mAP 验证阈值另按 model_val
验收结论         RKNN 相对 ONNX 的 mAP50/mAP50-95 下降是否可接受
```

只有验收通过的 `.rknn` 才复制到部署目录 `models/`（模型文件不提交 Git，`*.rknn` 不随
仓库保存）：

```bash
mkdir -p model_deployment/rk3588/manhole_cover_detection/models
cp model_convert/rk3588/output/manhole-cover-yolo11s-production.rknn \
  model_deployment/rk3588/manhole_cover_detection/models/
```

RKNN 运行时 `rknpu2/`（`include/rknn_api.h` + `lib/librknnrt.so`，aarch64）**已随仓库
提交**，无需额外准备即可编译。**MPP/RGA 开发头文件不随仓库提交**，需按工程内
`SOP.md` §1.1 或 `readme.txt`「MPP/RGA 依赖」准备（apt 或 GitHub 固定版本下载，
含完整 URL）。RKNN 运行时如需核对/重新获取（固定来自 Rockchip `rknn_model_zoo`
提交 `bad6c7334531becaf90a561988519b7bec34d0ab`，详见工程内 `SOP.md` §1 与
`readme.txt`），下载地址：

```bash
# ① rknn_api.h
curl -L -o rknpu2/include/rknn_api.h \
  "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/bad6c7334531becaf90a561988519b7bec34d0ab/3rdparty/rknpu2/include/rknn_api.h"
# ② librknnrt.so（aarch64）
curl -L -o rknpu2/lib/librknnrt.so \
  "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/bad6c7334531becaf90a561988519b7bec34d0ab/3rdparty/rknpu2/Linux/aarch64/librknnrt.so"
file rknpu2/lib/librknnrt.so   # 必须显示 ARM aarch64
```

## 2. 板端代码链路

板端工程 `manhole_cover_detection` 直接通过 RKNN Runtime 加载 `.rknn`。它与 AX650
同构：ConfigService（热更新）、VideoStreamManager（多流）、AIProcessor（dlopen 插件）、
InferenceManager（多模型调度）、OSDRenderer（画框）；解码/输出使用 OpenCV 与 FFmpeg。

### 2.1 数据流总览（主码流 = AI 流，单软件流水线）

```text
输入 RTSP / 本地视频（H.264）
   │
   ▼
H264Demux（FFmpeg libavformat，RTSP 强制 tcp；h264_mp4toannexb）
   │
   ▼
RkDecoder（MPP mpi_dec，H.264→NV12）◄────── 对应 AX650 VDEC
   │
   ├───────────────────────────────┬──────────────────────────────┐
   ▼                               ▼                              ▼
【主码流：输出流】              【AI 流：推理流】
 RGA NV12→BGR(输出尺寸)           FrameBroker 取最新 NV12 帧
   │  （对应 AX650 IVPS）            │
   ▼                                ▼
 CPU 画框（OSD 降级：            RGA NV12→BGR(640x640)
  拉取 SharedAIResult，              │
  DefaultOSDRenderer 绘制）          ▼
   │                              插件 libmanhole_plugin.so
   ▼                                （RKNN 推理 → AI_RESULT_T）
 RGA BGR→NV12(输出尺寸)              │
   │                                ▼
   ▼                            SharedAIResult.set()
 RkEncoder（MPP mpi_enc）           （对应 AX650 OSDAssociatedModel）
   │  （对应 AX650 VENC）            │
   ▼                                │
 rtp_pusher → MediaMTX(RTP)  ◄──────┘（主码流编码线程读取，跨流叠加画框）
 或离线 raw H.264 文件 + ffmpeg 封装 MP4
```

要点：与 AX650 完全一致，**每路输入拆主码流 + AI 流两条流水线**——主码流负责
`解封装→MPP 解码→RGA 缩放→画框→MPP 编码→RTP/文件`，AI 流负责
`RGA 640→RKNN 推理→SharedAIResult`；OSD 通过主码流编码线程拉取 AI 结果并 CPU 画框
（IVPS OSD region 的降级实现），主码流与 AI 流共享解码帧（FrameBroker，latest-frame
语义，对应 AX650 共享 VDEC 组）。

### 2.2 核心链路

```text
src/main.cpp
   -c/-m offline|stream/-o/--mediamtx*/--enable-raw 参数解析，
   加载 streams_config.json，初始化 ConfigService、VideoStreamManager

src/manager/config_service.cpp
   监控 /dev/shm/ai_config.json 的热更新，解析 streams/models/global_settings 配置
   将 model name/path/conf/nms/plugin 下发给每路流

src/manager/video_stream_manager.cpp
   根据配置创建多路 VideoStream；OSDAssociatedModel/initializeOSDForAIStream 与 ax650 一致

src/manager/video_stream.cpp
   OpenCV 解码 -> AIProcessor/InferenceManager 推理 -> OSDRenderer 画框
   -> OpenCV MP4（offline）或 FFmpeg h264_rkmpp RTSP（stream）

src/manager/ai_processor.cpp
   根据配置的 plugin 字段或默认 ./libmanhole_plugin.so 选择插件
   dlopen 插件，调用 CreateAIModel、Init、Inference、Deinit

src/manager/inference_engine.cpp / inference_manager.cpp
   单模型引擎适配 + Single/Parallel/Serial(ROI) 多模型调度（与 ax650 同语义）

src/manager/osd_renderer.cpp / src/osd_renderer_interface.cpp
   OSDRenderer + IOSDRenderer/DefaultOSDRenderer（绘制到 BGR 帧）

include/ai_interface.h
   平台无关插件 ABI：IAIModel、AI_RESULT_T、AI_FRAME_T（替代 AX_VIDEO_FRAME_T）

plugins/model_manhole_cover.cpp
   井盖模型自己的 RKNN 初始化、预处理、推理和后处理
```

编译产物为 `bin/demo` 和 `bin/libmanhole_plugin.so`，详见本目录 `FLOW2.md`。

## 3. 井盖插件

### 3.1 插件实现 `plugins/model_manhole_cover.cpp`

插件实现 `IAIModel`：

```cpp
class ManholeCoverModel : public IAIModel {
public:
    int Init(const char* model_path) override;
    void GetInputSize(int* w, int* h) override;
    int Inference(const AI_FRAME_T* pFrame, AI_RESULT_T* pResult) override;
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
读取 .rknn 模型文件内容
rknn_init -> rknn_query 查询 input/output tensor 属性
确认输入 3 通道、输出为 [1,9,8400]（兼容 [1,8400,9]）
```

`Inference()` 做：

```text
AI_FRAME_T（BGR24）包装为 cv::Mat
BGR letterbox 到 640x640，padding=114（与 model_convert / model_val 口径一致）
BGR 转 RGB，按输入属性选择 NHWC 或 NCHW 排列，uint8 送入 RKNN
rknn_inputs_set -> rknn_run -> rknn_outputs_get(want_float=1)
读取 output0 [1,9,8400]
按 [channels, anchors] 解析：0..3=cx/cy/w/h，4..8=good/broke/lose/uncovered/circle 分数
按 conf 过滤，按类别 NMS
去 letterbox padding，除 gain，还原原图坐标
写入 AI_RESULT_T，坐标为归一化 x/y/w/h，label 为五分类名称
```

不要直接套用 Model Zoo YOLO11 示例的三分支 DFL 后处理；井盖模型输出是单输出
`[1,9,8400]`，不是 `4*REG_MAX + class_num` 多头格式。

`Deinit()` 释放：

```text
rknn_destroy（context）
模型文件缓冲
```

### 3.2 编译文件 `CMakeLists.txt`

工程编译两个目标：

```cmake
add_library(manhole_plugin SHARED plugins/model_manhole_cover.cpp)
target_link_libraries(manhole_plugin "${RKNPU2_DIR}/lib/librknnrt.so" ${OpenCV_LIBS} dl pthread)

add_executable(demo src/main.cpp src/osd_renderer_interface.cpp src/manager/*.cpp)
target_link_libraries(demo ${OpenCV_LIBS} dl pthread m)

install(TARGETS demo DESTINATION bin)
install(TARGETS manhole_plugin DESTINATION bin)
```

产物：`bin/demo`（主程序）和 `bin/libmanhole_plugin.so`（井盖插件）。

### 3.3 插件加载逻辑（ai_processor.cpp）

`AIProcessor::loadModel()`：

```text
优先使用配置中 models[].plugin 或 stream 级 plugin 字段指定的插件路径
未配置 plugin 字段时，默认加载 ./libmanhole_plugin.so
dlopen 失败时回退尝试 ./bin/libmanhole_plugin.so
```

`AIProcessor::applyModelParamsToEnv()` 把配置阈值透传为环境变量：

```text
conf_threshold -> MANHOLE_CONF_THRESH / MODEL_CONF_THRESH
nms_threshold  -> MANHOLE_NMS_THRESH / MODEL_NMS_THRESH
```

插件侧 `Init()` 优先读 `MANHOLE_CONF_THRESH` / `MANHOLE_NMS_THRESH`，未设置时回退到
`MODEL_CONF_THRESH` / `MODEL_NMS_THRESH`，因此配置文件中的阈值会真实作用于 RKNN
后处理，而不是只显示在配置里。

### 3.4 模型路径解析（config_service.cpp）

`getModelPath()` 当前实现：

```text
模型名包含 "/" 或 "\\" 时按完整路径返回
模型名以 .rknn 结尾时返回 ../models/<模型名>
其他逻辑名统一映射到 ../models/manhole-cover-yolo11s-production.rknn
```

也可以不改代码，在 `streams_config.json` 的 `models[].path` 里直接写完整路径。

## 4. 配置运行

推荐使用 `models` 数组配置，写入 `config/streams_config.json`（字段与 AX650 完全一致）：

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
      "enable_raw_stream": false,
      "plugin": "./libmanhole_plugin.so",
      "models": [
        {
          "name": "manhole_cover",
          "path": "../models/manhole-cover-yolo11s-production.rknn",
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

支持的关键字段（与 AX650 一致）：

```text
stream 级：
  input_source      输入文件路径或 RTSP URL
  input_codec       h264（兼容字段，OpenCV 自动识别）
  conf_thres/nms_thres          流级默认阈值
  enable_ai         是否启用 AI
  output_width/output_height/fps    输出尺寸/帧率（RK3588 输出跟随输入源，用于校验）
  ai_output_width/ai_output_height/ai_fps   AI 输入尺寸/帧率（插件固定 640）
  enable_raw_stream          端云比对 raw 流（RK3588 暂不支持）
  plugin            插件 .so 路径（可放在 models[] 里按模型指定）
  models[]          每项：name/path/conf_threshold/nms_threshold/plugin/params
                    （多模型支持 roi_from_previous/independent）

global_settings 级：
  mediamtx_host/mediamtx_port    MediaMTX 端点
  default_model/default_conf_thres/default_nms_thres
  enable_raw_stream              端云比对 raw 流开关（默认 false）
```

运行时热更新文件固定是 `/dev/shm/ai_config.json`：

```bash
cp config/streams_config.json /dev/shm/ai_config.json
```

ConfigService 监控 `/dev/shm/ai_config.json`，文件变化时动态下发配置。

## 5. 编译和启动

在 RK3588 板端或对应交叉编译环境中（`rknpu2/` 需先按 `SOP.md` 放回）：

```bash
cd model_deployment/rk3588/manhole_cover_detection
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cmake --install build
```

前置条件：

```text
rknpu2/include/rknn_api.h 和 rknpu2/lib/librknnrt.so 已随仓库提交（直接可用）
MPP/RGA 开发头文件已准备（SOP.md §1.1：apt 或 third-party 下载，含官方 URL）
FFmpeg 开发库（libavformat/libavcodec/libavutil）与 OpenCV core/imgproc
models/manhole-cover-yolo11s-production.rknn 已放置（模型不提交 Git）
```

在线模式（RTSP/文件输入，h264_rkmpp 推流到 MediaMTX）：

```bash
cd model_deployment/rk3588/manhole_cover_detection/bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m stream
```

离线模式（输入为本地文件，输出带框 MP4）：

```bash
./demo -c ../config/streams_config.json -m offline -o /tmp/output_manhole.mp4
```

如果从源码目录运行，注意配置中的相对路径是相对当前工作目录解析的，通常应从
`bin/` 启动，或把 `models[].path`、`plugin` 写成完整路径。

## 6. 运行模式说明

主程序只支持 `-m offline` 和 `-m stream` 两种模式，参数与 AX650 一致：

```text
-c <file>       streams 配置 JSON（默认 config/streams_config.json）
-m <mode>       offline 或 stream
-o <file>       offline 输出 MP4 路径（offline only）
--mediamtx IP:PORT       MediaMTX 端点（兼容字段）
--mediamtx-host H / --mediamtx-port P  兼容字段
--enable-raw            端云比对 raw 流（RK3588 暂不支持，忽略）
-h              帮助
```

### 6.1 离线模式（-m offline）

```text
本地视频 -> OpenCV 解码 -> AI 插件推理（RKNN）-> CPU 画框
   -> OpenCV VideoWriter（mp4v）-> MP4
```

用于先确认模型、预处理、后处理和画框正确，不需要 MediaMTX。

### 6.2 在线模式（-m stream）

```text
RTSP/文件输入 -> OpenCV 解码 -> AI 插件推理 -> CPU 画框
   -> FFmpeg h264_rkmpp 管道 -> RTSP -> MediaMTX（默认 rtsp://127.0.0.1:8554/ai_out）
```

输出走板端 FFmpeg `h264_rkmpp` 硬件编码，RTSP 地址默认 `ai_out`，多路时自动加流号，
可在配置 `rtsp_output_url` 覆盖。不要把 AX650 文档中的 VENC/RTP 命令复制到这里。

## 7. 检测框和 OSD 显示说明

井盖插件本身不直接在视频帧上画框。它在 `Inference()` 中完成推理和后处理，将检测结果
写入 `AI_RESULT_T`，包括归一化的 `x/y/w/h`、类别和置信度。

RK3588 上画框由 `VideoStream::runLoop()` 内联完成：推理返回后调用
`OSDRenderer::update`，由 `DefaultOSDRenderer`（`IOSDRenderer` 默认实现）把框、类别名、
置信度直接用 OpenCV 绘制到**即将编码输出的 BGR 帧**上。井盖模型当前没有专用 OSD
Renderer（`AIProcessor::getOSDRenderer()` 返回 nullptr），使用通用 `DefaultOSDRenderer`。

```text
输入帧
   ├── 推理分支：AIProcessor/InferenceManager -> AI_RESULT_T
   └── 输出帧：同一帧上 OSDRenderer 画框 -> 编码 -> MediaMTX
```

因此，检测框应当在 MediaMTX 输出流中查看。配置中的 `enable_ai: true` 只表示启用推理；
要看到框，还要求当前帧至少检测到一个目标（`AI_RESULT_T.nObjSize > 0`）。

启动前确认：

```text
manhole_cover_detection/models/manhole-cover-yolo11s-production.rknn 存在
bin/demo 存在
bin/libmanhole_plugin.so 存在
LD_LIBRARY_PATH 包含 bin、/soc/lib 和 /usr/lib
```

建议按以下日志确认画框链路：

```text
[AIProcessor] Loading plugin: ./libmanhole_plugin.so
[AIProcessor] Model initialized successfully
[ManholeCover] Loading model: ...
[ManholeCover] thresholds: conf=0.250 nms=0.450
[ManholeCover] detections=... confidence: ...
[VideoStream] Stream 1: output_mode=file /tmp/output_manhole.mp4
```

其中，只有看到 `nObjSize > 0` 的检测结果时才会实际绘制目标框；如果模型运行正常但
没有检测到目标，视频中不会出现框，这不代表画框链路失效。

## 8. 跑通标准

日志必须看到：

```text
[AIProcessor] Loading plugin: ./libmanhole_plugin.so
[AIProcessor] Model initialized successfully
```

不得出现：

```text
dlopen failed
cannot open model
rknn_init failed
cannot open output video
```

验收顺序：

```text
1. 单张图片：model_convert/rk3588 转换产物可被 RKNN 加载
2. 精度：model_val 中 ONNX 和 RKNN mAP 对比通过
3. 板端：demo 能加载 .rknn 和 libmanhole_plugin.so
4. 实流：在 MediaMTX 输出流中确认 OSD 框、类别、置信度和仿真/验证结果基本一致
5. 稳定性：连续推理至少 100 次，记录平均延迟、P95 延迟、峰值内存和异常次数
```

如果实流中看不到框，按以下顺序排查：

```text
1. 确认查看的是 MediaMTX 输出流（ai_out），而不是原始输入流（src_in）
2. 确认 AI_RESULT_T.nObjSize > 0，排除当前画面确实没有达到阈值的目标
3. 确认模型路径、插件路径和 LD_LIBRARY_PATH 正确
4. 确认 ffmpeg h264_rkmpp 编码器可用且管道未提前退出
```

## 9. 最短 SOP

已写入的代码和作用：

```text
plugins/model_manhole_cover.cpp
  新增 ManholeCoverModel : IAIModel。
  Init(): 读取 MANHOLE_CONF_THRESH/MANHOLE_NMS_THRESH（回退 MODEL_*，默认 0.25/0.45），
  rknn_init，查询 tensor 属性。
  Inference(): BGR->RGB letterbox(640,pad=114)，rknn_inputs_set/rknn_run/rknn_outputs_get，
  解码 output0 [1,9,8400]，分类别 NMS，写 AI_RESULT_T（归一化坐标 + 类别名 + 置信度）。
  Deinit(): rknn_destroy。
  导出: CreateAIModel() / DestroyAIModel()，供 AIProcessor dlopen/dlsym 调用。

CMakeLists.txt
  新增 add_library(manhole_plugin SHARED plugins/model_manhole_cover.cpp)。
  主程序目标为 demo（src/main.cpp + src/manager/*.cpp + src/osd_renderer_interface.cpp）。
  install 安装 demo 和 manhole_plugin 到 bin/。

src/manager/ai_processor.cpp
  loadModel(): 配置 plugin 字段优先，未配置默认 ./libmanhole_plugin.so，失败回退 ./bin/libmanhole_plugin.so。
  applyModelParamsToEnv(): 将 conf_threshold/nms_threshold 写入 MANHOLE_*/MODEL_* 环境变量。

src/manager/video_stream.cpp
  runLoop(): OpenCV 解码 -> 推理 -> OSDRenderer 画框 -> MP4 / h264_rkmpp RTSP 输出。

src/manager/config_service.cpp
  getModelPath(): 非路径模型名统一映射到 ../models/manhole-cover-yolo11s-production.rknn。

config/streams_config.json
  井盖模型运行配置；models[0].name=manhole_cover，path 指向 manhole-cover-yolo11s-production.rknn。
```

需要放置的文件：

```bash
mkdir -p model_deployment/rk3588/manhole_cover_detection/models
cp model_convert/rk3588/output/manhole-cover-yolo11s-production.rknn \
  model_deployment/rk3588/manhole_cover_detection/models/
# rknpu2/include/rknn_api.h 与 rknpu2/lib/librknnrt.so 按 SOP.md 放回
```

编译：

```bash
cd model_deployment/rk3588/manhole_cover_detection
cmake -S . -B build
cmake --build build -j$(nproc)
cmake --install build
```

运行：

```bash
cd model_deployment/rk3588/manhole_cover_detection/bin
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
cd model_deployment/rk3588/manhole_cover_detection/bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m offline -o /tmp/output_manhole.mp4
```

把 `streams_config.json` 的 `input_source` 指向本地视频（例如
`/root/kaohe_6076/test.mp4`）。输出为 OpenCV `mp4v` 编码的带框 MP4。

处理完成后检查：

```bash
ffprobe -v error -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 /tmp/output_manhole.mp4
```

日志中应看到：

```text
[VideoStream] Stream 1: output_mode=file /tmp/output_manhole.mp4
[ManholeCover] detections=... confidence: ...
[Main] Offline input completed
```

如果 `detections=0`，输出视频会生成但不会出现检测框；此时应先查看井盖插件的
`[ManholeCover]` 日志，确认模型是否加载成功以及 `AI_RESULT_T.nObjSize` 是否大于 0，
或先降低配置里的 `conf_threshold` 验证。
