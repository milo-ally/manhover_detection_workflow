# RK3588 井盖检测 Demo（AX650 同构架构 · 档位2：MPP/RGA 双流硬件流水线）

本目录是 RK3588 的独立板端部署工程，与 AX650 的
`model_deployment/ax650/manhole_cover_detection` **架构完全同构**：配置驱动
（`config/streams_config.json` + `/dev/shm/ai_config.json` 热更新）、dlopen 插件 ABI
（`libmanhole_plugin.so` + `IAIModel`）、多流管理（VideoStreamManager）、每路输入拆
**主码流 + AI 流**两条流水线、`-c/-m offline|stream/-o` 命令行。

底层硬件对应关系（档位2）：

```text
AX650                RK3588（本工程）
VDEC 硬件解码   ->   MPP mpi_dec（RkDecoder）
IVPS 缩放/转换 ->   RGA（rga_ops：resize / NV12<->BGR）
VENC 硬件编码   ->   MPP mpi_enc（RkEncoder）
RTP 推流        ->   rtp_pusher（与 ax650 同源，RTP/UDP -> MediaMTX）
IVPS OSD region ->   降级：CPU 画框（主码流编码线程拉取 AI 结果，绘制到 BGR 帧）
VideoDemux      ->   H264Demux（FFmpeg libavformat：RTSP/本地文件 -> H.264 AnnexB）
```

```text
输入 RTSP / 本地视频（H.264）
   │
   ▼
H264Demux（libavformat，RTSP 强制 tcp；h264_mp4toannexb）
   │
   ▼
RkDecoder（MPP mpi_dec，H.264 -> NV12）◄──────── 对应 AX650 VDEC
   │
   ├──────────────────────────────┬──────────────────────────────┐
   ▼                              ▼                              ▼
【主码流：输出流】            【AI 流：推理流】
 RGA NV12->BGR(输出尺寸)        FrameBroker 取最新 NV12 帧
   │  （对应 AX650 IVPS）          │
   ▼                              ▼
 CPU 画框（OSD 降级：           RGA NV12->BGR(640x640)
  拉取 SharedAIResult，            │
  DefaultOSDRenderer 绘制）        ▼
   │                            插件 libmanhole_plugin.so
   ▼                              （RKNN 推理 -> AI_RESULT_T）
 RGA BGR->NV12(输出尺寸)           │
   │                              ▼
   ▼                          SharedAIResult.set()
 RkEncoder（MPP mpi_enc）         （对应 AX650 OSDAssociatedModel）
   │  （对应 AX650 VENC）          │
   ▼                              │
 rtp_pusher -> MediaMTX(RTP)  ◄───┘（主码流编码线程读取，跨流叠加画框）
 或离线 raw H.264 文件 + ffmpeg 封装 MP4
```

当前模型输出是单个 `output0 [1,9,8400]`，不是 Model Zoo YOLO11 示例中的三分支 DFL
输出，因此本工程只复用示例的 RKNN API 调用和输入处理方式，后处理按当前模型单独实现。

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
├── config/streams_config.json         # 字段与 AX650 streams_config.json 完全一致
├── common/
│   ├── rk_media.{h,cpp}               # 档位2：MPP 解码/编码 + RGA 封装（对应 VDEC/IVPS/VENC）
│   ├── h264_demux.{h,cpp}             # FFmpeg libavformat 解封装（对应 VideoDemux）
│   ├── frame_broker.h                 # 主码流<->AI 流共享（NV12 帧 / AI 结果）
│   └── common_pipeline/rtp_pusher.{c,h}  # 与 ax650 同源：RTP/UDP 推 MediaMTX
├── include/
│   ├── ai_interface.h                 # 平台无关插件 ABI（IAIModel / AI_RESULT_T / AI_FRAME_T）
│   ├── rknpu_manhole.hpp              # RKNN 检测器类声明
│   ├── osd_renderer_interface.h       # IOSDRenderer / DefaultOSDRenderer（绘制到 BGR 帧）
│   └── manager/                       # config_service / ai_processor / video_stream /
│                                      # video_stream_manager / inference_engine /
│                                      # inference_manager / osd_renderer / ai_pipeline_config
├── plugins/model_manhole_cover.cpp    # -> libmanhole_plugin.so（IAIModel 实现，内部 rknn）
├── src/
│   ├── main.cpp                       # -c/-m offline|stream/-o/--mediamtx*/--enable-raw
│   ├── osd_renderer_interface.cpp
│   └── manager/*.cpp                  # 与 ax650 src/manager/ 一一对应
├── utilities/                         # json.hpp / sample_log.h（平台无关）
├── models/manhole-cover-yolo11s-production.rknn
└── rknpu2/                            # RKNN 运行时（随仓库提交）
    ├── include/rknn_api.h
    └── lib/librknnrt.so
```

工程类名/方法名与 AX650 完全一致（ConfigService / VideoStreamManager / VideoStream /
AIProcessor / InferenceManager / OSDRenderer / DefaultOSDRenderer / IAIModel），
`streams_config.json` 字段格式与 AX650 一致（`output_width/output_height/fps`、
`bitrate_kbps`、`ai_output_width/ai_output_height/ai_fps`、`enable_raw_stream`、
`plugin`、`models[]`（`name/path/plugin/conf_threshold/nms_threshold/
roi_from_previous/independent/params`）、`global_settings`
（`mediamtx_host/mediamtx_port/default_*/enable_raw_stream`））。

## RKNN 运行时（rknpu2/）

`rknpu2/`（`include/rknn_api.h` + `lib/librknnrt.so`，aarch64）**已随仓库提交**，无需
额外准备即可编译。可复现下载地址（Rockchip 官方 `rknn_model_zoo`，固定提交
`bad6c7334531becaf90a561988519b7bec34d0ab`）：

```bash
cd model_deployment/rk3588/manhole_cover_detection
mkdir -p rknpu2/include rknpu2/lib
# ① rknn_api.h（头文件，与架构无关）
curl -L -o rknpu2/include/rknn_api.h \
  "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/bad6c7334531becaf90a561988519b7bec34d0ab/3rdparty/rknpu2/include/rknn_api.h"
# ② librknnrt.so（aarch64 运行库；仓库内路径 3rdparty/rknpu2/Linux/aarch64/librknnrt.so）
curl -L -o rknpu2/lib/librknnrt.so \
  "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/bad6c7334531becaf90a561988519b7bec34d0ab/3rdparty/rknpu2/Linux/aarch64/librknnrt.so"
file rknpu2/lib/librknnrt.so   # 必须显示 ARM aarch64
```

## MPP / RGA 依赖（档位2，必做）

档位2 使用 Rockchip **MPP**（硬件解码/编码）与 **RGA**（2D 缩放/格式转换）。板端
LubanCat/官方镜像一般已带运行库（`/usr/lib/librockchip_mpp*.so`、`/usr/lib/librga.so`），
但**开发头文件不一定齐全**，克隆仓库后按下面任一方式准备：

**方式 A：板端 apt 安装开发包（若镜像源里有）**

```bash
sudo apt update
sudo apt install -y librockchip-mpp-dev librga-dev   # 名称以板端仓库实际为准
```

**方式 B：从官方 GitHub 下载头文件（可复现，固定版本）**

```bash
cd model_deployment/rk3588/manhole_cover_detection
mkdir -p third-party/mpp third-party/rga

# ① MPP：rockchip-linux/mpp，tag 1.1.0（commit 075030f987bc960b54b5fcbcf5711166a408b7c3 的
#    对应提交 c08762ebfadeb4e986d2fed993bc7a54862d3ebe），头文件在 inc/
git clone --depth 1 --branch 1.1.0 https://github.com/rockchip-linux/mpp.git /tmp/mpp
cp /tmp/mpp/inc/*.h third-party/mpp/inc/
# （运行库 librockchip_mpp.so：方式 A 提供，或用板端 SDK/cmake 构建 mpp 后拷贝）

# ② RGA：airockchip/rknn_model_zoo 提交 bad6c733.../3rdparty/librga
#    （头文件 include/ + aarch64 预编译库 Linux/aarch64/librga.so）
mkdir -p third-party/rga/include third-party/rga/lib
curl -L -o third-party/rga/include/im2d.h \
  "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/bad6c7334531becaf90a561988519b7bec34d0ab/3rdparty/librga/include/im2d.h"
curl -L -o third-party/rga/lib/librga.so \
  "https://raw.githubusercontent.com/airockchip/rknn_model_zoo/bad6c7334531becaf90a561988519b7bec34d0ab/3rdparty/librga/Linux/aarch64/librga.so"
file third-party/rga/lib/librga.so   # 必须显示 ARM aarch64
```

> 出处汇总：MPP 源码 `https://github.com/rockchip-linux/mpp`（tag 1.1.0）；RGA 头文件/库
> 位于 `rknn_model_zoo`（提交 `bad6c7334531becaf90a561988519b7bec34d0ab`）的
> `3rdparty/librga/` 下（`include/` + `Linux/aarch64/librga.so`）。
> `CMakeLists.txt` 查找顺序：系统路径（`/usr/include/rockchip`、`/usr/include/rga`）
> → `third-party/mpp`、`third-party/rga`。

## 板端编译

依赖：C++ 编译器、CMake、FFmpeg 开发库（libavformat/libavcodec/libavutil）、OpenCV
`core/imgproc`、MPP、RGA、RKNN Runtime：

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config ffmpeg libavformat-dev libavcodec-dev libavutil-dev
sudo apt install -y libopencv-dev            # 若与 rkmpp FFmpeg 冲突，改用 install_opencv.sh
# MPP/RGA 按上文「MPP / RGA 依赖」准备（apt 或 third-party）
```

如果 `libopencv-dev` 因板端 `rkmpp` FFmpeg 版本冲突无法安装，运行本目录的安装脚本：

```bash
chmod +x install_opencv.sh
./install_opencv.sh
cmake -S . -B build -DOpenCV_DIR="$HOME/opt/opencv-dev/usr/lib/aarch64-linux-gnu/cmake/opencv4"
```

将本目录完整复制到板端后执行：

```bash
cd model_deployment/rk3588/manhole_cover_detection
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
cmake --install build
```

产物：`bin/demo` 和 `bin/libmanhole_plugin.so`。若 MPP/RGA 头文件不在系统路径，把
`third-party/mpp`、`third-party/rga` 放好再重新 configure（或用
`-DMPP_INCLUDE_DIR=... -DRGA_INCLUDE_DIR=...` 指定）。

## 离线视频运行（-m offline）

把 `config/streams_config.json` 的 `input_source` 指向本地 H.264 视频后：

```bash
cd bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m offline -o /tmp/output_manhole.mp4
```

流程：`H264Demux -> MPP 解码 -> RGA 缩放 -> CPU 画框 -> RGA 转 NV12 -> MPP 编码
（raw H.264）-> ffmpeg 封装 MP4`（与 AX650 离线链路一致：先写 raw H.264，结束后
`ffmpeg -f h264 -c copy` 封装）。模型路径默认映射到
`../models/manhole-cover-yolo11s-production.rknn`（可在配置 `models[].path` 覆盖）。

## 在线推流运行（-m stream）

按上级 `FLOW1.md` 建立 SSH 隧道后，把 `streams_config.json` 的 `input_source` 改为
`rtsp://127.0.0.1:8556/src_in`：

```bash
cd bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m stream
```

流程：`H264Demux(RTSP) -> MPP 解码 -> RGA 缩放 -> CPU 画框 -> RGA 转 NV12 -> MPP 编码
-> rtp_pusher(RTP/UDP) -> MediaMTX`（与 AX650 完全一致）。MediaMTX 端点默认
`127.0.0.1:8000`，可由 `--mediamtx IP:PORT` / 配置 `mediamtx_host/mediamtx_port` /
环境变量覆盖；主机通过 SSH `-L 8557` 查看 `rtsp://127.0.0.1:8557/ai_out`。

## 配置热更新

`demo` 固定监控 `/dev/shm/ai_config.json`（ConfigService），文件变化时动态下发
阈值/模型更新：

```bash
cp config/streams_config.json /dev/shm/ai_config.json
```

## 实现注意事项

- 插件 ABI 与 AX650 同构：`IAIModel`（`Init/GetInputSize/Inference/Deinit`）+ 
  `CreateAIModel()/DestroyAIModel()`，主程序 `AIProcessor` 负责 `dlopen`；
  多模型（并行/串行 ROI）由 `InferenceManager` 调度（与 ax650 同语义）。
- **每路输入拆两条流（与 AX650 一致）**：主码流（H264Demux+MPP 解码+RGA+OSD+MPP 编码
  +rtp_pusher）与 AI 流（FrameBroker 取帧+RGA 640+RKNN 推理+SharedAIResult）；
  主码流编码线程在编码前拉取 AI 结果并 CPU 画框（IVPS OSD region 的降级实现）。
- 阈值透传：`AIProcessor::applyModelParamsToEnv` 把配置 `conf_threshold/nms_threshold`
  写入 `MANHOLE_CONF_THRESH/MANHOLE_NMS_THRESH`（并回退 `MODEL_*`）；插件 `Init()`
  读取这两个环境变量，默认 0.25 / 0.45。
- 输入使用 RGB `uint8`，与当前转换配置的 `/255` 预处理对应（letterbox padding=114；
  AI 流经 RGA 缩放到 640x640 后 letterbox 为恒等）。
- 部署模型使用 FP RKNN；输出通过 `want_float = 1` 请求 Runtime 返回 FP32。
- 不要直接使用当前 INT8 产物：该检测头的坐标和类别分数共用输出量化范围，可能导致类别分数全部为 0。
- RKNN Runtime 必须与板端架构匹配，仓库随附 `aarch64` 的 `librknnrt.so`（`rknpu2/lib/`）。
