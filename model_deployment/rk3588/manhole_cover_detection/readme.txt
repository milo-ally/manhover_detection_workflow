# RK3588 井盖检测 Demo（AX650 同构架构）

本目录是 RK3588 的独立板端部署工程，使用 Rockchip RKNN Runtime 和 OpenCV。
工程结构与 AX650 的 `model_deployment/ax650/manhole_cover_detection` 同构：
配置驱动（`config/streams_config.json` + `/dev/shm/ai_config.json` 热更新）、
dlopen 插件 ABI（`libmanhole_plugin.so` + `IAIModel`）、多流管理（VideoStreamManager）、
`-c/-m offline|stream/-o` 命令行。解码/输出使用 OpenCV 与 FFmpeg（`h264_rkmpp`），
不依赖 AX 硬件流水线。

```text
输入视频 -> OpenCV 解码 -> AI 插件推理（RKNN）-> 五分类后处理/NMS
         -> CPU 画框 -> OpenCV MP4 或 FFmpeg h264_rkmpp RTSP 输出
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
├── config/streams_config.json         # 字段与 AX650 streams_config.json 完全一致
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
└── rknpu2/                            # 不随仓库提交（.gitignore 忽略），编译前需要放回
    ├── include/rknn_api.h
    └── lib/librknnrt.so
```

工程类名/方法名与 AX650 完全一致（ConfigService / VideoStreamManager / VideoStream /
AIProcessor / InferenceManager / OSDRenderer / DefaultOSDRenderer / IAIModel），
`streams_config.json` 字段格式与 AX650 一致（`output_width/output_height/fps`、
`ai_output_width/ai_output_height/ai_fps`、`enable_raw_stream`、`plugin`、
`models[]`（`name/path/plugin/conf_threshold/nms_threshold/roi_from_previous/
independent/params`）、`global_settings`（`mediamtx_host/mediamtx_port/default_*/
enable_raw_stream`））。仅底层硬件实现不同：AX650 走 VDEC/IVPS/VENC 硬件流水线，
RK3588 用 OpenCV 解码 + FFmpeg h264_rkmpp 输出 + RKNN 推理。

`rknpu2/` 目录被 `.gitignore` 忽略，不会出现在克隆的仓库里；`CMakeLists.txt` 默认从
`rknpu2/include` 和 `rknpu2/lib` 查找 `rknn_api.h` 与 `librknnrt.so`，编译前必须先从
Rockchip 官方 `rknn_model_zoo` 的 `3rdparty/rknpu2` 把对应 RK3588/aarch64 的 Runtime
放回本目录（提交见 `SOP.md`）。

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
cmake -S . -B build -DOpenCV_DIR="$HOME/opt/opencv-dev/usr/lib/aarch64-linux-gnu/cmake/opencv4"
```

将本目录完整复制到板端后执行：

```bash
cd model_deployment/rk3588/manhole_cover_detection
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
cmake --install build
```

产物：`bin/demo` 和 `bin/libmanhole_plugin.so`。

## 离线视频运行

把 `config/streams_config.json` 的 `input_source` 指向本地视频文件后：

```bash
cd bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m offline -o /tmp/output_manhole.mp4
```

参数：`-c` 指定配置、`-m offline` 离线模式、`-o` 输出 MP4。模型路径默认映射到
`../models/manhole-cover-yolo11s-production.rknn`（也可在配置 `models[].path` 写完整路径）。
程序逐帧输出检测数量和推理耗时，结束后保存带框 MP4（OpenCV `mp4v` 编码）。

## SSH 隧道 RTSP 运行

按上级 `FLOW1.md` 建立 SSH 隧道后，把 `streams_config.json` 的 `input_source` 改为
`rtsp://127.0.0.1:8556/src_in`：

```bash
cd bin
export LD_LIBRARY_PATH=$PWD:/soc/lib:/usr/lib:$LD_LIBRARY_PATH
./demo -c ../config/streams_config.json -m stream
```

程序使用板端 `h264_rkmpp` 编码器向本地 MediaMTX 发布 RTSP（默认
`rtsp://127.0.0.1:8554/ai_out`，可在配置 `rtsp_output_url` 覆盖）。RK3588 上验证
输入使用 `127.0.0.1:8556`；主机通过 SSH `-L` 映射后的
`rtsp://127.0.0.1:8557/ai_out` 查看结果。

## 配置热更新

`demo` 固定监控 `/dev/shm/ai_config.json`（ConfigService），文件变化时动态下发
阈值/模型更新：

```bash
cp config/streams_config.json /dev/shm/ai_config.json
```

## 实现注意事项

- 插件 ABI 与 AX650 同构：`IAIModel`（`Init/GetInputSize/Inference/Deinit`）+ 
  `CreateAIModel()/DestroyAIModel()`，主程序 `AIProcessor` 负责 `dlopen`；
  多模型（并行/串行 ROI）由 `InferenceManager` 调度（`inference_engine` /
  `inference_manager` 与 ax650 同名同语义）。
- OSD 与 ax650 同名同语义：`IOSDRenderer` / `DefaultOSDRenderer` / `OSDRenderer`；
  差异仅在输出目标——AX650 走 IVPS 硬件 region 叠加，RK3588 直接绘制到待编码的 BGR 帧。
- 阈值透传：`AIProcessor::applyModelParamsToEnv` 把配置 `conf_threshold/nms_threshold`
  写入 `MANHOLE_CONF_THRESH/MANHOLE_NMS_THRESH`（并回退 `MODEL_*`）；插件 `Init()`
  读取这两个环境变量，默认 0.25 / 0.45。
- 输入使用 RGB `uint8`，与当前转换配置的 `/255` 预处理对应（letterbox padding=114）。
- 部署模型使用 FP RKNN；输出通过 `want_float = 1` 请求 Runtime 返回 FP32。
- 不要直接使用当前 INT8 产物：该检测头的坐标和类别分数共用输出量化范围，可能导致类别分数全部为 0。
- RKNN Runtime 必须与板端架构匹配，使用 `aarch64` 的 `librknnrt.so`；该文件不随仓库提交，需要先放回 `rknpu2/lib/`。
