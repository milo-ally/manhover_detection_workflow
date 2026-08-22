# RK3588 井盖模型集成

模型：`yolo11s-manhole-detection.rknn`  
输入：`640x640 RGB`；输出：`[1,9,8400]`；类别：`good,broke,lose,uncovered,circle`。

## 1. 插件名称路由

文件：`src/rk3588/application.cpp`

在 `plate` 分支后、`crowd` 分支前加入：

```cpp
if (hint.find("fire") != std::string::npos || hint.find("smoke") != std::string::npos)
    return "libsmoke_fire_plugin.so";
if (hint.find("plate") != std::string::npos)
    return "libplate_detection_plugin.so";

// 新增：井盖检测模型
if (hint.find("manhole") != std::string::npos ||
    hint.find("manhole_cover") != std::string::npos)
    return "libmanhole_plugin.so";

if (hint.find("crowd") != std::string::npos || hint.find("group") != std::string::npos)
    return "libcrowd_plugin.so";
```

定位：

```bash
vim +'/libmanhole_plugin.so' src/rk3588/application.cpp
```

模型名 `manhole_cover` 会加载 `bin/libmanhole_plugin.so`。

## 2. 注册 RKNN 插件目标

文件：`CMakeLists.txt`

在插件名称列表中加入 `manhole`：

```cmake
set(RK_PLUGIN_NAMES
    helmet fall constructionsite smoke_fire plate_detection human_detection
    behavior crowd face_detection face_recognition manhole yolo)
```

下面的原有循环会自动生成插件：

```cmake
foreach(plugin IN LISTS RK_PLUGIN_NAMES)
    add_library(${plugin}_plugin SHARED plugins/rknn_plugin.cpp)
    target_compile_definitions(${plugin}_plugin PRIVATE RK_PLUGIN_NAME="${plugin}")
    target_link_libraries(${plugin}_plugin PRIVATE rkdl_lib ByteTrack)
endforeach()
```

定位：`vim +'/RK_PLUGIN_NAMES' CMakeLists.txt`。生成：`bin/libmanhole_plugin.so`。

## 3. 井盖类别解码

文件：`librkdl/src/yolo_postprocess.cpp`

在 `smoke` 分支后、`face` 分支前加入：

```cpp
if (profile.find("smoke") != std::string::npos) {
    static const char* names[] = {"smoke", "fire"};
    return class_id >= 0 && class_id < 2 ? names[class_id] : "object";
}

// 新增：井盖检测模型类别顺序必须与 RKNN 输出一致
if (profile.find("manhole") != std::string::npos) {
    static const char* names[] = {
        "good", "broke", "lose", "uncovered", "circle"
    };
    return class_id >= 0 && class_id < 5 ? names[class_id] : "object";
}

if (profile.find("face") != std::string::npos)
    return "face";
```

定位：`vim +'/profile.find("manhole")' librkdl/src/yolo_postprocess.cpp`。

预处理、RKNN 输入输出和 NMS 仍复用现有 `rkdl::RknnModel`，只有类别名称需要增加映射。

## 4. 新增第 11 路

文件：`config/streams_config.json`

在 `stream_id: 10` 后、数组结束前加入：

```json
{
    "stream_id": 11,
    "input_source": "rtsp://172.19.31.36:8554/lygstream11",
    "model_name": "manhole_cover",
    "conf_thres": 0.25,
    "nms_thres": 0.45,
    "output_width": 1920,
    "output_height": 1080,
    "fps": 30,
    "enable_ai": true,
    "ai_output_width": 640,
    "ai_output_height": 640,
    "ai_fps": 60
}
```

定位：`vim +'/"stream_id": 11' config/streams_config.json`。

`model_name` 会经过 `resolve_rknn_model()` 自动解析为 `../models/manhole_cover.rknn`；也可以直接填写 `.rknn` 文件路径。

## 5. 告警规则

文件：`config/alarm_rules.json`

在 `alarm_rules` 数组中加入：

```json
{
    "model_type": "manhole_cover",
    "alarm_type": "井蓋異常",
    "severity": "high",
    "report_all": false,
    "labels": ["broke", "lose", "uncovered"]
}
```

## 6. 编译与运行

```bash
# 放置模型
cp yolo11s-manhole-detection.rknn models/manhole_cover.rknn

# 全量编译
cmake -S . -B build -DOpenCV_DIR=/opt/opencv-rk3588/lib/cmake/opencv4 -DBUILD_TESTING=OFF
cmake --build build
./bin/demo_helmet --help

# 新增井盖模型：只编译插件，同时重新编译 demo_helmet
cmake -S . -B build -DOpenCV_DIR=/opt/opencv-rk3588/lib/cmake/opencv4 -DBUILD_TESTING=OFF
cmake --build build --target manhole_plugin demo_helmet
./bin/demo_helmet --help

# 使用现有配置运行第 11 路
./bin/demo_helmet -c config/streams_config.json --mediamtx 127.0.0.1:8000
```

第 11 路 RTSP 地址按实际情况修改。板端必须提供 RKNN Runtime、MPP、RGA 和 OpenCV。

## MediaMTX

文件：`mediamtx.yml`。设备将主码流 RTP 推到 `8000,8002,...,8020`，MediaMTX 对外提供：

固定使用主项目 `/home/milo/桌面/00_lyg_ai_platform` 使用的 MediaMTX v1.20.0。

```bash
# 启动 MediaMTX
./mediamtx ./mediamtx.yml

# 启动设备程序
./bin/demo_helmet -c config/streams_config.json --mediamtx 127.0.0.1:8000

# 查看第 11 路井盖检测视频
ffplay rtsp://127.0.0.1:8554/live11
```

`live11_raw` 使用相邻端口 `8021`，只有启用 raw 输出时才会产生。
