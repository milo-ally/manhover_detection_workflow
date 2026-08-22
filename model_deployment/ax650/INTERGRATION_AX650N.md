# AX650N 井盖模型集成

模型：`yolo11s-manhole-detection.axmodel`  
输入：`640x640 RGB`；输出：`[1,9,8400]`；类别：`good,broke,lose,uncovered,circle`。

## 1. 新增推理插件

文件：`plugins/model_manhole_cover.cpp`

```cpp
// 搜索：class ManholeCoverModel
class ManholeCoverModel final : public IAIModel {
    // 井盖模型：NV12 -> RGB -> letterbox(640x640)
    // AX_ENGINE_RunSync() -> [1, 9, 8400]
    // 解码 good/broke/lose/uncovered/circle，按类别 NMS
};

// 文件末尾，保持插件 ABI
extern "C" IAIModel* CreateAIModel() {
    return new ManholeCoverModel();
}
extern "C" void DestroyAIModel(IAIModel* model) {
    delete model;
}
```

定位：

```bash
vim +'/class ManholeCoverModel' plugins/model_manhole_cover.cpp
vim +'/AX_ENGINE_RunSync' plugins/model_manhole_cover.cpp
```

## 2. 注册 CMake 目标

文件：`CMakeLists.txt`

在原有 `yolo_plugin` 后、`demo_helmet` 前插入：

```cmake
# 原有：安全帽插件
add_library(yolo_plugin SHARED plugins/model_helmet.cpp)
target_link_libraries(yolo_plugin axdl_lib ByteTrack ${OpenCV_LIBS})

# 新增：井盖检测插件
add_library(manhole_plugin SHARED plugins/model_manhole_cover.cpp)
target_link_libraries(manhole_plugin ax_engine ax_sys ${OpenCV_LIBS})

# 下方继续是主程序
add_executable(demo_helmet ${SRC_APP} ${SRC_LIST_LIBS})
```

定位：`vim +'/add_library(yolo_plugin' CMakeLists.txt`。输出：`bin/libmanhole_plugin.so`。

## 3. 在插件选择 if/else 中加入井盖分支

文件：`src/manager/ai_processor.cpp`

在 `fire/smoke`、`crowd` 分支之间加入：

```cpp
} else if (pluginHint.find("fire") != std::string::npos ||
           pluginHint.find("smoke") != std::string::npos) {
    pluginPath = "./libsmoke_fire_plugin.so";
    ALOGN("[AIProcessor] Using smoke/fire detection plugin for model: %s",
          modelPath.c_str());

// 新增：井盖检测模型
} else if (pluginHint.find("manhole") != std::string::npos ||
           pluginHint.find("manhole_cover") != std::string::npos) {
    pluginPath = "./libmanhole_plugin.so";
    ALOGN("[AIProcessor] Using manhole-cover detection plugin for model: %s",
          modelPath.c_str());

} else if (pluginHint.find("crowd") != std::string::npos ||
           pluginHint.find("human_group") != std::string::npos) {
    pluginPath = "./libcrowd_plugin.so";
    ALOGN("[AIProcessor] Using crowd detection plugin for model: %s",
          modelPath.c_str());
}
```

定位：`vim +'/Using manhole-cover detection plugin' src/manager/ai_processor.cpp`。

## 4. 在阈值 if/else 中加入井盖分支

文件：`src/manager/ai_processor.cpp`

在 `plate` 分支后、`behavior` 分支前加入：

```cpp
} else if (lowerHint.find("plate") != std::string::npos) {
    setFloat("conf_threshold", "PLATE_CONF_THRESH");
    setFloat("nms_threshold", "PLATE_NMS_THRESH");

// 新增：井盖检测模型阈值
} else if (lowerHint.find("manhole") != std::string::npos) {
    setFloat("conf_threshold", "MANHOLE_CONF_THRESH");
    setFloat("nms_threshold", "MANHOLE_NMS_THRESH");

} else if (lowerHint.find("behavior") != std::string::npos) {
    setFloat("conf_threshold", "BEHAVIOR_CONF_THRESH");
    setFloat("nms_threshold", "BEHAVIOR_NMS_THRESH");
}
```

定位：`vim +'/MANHOLE_CONF_THRESH' src/manager/ai_processor.cpp`。

## 5. 在模型路径 if/else 中加入井盖路径

文件：`src/manager/config_service.cpp`

在“路径包含 `/` 直接返回”之后、helmet 分支之前加入：

```cpp
if (modelPath.find("/") != std::string::npos) {
    return modelPath;
}

// 新增：井盖检测模型路径
if (modelPath.find("manhole") != std::string::npos) {
    return "../models/yolo11s-manhole-detection.axmodel";
}

// 原有：安全帽模型路径
if (modelPath == "yolo11_helmet.axmodel" ||
    modelPath == "lyg_helmet.axmodel" ||
    (modelPath.find("helmet") != std::string::npos &&
     modelPath.find("pose") == std::string::npos)) {
    return "../models/yolo11_helmet.axmodel";
}
```

定位：`vim +'/井盖检测模型路径' src/manager/config_service.cpp`。

## 6. 新增第 11 路流

文件：`config/streams_config.json`

在现有 `stream_id: 10` 对象后、`streams` 数组结束前加入：

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

## 7. 告警规则

文件：`config/alarm_rules.json`

在规则数组中加入：

```json
{
    "model_type": "manhole_cover",
    "alarm_type": "井盖异常",
    "severity": "high",
    "labels": ["broke", "lose", "uncovered"]
}
```

文件：`src/main.cpp`。在 `AX_ENGINE_Init` 成功后、创建 `ConfigService` 前加载规则：

```cpp
if (AX_ENGINE_Init(&npu_attr) != 0) {
    // ... 原有失败处理
}

// 新增：加载井盖检测告警规则
std::string alarmRulesPath = "config/alarm_rules.json";
if (access(alarmRulesPath.c_str(), F_OK) != 0)
    alarmRulesPath = "../config/alarm_rules.json";
AlarmFilter::loadRulesFromFile(alarmRulesPath);

ConfigService configService("/dev/shm/ai_config.json");
```

## 8. 编译运行

```bash
# 放置模型
cp yolo11s-manhole-detection.axmodel models/

# 全量编译
cmake -S . -B build
cmake --build build
./bin/demo_helmet --help

# 新增井盖模型：只编译插件，并重新生成 demo_helmet
cmake -S . -B build
cmake --build build --target manhole_plugin demo_helmet
./bin/demo_helmet --help

# 使用第 11 路井盖配置运行
cd bin
./demo_helmet -c ../config/streams_config.json --mediamtx 127.0.0.1:8000
```

第 11 路输入地址按实际 RTSP 地址修改。主机没有 AX SDK 时只能完成语法和配置检查，最终链接及推理需在 AX650N 板端完成。

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
