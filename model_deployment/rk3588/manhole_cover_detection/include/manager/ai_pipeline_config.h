#ifndef AI_PIPELINE_CONFIG_H
#define AI_PIPELINE_CONFIG_H

// RK3588 版流/模型配置。与 AX650 的 include/manager/ai_pipeline_config.h 同构：
// 字段名、默认值、models[]/global_settings 语义一致；仅去掉 AX 硬件组字段
// （ivpsGroup/vdecGroup），并补充 RK3588 的 OpenCV 输出字段。

#include <string>
#include <vector>
#include "../../utilities/json.hpp"

/** AI 管線模式：並行（同一幀多模型）或串行（前序結果驅動 ROI 推理） */
enum class AIPipelineMode {
    Parallel,  // 同一幀並行跑多個模型，結果合併
    Serial     // 先跑階段0，再依其結果裁剪 ROI 跑階段1、2...
};

/** 單一模型階段配置（用於多模型/串行管線） */
struct ModelStageConfig {
    std::string modelPath;
    std::string modelName;
    std::string pluginPath;
    float confThreshold = 0.45f;
    float nmsThreshold = 0.45f;
    bool roiFromPrevious = false;  // 串行時：是否基於前一階段檢測框裁剪 ROI 再推理
    bool independent = false;      // 是否獨立運行（不依賴前序結果，可與其他獨立模型並行）
    // 前端模型專屬微調參數，透傳到設備端後處理/OSD
    nlohmann::json params = nlohmann::json::object();
};

/** 單路流的配置（字段名與 ax650 完全一致） */
struct StreamConfig {
    int streamId = -1;
    std::string pluginPath;      // 插件 .so 路徑；空則默認 ./libmanhole_plugin.so
    std::string inputSource;     // 本地文件路徑或 RTSP URL
    std::string inputCodec = "h264";  // 兼容字段（OpenCV 自動識別）
    bool enableAI = false;
    std::string modelPath;       // 單模型時使用
    std::string modelName;       // 單模型時使用
    bool isCommandLineModel = false;  // 模型是否由命令行指定（不應被配置文件覆蓋）
    float confThreshold = 0.45f;
    float nmsThreshold = 0.45f;
    // 輸出尺寸/幀率（RK3588 上 OpenCV 輸出跟隨輸入源，這些字段用於校驗/日誌）
    int outputWidth = 1920;
    int outputHeight = 1080;
    int fps = 30;
    int bitrateKbps = 4000;      // 编码码率（档位2 MPP VENC；可配置 bitrate_kbps）
    int aiOutputWidth = 640;     // ai_output_width（RK3588 插件固定 640）
    int aiOutputHeight = 640;    // ai_output_height
    int aiFps = 15;              // ai_fps
    bool enableRawStream = false;   // enable_raw_stream（端雲比對 raw 流，RK3588 暂不支持）
    std::string rtspEndpoint = "axstream";  // 兼容字段（RK3588 使用 rtspOutputUrl）
    bool isRTSPOutput = false;   // 兼容字段
    bool isMediaMTXOutput = false;  // 兼容字段
    bool isFileOutput = false;   // offline 文件輸出
    std::string outputFilePath;  // 文件輸出路徑（.mp4）
    std::string rtspOutputUrl;   // RK3588：RTSP 輸出 URL（如 rtsp://127.0.0.1:8554/ai_out）
    std::string mediamtxEndpoint = "127.0.0.1:8000";  // MediaMTX 端點（兼容字段）
    // 多模型：同一流並行多模型或串行階段
    AIPipelineMode aiPipelineMode = AIPipelineMode::Parallel;
    std::vector<ModelStageConfig> modelStages;  // 非空時優先於單一 modelPath/modelName
};

#endif  // AI_PIPELINE_CONFIG_H
