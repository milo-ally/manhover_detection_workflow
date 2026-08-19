#ifndef AI_PIPELINE_CONFIG_H
#define AI_PIPELINE_CONFIG_H

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
    // 前端模型專屬微調參數，透傳到設備端後處理/OSD（例如 ArcFace 閾值、追蹤鎖名門檻）
    nlohmann::json params = nlohmann::json::object();
};

#endif  // AI_PIPELINE_CONFIG_H
