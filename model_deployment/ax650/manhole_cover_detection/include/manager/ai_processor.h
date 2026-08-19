#ifndef AI_PROCESSOR_H
#define AI_PROCESSOR_H

#include <memory>
#include <string>
#include <mutex>
#include "../../utilities/json.hpp"
#include "../ai_interface.h"

// 前向聲明
class IOSDRenderer;

class AIProcessor {
public:
    // modelName 可選：人員聚集與人員偵測共用 human_detection.axmodel，需用 modelName 區分插件
    explicit AIProcessor(const std::string& modelPath, const std::string& modelName = "",
                         const nlohmann::json& modelParams = nlohmann::json::object());
    ~AIProcessor();
    
    bool loadModel(const std::string& modelPath, const std::string& modelName = "",
                   const nlohmann::json& modelParams = nlohmann::json::object());
    void unloadModel();
    bool processFrame(const AX_VIDEO_FRAME_T* frame, AI_RESULT_T* result);
    
    void setThresholds(float conf, float nms);
    void getInputSize(int* w, int* h) const;
    
    std::string getModelPath() const;
    bool isModelLoaded() const { return model_ != nullptr; }
    
    // 獲取 OSD 渲染器（每個模型可以有自己的渲染器）
    // 如果返回 nullptr，則使用默認渲染器
    std::shared_ptr<IOSDRenderer> getOSDRenderer();

private:
    IAIModel* model_ = nullptr;
    void* pluginHandle_ = nullptr;
    std::string modelPath_;
    std::string modelName_;  // 用於插件/OSD 選擇（如 crowd 與 human 共用同一 path）
    nlohmann::json modelParams_ = nlohmann::json::object();
    
    float confThreshold_ = 0.5f;
    float nmsThreshold_ = 0.45f;
    
    mutable std::mutex modelMutex_;  // 保護模型訪問
    
    // OSD 渲染器（由模型插件創建，如果有的話）
    std::shared_ptr<IOSDRenderer> osdRenderer_;

    static void applyModelParamsToEnv(const std::string& modelHint, const nlohmann::json& modelParams);
};

#endif // AI_PROCESSOR_H

