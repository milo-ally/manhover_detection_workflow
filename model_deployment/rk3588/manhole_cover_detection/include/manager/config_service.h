#ifndef CONFIG_SERVICE_H
#define CONFIG_SERVICE_H

// RK3588 版配置服務。与 AX650 的 src/manager/config_service.{h,cpp} 同构：
// 监控 /dev/shm/ai_config.json 的热更新，解析 streams 配置并下发给监听者。

#include <string>
#include <mutex>
#include <thread>
#include <functional>
#include <vector>
#include "../../utilities/json.hpp"
#include "ai_pipeline_config.h"  // 引入 ModelStageConfig 和 AIPipelineMode

struct ConfigUpdate {
    std::string modelName;
    std::string modelPath;
    float confThreshold;
    float nmsThreshold;
    bool valid;
    int streamId = -1;  // 流 ID，-1 表示更新所有流
    // 多模型配置（支持并行/串行）
    std::vector<ModelStageConfig> modelStages;  // 非空时优先于单一 modelName/modelPath
    AIPipelineMode aiPipelineMode = AIPipelineMode::Parallel;
};

class ConfigService {
public:
    explicit ConfigService(const std::string& configPath);
    ~ConfigService();

    void startMonitoring();
    void stopMonitoring();

    void registerConfigListener(std::function<void(const ConfigUpdate&)> listener);

    std::string getCurrentModel() const;
    float getConfThreshold() const;
    float getNmsThreshold() const;
    bool isConfigValid() const;
    bool isShutdownRequested() const { return shutdownRequested_; }

    // 根据模型名获取模型路径（本部署只有井盖模型，逻辑名统一映射）
    std::string getModelPath(const std::string& modelName) const;

private:
    void monitorConfig();
    bool loadConfig();

    std::string configPath_;
    mutable std::mutex configMutex_;
    std::thread monitorThread_;
    bool running_ = false;
    bool shutdownRequested_ = false;

    std::string currentModelName_;
    float confThreshold_ = 0.25f;
    float nmsThreshold_ = 0.45f;
    bool configValid_ = false;

    std::vector<std::function<void(const ConfigUpdate&)>> listeners_;
    mutable std::mutex listenersMutex_;
};

#endif  // CONFIG_SERVICE_H
