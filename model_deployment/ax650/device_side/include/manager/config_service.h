#ifndef CONFIG_SERVICE_H
#define CONFIG_SERVICE_H

#include <string>
#include <mutex>
#include <thread>
#include <functional>
#include <vector>
#include "../../utilities/json.hpp"
#include "video_stream.h"  // 引入 ModelStageConfig 和 AIPipelineMode

struct ConfigUpdate {
    std::string modelName;
    std::string modelPath;
    float confThreshold;
    float nmsThreshold;
    bool valid;
    int streamId = -1;  // 流 ID，-1 表示更新所有流（向後兼容）
    int cameraId = -1;  // 攝像頭 ID，用於映射到流 ID
    // 多模型配置（支持並行/串行）
    std::vector<ModelStageConfig> modelStages;  // 非空時優先於單一 modelName/modelPath
    AIPipelineMode aiPipelineMode = AIPipelineMode::Parallel;
};

class ConfigService {
public:
    explicit ConfigService(const std::string& configPath);
    ~ConfigService();
    
    void startMonitoring();
    void stopMonitoring();
    
    void registerConfigListener(std::function<void(const ConfigUpdate&)> listener);
    
    // Current configuration access
    std::string getCurrentModel() const;
    float getConfThreshold() const;
    float getNmsThreshold() const;
    bool isConfigValid() const;
    std::string getModelPath(const std::string& modelName) const;  // 公开方法
    
    bool isShutdownRequested() const { return shutdownRequested_; }

private:
    void monitorConfig();
    bool loadConfig();
    
    std::string configPath_;
    mutable std::mutex configMutex_;
    
    // Current configuration
    std::string currentModelName_ = "none";
    float confThreshold_ = 0.5f;
    float nmsThreshold_ = 0.45f;
    bool configValid_ = false;
    
    // Monitoring thread
    std::thread monitorThread_;
    bool running_ = false;
    bool shutdownRequested_ = false;
    
    // Listeners for config changes
    std::vector<std::function<void(const ConfigUpdate&)>> listeners_;
    std::mutex listenersMutex_;
};

#endif // CONFIG_SERVICE_H

