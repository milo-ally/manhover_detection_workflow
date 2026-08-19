#ifndef VIDEO_STREAM_MANAGER_H
#define VIDEO_STREAM_MANAGER_H

// RK3588 版视频流管理器。与 AX650 的 include/manager/video_stream_manager.h 同构：
// 方法名/OSDAssociatedModel 结构与 ax650 一致；loadStreamsFromConfig 解析
// streams_config.json 的全部字段（含 global_settings）。

#include <vector>
#include <map>
#include <functional>
#include <mutex>
#include <thread>
#include <memory>
#include <cstring>
#include "video_stream.h"
#include "config_service.h"
#include "../ai_interface.h"
#include "../osd_renderer_interface.h"
#include "../../utilities/sample_log.h"

// 前向聲明
class IOSDRenderer;

// OSD 管理結構（与 ax650 一致）
struct OSDAssociatedModel {
    std::weak_ptr<AIProcessor> aiProcessor;    // AI 處理器（使用 weak_ptr 避免循環引用）
    std::shared_ptr<IOSDRenderer> osdRenderer; // OSD 渲染器（由模型提供，如果沒有則使用默認渲染器）
    std::vector<AI_RESULT_T> latestResults;    // 最新的檢測結果（每路流一份）
    AI_RESULT_T latestResult;                  // 兼容字段
    std::mutex resultMutex;                    // 保護結果的互斥鎖

    OSDAssociatedModel() {
        memset(&latestResult, 0, sizeof(latestResult));
    }
};

class VideoStreamManager {
public:
    explicit VideoStreamManager(ConfigService& configService);
    ~VideoStreamManager();

    void addStream(const StreamConfig& config);
    void removeStream(int streamId);

    void initializeFromConfig(const ConfigService& configService);

    // Configuration handling
    void handleConfigUpdate(const ConfigUpdate& update);
    void enableAIStream(int streamId, bool enable);
    void updateAIStream(int streamId, const std::string& modelPath,
                        float confThreshold,
                        float nmsThreshold,
                        const std::string& modelName = "");
    /** 更新 AI 流的多模型配置（並行/串行） */
    void updateAIStreamWithStages(int streamId, const std::vector<ModelStageConfig>& stages, AIPipelineMode mode);

    // Accessors
    VideoStream* getStream(int streamId);
    std::vector<std::unique_ptr<VideoStream>>& getStreams() { return streams_; }
    const std::vector<std::unique_ptr<VideoStream>>& getStreams() const { return streams_; }

    // Multi-stream support
    void addMultiStreamConfig(const std::vector<StreamConfig>& configs);
    void updateStreamModel(int streamId, const std::string& modelPath);

    // Load streams from JSON configuration file（字段与 ax650 streams_config.json 完全一致）
    // mediamtxEndpoint: 可選的 MediaMTX 地址（IP:PORT 格式），如果為空則使用配置文件或環境變量
    bool loadStreamsFromConfig(const std::string& configPath, const std::string& mediamtxEndpoint = "",
                               bool offlineMode = false, const std::string& outputPath = "");

    // 更新 AI 檢測結果（由 AI 回調調用）
    void updateAIResult(int aiStreamId, const AI_RESULT_T* result);

    // 初始化 AI 流的 OSD 管理（在流啟動後調用）
    void initializeOSDForAIStream(int aiStreamId);

    void notifyAIError(int streamId, const std::string& error);

    // 离线模式：所有流都读到 EOF
    bool allEnded() const;

private:
    // 初始化 OSD 管理
    void initializeOSDForStream(int aiStreamId, VideoStream* aiStream);
    // 清理 OSD 管理
    void cleanupOSDForStream(int aiStreamId);

    std::vector<std::unique_ptr<VideoStream>> streams_;
    ConfigService& configService_;
    std::map<int, int> aiStreamMap_;  // streamId -> index in streams_
    mutable std::mutex streamsMutex_;

    // OSD 管理
    std::map<int, OSDAssociatedModel*> osdTargetMap_;  // AI streamId -> OSDAssociatedModel*
    std::mutex osdMapMutex_;  // 保護 osdTargetMap_ 的互斥鎖

    std::string runMode_ = "stream";
};

#endif // VIDEO_STREAM_MANAGER_H
