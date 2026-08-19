#ifndef VIDEO_STREAM_MANAGER_H
#define VIDEO_STREAM_MANAGER_H

#include <vector>
#include <map>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <memory>
#include <cstring>
#include "video_stream.h"
#include "config_service.h"
#include "../ai_interface.h"
#include "../../utilities/sample_log.h"

// 前向聲明
class IOSDRenderer;

// 參考 sample_multi_demux：OSD 管理結構
struct OSDAssociatedModel {
    std::weak_ptr<AIProcessor> aiProcessor;    // AI 處理器（使用 weak_ptr 避免循環引用）
    std::shared_ptr<IOSDRenderer> osdRenderer; // OSD 渲染器（由模型提供，如果沒有則使用默認渲染器）
    std::vector<pipeline_t*> pipes_need_osd;   // 需要 OSD 的 pipeline（主碼流等）
    AI_RESULT_T latestResult;                  // 最新的檢測結果
    std::mutex resultMutex;                    // 保護結果的互斥鎖
    std::condition_variable updateCv;          // [新增] 條件變量用於通知 OSD 更新和優雅關閉
    std::atomic<bool> shouldUpdateOSD{false};  // [改進] 使用 atomic 替代 volatile
    std::thread osdUpdateThread;               // OSD 更新線程
    std::atomic<bool> osdThreadRunning{false}; // [改進] 使用 atomic 替代 volatile
    
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
    
    // Frame processing
    void processFrame(pipeline_buffer_t* buffer, const std::string& inputSource = "");
    
    // Accessors
    VideoStream* getStream(int streamId);
    std::vector<VideoStream>& getStreams() { return streams_; }
    
    // Multi-stream support
    void addMultiStreamConfig(const std::vector<StreamConfig>& configs);
    void updateStreamModel(int streamId, const std::string& modelPath);
    
    // Load streams from JSON configuration file
    // mediamtxEndpoint: 可選的 MediaMTX 地址（IP:PORT 格式），如果為空則使用配置文件或環境變量
    bool loadStreamsFromConfig(const std::string& configPath, const std::string& mediamtxEndpoint = "");
    
    // 參考 sample_multi_demux：更新 AI 檢測結果（由 AI 回調調用）
    void updateAIResult(int aiStreamId, const AI_RESULT_T* result);
    
    // 初始化 AI 流的 OSD 管理（在流啟動後調用）
    void initializeOSDForAIStream(int aiStreamId);

    int getNextAvailableGroup(int baseGroup, int& vdecGroup, int& ivpsGroup);

    void notifyAIError(int streamId, const std::string& error) {
        // 实现错误通知逻辑
        ALOGE("AI error on stream %d: %s", streamId, error.c_str());
    }

private:
    std::vector<VideoStream> streams_;
    ConfigService& configService_;
    std::map<int, int> aiStreamMap_;  // streamId -> index in streams_
    mutable std::shared_mutex streamsMutex_; 
    
    // 參考 sample_multi_demux：OSD 管理
    std::map<int, OSDAssociatedModel*> osdTargetMap_;  // AI streamId -> OSDAssociatedModel*
    std::mutex osdMapMutex_;  // 保護 osdTargetMap_ 的互斥鎖
    
    // 參考 sample_multi_demux：OSD 更新線程函數
    static void osdUpdateThreadFunc(OSDAssociatedModel* model);
    
    // 初始化 OSD 管理（在 pipeline 創建後調用）
    void initializeOSDForStream(int aiStreamId, VideoStream* aiStream);
    
    // 清理 OSD 管理
    void cleanupOSDForStream(int aiStreamId);
};

#endif // VIDEO_STREAM_MANAGER_H

