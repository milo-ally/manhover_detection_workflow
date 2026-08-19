#include "video_stream_manager.h"
#include "ai_processor.h"
#include "../../utilities/sample_log.h"
#include "ax_ivps_api.h"
#include "../include/osd_renderer_interface.h"
#include <algorithm>
#include <set>
#include <map>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstring>
#include <mutex>
#include "../../utilities/json.hpp"

#ifndef AX_IVPS_INVALID_REGION_HANDLE
#define AX_IVPS_INVALID_REGION_HANDLE ((IVPS_RGN_HANDLE)-1)
#endif
#ifndef AX_IVPS_REGION_MAX_DISP_NUM
#define AX_IVPS_REGION_MAX_DISP_NUM (32)
#endif

namespace {
struct VsmPerSourceAgg {
    uint64_t window_start_ms = 0;
    uint64_t main_frames = 0;
    uint64_t ai_frames = 0;
    uint64_t main_cost_us = 0;
    uint64_t ai_skip_stride = 0;
    uint64_t ai_skip_big = 0;
};
std::mutex g_vsm_agg_mutex;
std::map<std::string, VsmPerSourceAgg> g_vsm_agg_by_source;
} // namespace

// 构造函数，初始化配置服务引用
VideoStreamManager::VideoStreamManager(ConfigService& configService)
    : configService_(configService) {
    // 設置全局 VideoStreamManager 指針，供 AI 回調使用
    VideoStream::setGlobalStreamManager(this);
}

// 析构函数
VideoStreamManager::~VideoStreamManager() {
    // 先停止所有 OSD 線程
    {
        std::lock_guard<std::mutex> osdLock(osdMapMutex_);
        for (auto& pair : osdTargetMap_) {
            if (pair.second) {
                // 使用 atomic store 並通知條件變量
                pair.second->osdThreadRunning.store(false, std::memory_order_release);
                pair.second->updateCv.notify_all();
                if (pair.second->osdUpdateThread.joinable()) {
                    pair.second->osdUpdateThread.join();
                }
                delete pair.second;
            }
        }
        osdTargetMap_.clear();
    }
    
    std::lock_guard<std::shared_mutex> lock(streamsMutex_);
    for (auto& stream : streams_) {
        stream.stop();
    }
    streams_.clear();
    aiStreamMap_.clear();
    
    // 清除全局指針
    VideoStream::setGlobalStreamManager(nullptr);
}

// 添加视频流，如果启用AI，则记录AI流ID
void VideoStreamManager::addStream(const StreamConfig& config) {
    std::unique_lock<std::shared_mutex> lock(streamsMutex_);
    
    // 检查是否已存在相同ID的流
    for (const auto& stream : streams_) {
        if (stream.getStreamId() == config.streamId) {
            ALOGW("Stream with ID %d already exists", config.streamId);
            return;
        }
    }
    
    streams_.emplace_back(config);
    
    // 區分主碼流和 AI 流：
    // 1. 主碼流：enableAI=true 但 isMediaMTXOutput=true，只是為了 OSD，不進行推理
    // 2. AI 流：enableAI=true 但 isMediaMTXOutput=false，進行 AI 推理
    if (config.enableAI && !config.isMediaMTXOutput) {
        // 這是 AI 流，註冊到 aiStreamMap_（即使初始時沒有 modelPath）
        // 因為模型可能通過 Web 配置動態設置
        aiStreamMap_[config.streamId] = streams_.size() - 1;
        ALOGI("AI stream registered with ID: %d (modelPath: %s)", 
              config.streamId, 
              config.modelPath.empty() ? "(empty, will be set via config)" : config.modelPath.c_str());
        
        VideoStream& newStream = streams_.back();
        // 多模型階段（並行或串行）
        if (!config.modelStages.empty()) {
            std::vector<std::unique_ptr<AIProcessor>> processors;
            for (size_t i = 0; i < config.modelStages.size(); i++) {
                const ModelStageConfig& stage = config.modelStages[i];
                std::string fullPath = stage.modelPath;
                if (fullPath.empty() || fullPath == "none") {
                    fullPath = configService_.getModelPath(stage.modelName.empty() ? stage.modelPath : stage.modelName);
                } else if (fullPath.find("/") == std::string::npos) {
                    fullPath = configService_.getModelPath(stage.modelName.empty() ? fullPath : stage.modelName);
                }
                if (fullPath.empty()) continue;
                nlohmann::json effParams = stage.params.is_object() ? stage.params : nlohmann::json::object();
                if (!stage.pluginPath.empty() && !effParams.contains("plugin"))
                    effParams["plugin"] = stage.pluginPath;
                if (!effParams.contains("conf_threshold")) effParams["conf_threshold"] = stage.confThreshold;
                if (!effParams.contains("nms_threshold")) effParams["nms_threshold"] = stage.nmsThreshold;
                std::unique_ptr<AIProcessor> p = std::make_unique<AIProcessor>(fullPath, stage.modelName, effParams);
                p->setThresholds(stage.confThreshold, stage.nmsThreshold);
                processors.push_back(std::move(p));
                ALOGN("[VideoStreamManager] Stream %d stage %zu: %s", config.streamId, i, fullPath.c_str());
            }
            if (!processors.empty()) {
                newStream.setAIProcessors(std::move(processors));
                ALOGI("AI stream %d loaded with %zu model stages (mode=%s)", config.streamId, config.modelStages.size(),
                      config.aiPipelineMode == AIPipelineMode::Serial ? "serial" : "parallel");
            }
        } else if (!config.modelPath.empty() && config.modelPath != "none") {
            // 單模型向後兼容
            std::string fullModelPath = config.modelPath;
            if (fullModelPath.find("/") == std::string::npos) {
                fullModelPath = configService_.getModelPath(config.modelName.empty() ? fullModelPath : config.modelName);
            }
            if (!fullModelPath.empty()) {
                ALOGN("[VideoStreamManager] Creating AIProcessor for stream %d...", config.streamId);
                nlohmann::json effParams = nlohmann::json::object();
                if (!config.pluginPath.empty()) effParams["plugin"] = config.pluginPath;
                effParams["conf_threshold"] = config.confThreshold;
                effParams["nms_threshold"] = config.nmsThreshold;
                newStream.setAIProcessor(std::make_unique<AIProcessor>(fullModelPath, config.modelName, effParams));
                newStream.setThresholds(config.confThreshold, config.nmsThreshold);
                ALOGI("AI stream %d loaded with model: %s", config.streamId, fullModelPath.c_str());
            }
        } else {
            ALOGN("[VideoStreamManager] AI stream %d registered without model (will be set via config)", config.streamId);
        }
    } else if (config.enableAI && config.isMediaMTXOutput) {
        // 主碼流：enableAI=true 但沒有 modelPath，只是為了 OSD
        ALOGI("Main stream %d registered with OSD support (no AI inference)", config.streamId);
    }
}

// 移除指定ID的视频流
void VideoStreamManager::removeStream(int streamId) {
    std::unique_lock<std::shared_mutex> lock(streamsMutex_);
    
    auto it = std::find_if(streams_.begin(), streams_.end(),
                            [streamId](const VideoStream& s) {
                                return s.getStreamId() == streamId;
                            });
    if (it != streams_.end()) {
        it->stop();
        streams_.erase(it);
        aiStreamMap_.clear();
        for(size_t i = 0; i < streams_.size(); ++i) {
            if(streams_[i].isAIEnabled() && !streams_[i].isMediaMTXOutput()) {
                aiStreamMap_[streams_[i].getStreamId()] = i;
            }
        }
        ALOGI("Stream %d removed and map re-indexed", streamId);
    }
}

// 处理配置更新，根据配置决定是否启用AI流或更新AI流参数
void VideoStreamManager::handleConfigUpdate(const ConfigUpdate& update) {
    ALOGN("[VideoStreamManager] handleConfigUpdate called: model=%s, valid=%d, streamId=%d, cameraId=%d", 
          update.modelName.c_str(), update.valid ? 1 : 0, update.streamId, update.cameraId);
    
    // 支持按流更新：如果指定了 streamId，只更新該流；否則更新所有流（向後兼容）
    int targetStreamId = update.streamId;
    
    // 如果指定了 cameraId，需要映射到對應的 AI 流 streamId
    // 攝像頭 ID 對應主碼流的 streamId，AI 流的 streamId = 主碼流 streamId + 1
    if (update.cameraId > 0 && targetStreamId < 0) {
        // cameraId 1 -> 主碼流 streamId=1, AI 流 streamId=2
        // cameraId 2 -> 主碼流 streamId=3, AI 流 streamId=4
        // 規律：AI 流 streamId = (cameraId - 1) * 2 + 2 = cameraId * 2
        targetStreamId = update.cameraId * 2;
        ALOGN("[VideoStreamManager] Mapped cameraId=%d to AI streamId=%d", update.cameraId, targetStreamId);
    }
    
    std::vector<int> streamIds;
    if (targetStreamId > 0) {
        // 按流更新：只更新指定的流
        {
            std::shared_lock<std::shared_mutex> lock(streamsMutex_);
            if (aiStreamMap_.find(targetStreamId) != aiStreamMap_.end()) {
                streamIds.push_back(targetStreamId);
            } else {
                ALOGW("[VideoStreamManager] Stream %d not found in aiStreamMap_", targetStreamId);
                return;
            }
        }
        ALOGN("[VideoStreamManager] Updating specific AI stream %d", targetStreamId);
    } else {
        // 全局更新（/dev/shm/ai_config.json 的舊格式）：
        // 不要覆蓋命令列 -m 指定的模型，避免啟動後模型狀態失控。
        // UI 若要控制，必須走 cameraId/streamId 的按流更新。
        size_t skipped_cmdline = 0;
        {
            std::shared_lock<std::shared_mutex> lock(streamsMutex_);
            for (auto& pair : aiStreamMap_) {
                int aiStreamId = pair.first;
                size_t idx = pair.second;
                if (idx >= streams_.size()) continue;
                if (streams_[idx].isCommandLineModel()) {
                    skipped_cmdline++;
                    continue;
                }
                streamIds.push_back(aiStreamId);
            }
        }
        ALOGN("[VideoStreamManager] Global update will affect %zu AI streams (skipped command-line models: %zu)",
              streamIds.size(), skipped_cmdline);
    }
    
    // 在循环中处理每个流，确保不在耗时操作时持有Manager的锁
    if (!update.valid || update.modelName == "none") {
        // 禁用指定的 AI 流
        for (int streamId : streamIds) {
            enableAIStream(streamId, false);
        }
    } else {
        // 更新指定的 AI 流
        for (int streamId : streamIds) {
            // 按流更新（cameraId/streamId）視為「UI 控制」，允許覆蓋命令列模型
            // 必須在調用 updateAIStream 之前清除命令列標記，確保 updateAIStream 讀取時看到的是清除後的狀態
            if (targetStreamId > 0) {
                {
                    std::shared_lock<std::shared_mutex> lock(streamsMutex_);
                    auto it = aiStreamMap_.find(streamId);
                    if (it != aiStreamMap_.end() && it->second < streams_.size()) {
                        bool wasCmdline = streams_[it->second].isCommandLineModel();
                        streams_[it->second].clearCommandLineModelFlag();
                        ALOGN("[VideoStreamManager] Cleared command-line flag for stream %d (wasCmdline=%d)", 
                              streamId, wasCmdline ? 1 : 0);
                    }
                }
                // 確保清除操作完成後再調用 updateAIStream
                // 使用一個小的延遲或確保鎖釋放後再繼續
            }
            // updateAIStream 內部不會獲取 streamsMutex_，但 initializeOSDForStream 會
            // 所以這裡不需要持有鎖，避免死鎖
            if (!update.modelStages.empty()) {
                // 多模型配置（並行/串行）
                ALOGN("[VideoStreamManager] Calling updateAIStreamWithStages for stream %d: %zu stages (mode=%s)", 
                      streamId, update.modelStages.size(),
                      update.aiPipelineMode == AIPipelineMode::Serial ? "serial" : "parallel");
                updateAIStreamWithStages(streamId, update.modelStages, update.aiPipelineMode);
            } else {
                // 單模型配置（向後兼容）
                ALOGN("[VideoStreamManager] Calling updateAIStream for stream %d: model=%s, path=%s", 
                      streamId, update.modelName.c_str(), update.modelPath.c_str());
                updateAIStream(streamId, update.modelPath, update.confThreshold, update.nmsThreshold, update.modelName);
            }
        }
    }
    ALOGN("[VideoStreamManager] handleConfigUpdate completed");
}

// 启用或禁用指定AI流
// Refactored to ensure thread safety while keeping heavy I/O outside the lock
void VideoStreamManager::enableAIStream(int streamId, bool enable) {
    std::string modelPathToLoad;
    int streamIndex = -1;

    {
        std::shared_lock<std::shared_mutex> lock(streamsMutex_);
        auto it = aiStreamMap_.find(streamId);
        if (it == aiStreamMap_.end()) {
            ALOGW("[VideoStreamManager] enableAIStream: No AI stream registered with ID %d", streamId);
            return;
        }
        if (it->second >= (int)streams_.size()) {
             ALOGE("[VideoStreamManager] enableAIStream: Index out of bounds");
             return;
        }

        streamIndex = it->second;

        if (enable) {
            std::string modelName = configService_.getCurrentModel();
            if (modelName == "none" || modelName.empty()) {
                return;
            }
            modelPathToLoad = configService_.getModelPath(modelName);
        }
    }

    std::unique_ptr<AIProcessor> newProcessor = nullptr;
    if (enable && !modelPathToLoad.empty()) {
        std::string modelNameForPlugin = configService_.getCurrentModel();
        ALOGN("[VideoStreamManager] enableAIStream: Pre-loading model: %s (name=%s)", modelPathToLoad.c_str(), modelNameForPlugin.c_str());
        nlohmann::json effParams = nlohmann::json::object();
        effParams["conf_threshold"] = configService_.getConfThreshold();
        effParams["nms_threshold"] = configService_.getNmsThreshold();
        newProcessor = std::make_unique<AIProcessor>(modelPathToLoad, modelNameForPlugin, effParams);
    }

    {
        std::shared_lock<std::shared_mutex> lock(streamsMutex_);
        if (streamIndex >= (int)streams_.size() || streams_[streamIndex].getStreamId() != streamId) {
             ALOGW("Stream %d disappeared or moved during model load", streamId);
             return;
        }
        
        VideoStream& aiStream = streams_[streamIndex];

        if (enable) {
             if (newProcessor) {
                 aiStream.setAIEnabled(true);  
                 aiStream.setAIProcessor(std::move(newProcessor));
                 initializeOSDForStream(streamId, &aiStream);
             }
        } else {
             ALOGN("[VideoStreamManager] enableAIStream: Disabling stream %d", streamId);
             aiStream.clearCommandLineModelFlag();
             
             cleanupOSDForStream(streamId);
             aiStream.clearOSD();
             
             std::string src = aiStream.getInputSource();
             for (auto& s : streams_) {
                 if (s.getStreamId() != streamId && s.getInputSource() == src) {
                     s.clearOSD();
                 }
             }
             
             aiStream.setAIEnabled(false);
             aiStream.setAIProcessor(nullptr);
        }
    }
}

// 更新AI流的模型和阈值参数
void VideoStreamManager::updateAIStream(int streamId, const std::string& modelPath, 
                                      float confThreshold, 
                                      float nmsThreshold,
                                      const std::string& modelName) {
    std::string currentModelPath;
    bool isCommandLine = false;
    VideoStream* targetStream = nullptr;

    {
        std::shared_lock<std::shared_mutex> lock(streamsMutex_);
        auto it = aiStreamMap_.find(streamId);
        if (it != aiStreamMap_.end() && it->second < (int)streams_.size()) {
            targetStream = &streams_[it->second];
            currentModelPath = targetStream->getModelPath();
            isCommandLine = targetStream->isCommandLineModel();
        }
    }

    if (!targetStream) {
        ALOGW("[VideoStreamManager] updateAIStream: Stream %d not found", streamId);
        return;
    }

    ALOGN("[VideoStreamManager] updateAIStream: streamId=%d, isCommandLine=%d, currentPath=%s, newPath=%s", 
          streamId, isCommandLine ? 1 : 0, currentModelPath.c_str(), modelPath.c_str());

    // 如果命令列標記存在且模型路徑相同，只更新閾值（避免重複加載）
    // 這個檢查應該在 handleConfigUpdate 清除標記之後執行，所以通常 isCommandLine 應該是 false
    if (isCommandLine && !modelPath.empty() && modelPath == currentModelPath) {
        ALOGN("[VideoStreamManager] updateAIStream: Command-line model unchanged (path=%s), only updating thresholds", modelPath.c_str());
        targetStream->setAIEnabled(true);   
        targetStream->setThresholds(confThreshold, nmsThreshold);
        return;
    }
    
    // 如果命令列標記存在但模型路徑不同，說明需要切換模型，應該清除標記並更新
    if (isCommandLine && !modelPath.empty() && modelPath != currentModelPath) {
        ALOGN("[VideoStreamManager] updateAIStream: Command-line model path changed (%s -> %s), clearing flag and updating", 
              currentModelPath.c_str(), modelPath.c_str());
        // 清除命令列標記，允許更新
        {
            std::shared_lock<std::shared_mutex> lock(streamsMutex_);
            auto it = aiStreamMap_.find(streamId);
            if (it != aiStreamMap_.end() && it->second < (int)streams_.size()) {
                streams_[it->second].clearCommandLineModelFlag();
            }
        }
        // 繼續執行後續的模型更新邏輯
    }
    
    std::string normalizedModelPath = modelPath;
    std::string normalizedCurrentPath = currentModelPath;
    
    // "../models/" 的長度是 10，所以應該使用 substr(10)
    if (normalizedModelPath.find("../models/") == 0) normalizedModelPath = normalizedModelPath.substr(10);
    if (normalizedCurrentPath.find("../models/") == 0) normalizedCurrentPath = normalizedCurrentPath.substr(10);

    if (modelPath.empty()) {
        enableAIStream(streamId, false);
        return;
    }

    std::unique_ptr<AIProcessor> newProcessor = nullptr;
    bool needModelReload = (normalizedModelPath != normalizedCurrentPath);
    
    ALOGN("[VideoStreamManager] updateAIStream: needModelReload=%d, normalizedCurrent=%s, normalizedNew=%s", 
          needModelReload ? 1 : 0, normalizedCurrentPath.c_str(), normalizedModelPath.c_str());
    
    if (needModelReload) {
        ALOGN("[VideoStreamManager] Pre-loading model: %s (name=%s)", modelPath.c_str(), modelName.c_str());
        nlohmann::json effParams = nlohmann::json::object();
        effParams["conf_threshold"] = confThreshold;
        effParams["nms_threshold"] = nmsThreshold;
        newProcessor = std::make_unique<AIProcessor>(modelPath, modelName, effParams);
    }

    {
        std::shared_lock<std::shared_mutex> lock(streamsMutex_);
        auto it = aiStreamMap_.find(streamId);
        if (it == aiStreamMap_.end()) {
            ALOGW("[VideoStreamManager] updateAIStream: Stream %d not found in map", streamId);
            return;
        }

        VideoStream& stream = streams_[it->second];
        
        if (newProcessor) {
            ALOGN("[VideoStreamManager] updateAIStream: Setting new model processor for stream %d", streamId);
            stream.setAIEnabled(true);  
            stream.setAIProcessor(std::move(newProcessor));
            stream.setModelPath(modelPath);  // 同步更新 config_.modelPath
            if (!modelName.empty()) stream.setModelName(modelName);  // 人員聚集與人員偵測共用 path 時區分插件
            cleanupOSDForStream(streamId);
            stream.clearOSD();
            
            std::string src = stream.getInputSource();
            for (auto& s : streams_) {
                if (s.getInputSource() == src) {
                     s.clearOSD();
                }
            }

            initializeOSDForStream(streamId, &stream);
        } else {
            // 同模型但更新阈值，确保 AI 仍启用
            ALOGN("[VideoStreamManager] updateAIStream: Model unchanged, only updating thresholds for stream %d", streamId);
            stream.setAIEnabled(true);  
        }
        
        stream.setThresholds(confThreshold, nmsThreshold);
    }
    ALOGN("[VideoStreamManager] updateAIStream completed for stream %d", streamId);
}

// 更新 AI 流的多模型配置（並行/串行）
void VideoStreamManager::updateAIStreamWithStages(int streamId, const std::vector<ModelStageConfig>& stages, AIPipelineMode mode) {
    VideoStream* targetStream = nullptr;
    {
        std::shared_lock<std::shared_mutex> lock(streamsMutex_);
        auto it = aiStreamMap_.find(streamId);
        if (it == aiStreamMap_.end() || it->second >= (int)streams_.size()) {
            ALOGW("[VideoStreamManager] updateAIStreamWithStages: Stream %d not found", streamId);
            return;
        }
        targetStream = &streams_[it->second];
    }
    
    if (stages.empty()) {
        enableAIStream(streamId, false);
        return;
    }
    
    std::vector<std::unique_ptr<AIProcessor>> processors;
    std::vector<ModelStageConfig> stagesAligned;
    stagesAligned.reserve(stages.size());
    for (const auto& stage : stages) {
        if (stage.modelPath.empty() || stage.modelPath == "none") continue;
        nlohmann::json effParams = stage.params.is_object() ? stage.params : nlohmann::json::object();
        if (!stage.pluginPath.empty() && !effParams.contains("plugin"))
            effParams["plugin"] = stage.pluginPath;
        if (!effParams.contains("conf_threshold")) effParams["conf_threshold"] = stage.confThreshold;
        if (!effParams.contains("nms_threshold")) effParams["nms_threshold"] = stage.nmsThreshold;
        std::unique_ptr<AIProcessor> p = std::make_unique<AIProcessor>(stage.modelPath, stage.modelName, effParams);
        p->setThresholds(stage.confThreshold, stage.nmsThreshold);
        processors.push_back(std::move(p));
        stagesAligned.push_back(stage);
        ALOGN("[VideoStreamManager] Stream %d stage: %s (conf=%.2f, nms=%.2f, roi=%d)", 
              streamId, stage.modelPath.c_str(), stage.confThreshold, stage.nmsThreshold, stage.roiFromPrevious ? 1 : 0);
    }
    
    if (processors.empty()) {
        ALOGW("[VideoStreamManager] updateAIStreamWithStages: No valid processors for stream %d", streamId);
        return;
    }
    if (!stagesAligned.empty()) stagesAligned[0].roiFromPrevious = false;

    {
        std::shared_lock<std::shared_mutex> lock(streamsMutex_);
        auto it = aiStreamMap_.find(streamId);
        if (it == aiStreamMap_.end() || it->second >= (int)streams_.size()) return;
        VideoStream& stream = streams_[it->second];
        
        stream.setAIEnabled(true);
        stream.setAIProcessors(std::move(processors));
        // 必須與 processors 一一對應，否則 InferenceManager 中 stages_.size()!=engines_.size() 會走舊串行路徑且與 roi 標記不一致
        stream.setModelStages(stagesAligned, mode);
        
        cleanupOSDForStream(streamId);
        stream.clearOSD();
        std::string src = stream.getInputSource();
        for (auto& s : streams_) {
            if (s.getInputSource() == src) s.clearOSD();
        }
        initializeOSDForStream(streamId, &stream);
    }
    
    ALOGN("[VideoStreamManager] updateAIStreamWithStages completed for stream %d: %zu stages (mode=%s)", 
          streamId, stages.size(), mode == AIPipelineMode::Serial ? "serial" : "parallel");
}

// 处理每一帧数据，只分发给使用相同输入源的流
// 對每個匹配的 stream 各調用一次 processFrame -> user_input(pipe, 1, buffer)。
// 注意：同一輸入源的主碼流與 AI 流共用同一 VDEC 組，故會對同一 VDEC 組 SendStream 兩次（user_input 內用 tmp_ 去重僅在同一調用內有效，不同調用會重複送）。
// 若改為「每 VDEC 組只送一幀」會導致綠屏/藍屏，推測 AX650 上同一 VDEC 多 Link 到多個 IVPS 時，需每路各送一幀才能正確出圖，故保留按 stream 各送一次。
void VideoStreamManager::processFrame(pipeline_buffer_t* buffer, const std::string& inputSource) {
    std::shared_lock<std::shared_mutex> lock(streamsMutex_);
    
    static int frame_count = 0;
    static int last_log_frame = 0;
    if (++frame_count % 300 == 0) {
        ALOGN("[VideoStreamManager] Distributing frame #%d from source: %s", frame_count, inputSource.c_str());
        last_log_frame = frame_count;
    }
    
    int matched_streams = 0;
    int matched_ai_streams = 0;
    int matched_main_streams = 0;
    for (auto& stream : streams_) {
        if (!stream.isRunning()) continue;
        if (!inputSource.empty() && stream.getInputSource() != inputSource) continue;
        if (stream.isMediaMTXOutput()) matched_main_streams++;
        else matched_ai_streams++;
    }
    const int total_matched = matched_ai_streams + matched_main_streams;
    // 收斂策略：保留主碼流優先，AI 僅做輕量降頻，不再使用長抑制窗口
    int dynamic_ai_stride = 6;
    if (total_matched >= 12) dynamic_ai_stride = 12;
    else if (total_matched >= 8) dynamic_ai_stride = 10;
    else if (total_matched >= 6) dynamic_ai_stride = 8;
    const int oversized_ai_frame_threshold = (total_matched >= 8) ? (96 * 1024) : (128 * 1024);
    static int ai_stride_counter[256] = {0};

    // 第一階段：只送主碼流，確保主鏈路優先
    auto t_main_start = std::chrono::steady_clock::now();
    for (auto& stream : streams_) {
        if (!stream.isRunning()) continue;
        if (!inputSource.empty() && stream.getInputSource() != inputSource) continue;
        if (!stream.isMediaMTXOutput()) continue;

        matched_streams++;
        if (frame_count == last_log_frame) {
            ALOGN("[VideoStreamManager] Sending frame to stream %d (inputSource=%s)",
                  stream.getStreamId(), stream.getInputSource().c_str());
        }
        stream.processFrame(buffer);
    }
    auto main_cost_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_main_start).count();
    uint64_t ai_skip_stride = 0;
    uint64_t ai_skip_big = 0;
    // 第二階段：AI 分發（受節流與抑制窗口控制）
    for (auto& stream : streams_) {
        if (!stream.isRunning()) continue;
        if (!inputSource.empty() && stream.getInputSource() != inputSource) continue;
        if (stream.isMediaMTXOutput()) continue;

        // 來源關鍵幀偶發超大（例如 200KB+）時，若同時餵 AI 支路會放大共享 VDEC 壓力並導致規律卡頓。
        // 對 AI 支路直接跳過超大壓縮幀，優先保證主碼流平滑。
        if (buffer && buffer->n_size > oversized_ai_frame_threshold) {
            ai_skip_big++;
            continue;
        }
        int sid = stream.getStreamId();
        int idx = (sid >= 0 && sid < 256) ? sid : 0;
        if ((++ai_stride_counter[idx] % dynamic_ai_stride) != 0) {
            ai_skip_stride++;
            continue;
        }

        matched_streams++;
        if (frame_count == last_log_frame) {
            ALOGN("[VideoStreamManager] Sending frame to stream %d (inputSource=%s)", 
                  stream.getStreamId(), stream.getInputSource().c_str());
        }
        stream.processFrame(buffer);
    }
    const uint64_t main_cost_us = static_cast<uint64_t>(main_cost_ms * 1000);
    const uint64_t add_main = static_cast<uint64_t>(matched_main_streams);
    const uint64_t add_ai = static_cast<uint64_t>(matched_streams - matched_main_streams);

    {
        std::lock_guard<std::mutex> agg_lock(g_vsm_agg_mutex);
        VsmPerSourceAgg& a = g_vsm_agg_by_source[inputSource];
        a.main_cost_us += main_cost_us;
        a.main_frames += add_main;
        a.ai_frames += add_ai;
        a.ai_skip_stride += ai_skip_stride;
        a.ai_skip_big += ai_skip_big;

        const auto now = std::chrono::steady_clock::now();
        const uint64_t now_ms = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
        if (a.window_start_ms == 0) {
            a.window_start_ms = now_ms;
        } else if (now_ms - a.window_start_ms >= 1000) {
            ALOGN("[LOWLAT][VSM] src=%s main_out=%llu ai_out=%llu main_cost_avg_us=%llu ai_skip_stride=%llu ai_skip_big=%llu ai_stride=%d",
                  inputSource.c_str(),
                  (unsigned long long)a.main_frames,
                  (unsigned long long)a.ai_frames,
                  (unsigned long long)((a.main_frames > 0) ? (a.main_cost_us / a.main_frames) : 0),
                  (unsigned long long)a.ai_skip_stride,
                  (unsigned long long)a.ai_skip_big,
                  dynamic_ai_stride);
            a.window_start_ms = now_ms;
            a.main_frames = 0;
            a.ai_frames = 0;
            a.main_cost_us = 0;
            a.ai_skip_stride = 0;
            a.ai_skip_big = 0;
        }
    }
    
    if (matched_streams == 0 && frame_count == last_log_frame) {
        ALOGW("[VideoStreamManager] No matching streams for source: %s", inputSource.c_str());
    } else if (frame_count == last_log_frame) {
        ALOGN("[VideoStreamManager] matched total=%d main=%d ai=%d ai_stride=%d ai_max_frame=%dKB",
              total_matched, matched_main_streams, matched_ai_streams, dynamic_ai_stride,
              oversized_ai_frame_threshold / 1024);
    }
}

// 根据流ID获取对应的VideoStream对象指针
VideoStream* VideoStreamManager::getStream(int streamId) {
    std::shared_lock<std::shared_mutex> lock(streamsMutex_); // 使用读锁
    auto it = std::find_if(streams_.begin(), streams_.end(),
                          [streamId](const VideoStream& s) { 
                              return s.getStreamId() == streamId; 
                          });
    return (it != streams_.end()) ? &(*it) : nullptr;
}

// 添加多流配置
void VideoStreamManager::addMultiStreamConfig(const std::vector<StreamConfig>& configs) {
    std::unique_lock<std::shared_mutex> lock(streamsMutex_); // 使用写锁
    
    int baseGroup = 0;
    for (const auto& config : configs) {
        StreamConfig newConfig = config;
        
        // 自动分配组号
        int vdecGroup, ivpsGroup;
        getNextAvailableGroup(baseGroup, vdecGroup, ivpsGroup);
        newConfig.vdecGroup = vdecGroup;
        newConfig.ivpsGroup = ivpsGroup;
        baseGroup = ivpsGroup + 1;
        
        addStream(newConfig);
    }
}

// 更新流的模型
void VideoStreamManager::updateStreamModel(int streamId, const std::string& modelPath) {
    // 简单包装 updateAIStream，复用优化后的逻辑
    updateAIStream(streamId, modelPath, 0.5f, 0.45f);
}

// 获取下一个可用的组号
int VideoStreamManager::getNextAvailableGroup(int baseGroup, int& vdecGroup, int& ivpsGroup) {
    std::set<int> usedVdecGroups;
    std::set<int> usedIvpsGroups;
    
    for (const auto& stream : streams_) {
        usedVdecGroups.insert(stream.getVdecGroup());
        usedIvpsGroups.insert(stream.getIvpsGroup());
    }
    
    // 找到第一个未使用的组号
    vdecGroup = baseGroup;
    while (usedVdecGroups.find(vdecGroup) != usedVdecGroups.end()) {
        vdecGroup++;
    }
    
    ivpsGroup = baseGroup;
    while (usedIvpsGroups.find(ivpsGroup) != usedIvpsGroups.end()) {
        ivpsGroup++;
    }
    
    return ivpsGroup;
}

// 從JSON配置文件加載多流配置
bool VideoStreamManager::loadStreamsFromConfig(const std::string& configPath, const std::string& mediamtxEndpoint) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        ALOGE("Failed to open streams config file: %s", configPath.c_str());
        return false;
    }
    
    try {
        nlohmann::json config;
        file >> config;
        
        // 設置 MediaMTX 地址（優先級：命令行參數 > 環境變量 > 配置文件 > 預設值）
        std::string mediamtxHost = "127.0.0.1";
        std::string mediamtxPort = "8000";
        
        if (!mediamtxEndpoint.empty()) {
            // 命令行參數（最高優先級）
            size_t colon_pos = mediamtxEndpoint.find(':');
            if (colon_pos != std::string::npos) {
                mediamtxHost = mediamtxEndpoint.substr(0, colon_pos);
                mediamtxPort = mediamtxEndpoint.substr(colon_pos + 1);
            } else {
                mediamtxHost = mediamtxEndpoint;
            }
            ALOGN("[VideoStreamManager] Using MediaMTX endpoint from command line: %s:%s", mediamtxHost.c_str(), mediamtxPort.c_str());
        } else {
            // 讀取配置文件中的全局設置
            if (config.contains("global_settings")) {
                auto& global = config["global_settings"];
                if (global.contains("mediamtx_host")) {
                    mediamtxHost = global["mediamtx_host"];
                }
                if (global.contains("mediamtx_port")) {
                    mediamtxPort = global["mediamtx_port"];
                }
            }
            
            // 檢查環境變量（優先級高於配置文件）
            const char* envHost = getenv("MEDIAMTX_HOST");
            const char* envPort = getenv("MEDIAMTX_RTP_PORT");
            if (envHost) mediamtxHost = envHost;
            if (envPort) mediamtxPort = envPort;
        }
        
        // 讀取流配置
        if (!config.contains("streams") || !config["streams"].is_array()) {
            ALOGE("Invalid config format: missing 'streams' array");
            return false;
        }
        
        int streamIdBase = 1;
        // 使用動態分配確保資源不衝突
        int baseGroup = 0;
        
        // 解析基礎端口
        uint16_t mediamtx_base_port = 8000;
        size_t colon_pos = mediamtxPort.find(':');
        if (colon_pos != std::string::npos) {
            mediamtx_base_port = (uint16_t)atoi(mediamtxPort.substr(colon_pos + 1).c_str());
        } else {
            mediamtx_base_port = (uint16_t)atoi(mediamtxPort.c_str());
        }
        
        int inputSourceIndex = 0;  // 輸入源索引（用於端口分配）
        // 端雲精度比對用 raw 主碼流：預設關閉，需比對時在該路或 global_settings 開啟 enable_raw_stream
        bool defaultEnableRawStream = false;
        if (config.contains("global_settings") && config["global_settings"].is_object() &&
            config["global_settings"].contains("enable_raw_stream")) {
            defaultEnableRawStream = config["global_settings"]["enable_raw_stream"].get<bool>();
        }

        // Raw 主碼流（無 OSD）候選：用同一 inputSource，但推到 *_raw 的 MediaMTX 路徑/端口。
        // 注意：為了不破壞既有 cameraId->AI streamId 的編號規則，我們把 raw stream 放在所有 AI stream 建完後再新增（避免 streamIdBase 被插入）。
        struct RawCandidate {
            std::string inputSource;
            uint16_t rawPort;
            int inputSourceIndex;
            int outputWidth;
            int outputHeight;
            int fps;
            int sharedVdecGroup;  // 與該路 OSD 主碼流/AI 流共用 VDEC，避免同一路 RTSP 解兩次造成花屏與頻寬翻倍
        };
        std::vector<RawCandidate> rawCandidates;

        for (const auto& streamConfig : config["streams"]) {
            std::string inputSource = streamConfig["input_source"];
            std::string inputCodec = streamConfig.contains("input_codec")
                ? streamConfig["input_codec"].get<std::string>()
                : "auto";

            // enable_ai：控制該 camera 的 OSD/AI 模組啟用；osd 主碼流需要 enableAI=true 才會建立 OSD region
            bool enableAI = streamConfig.contains("enable_ai") ? streamConfig["enable_ai"].get<bool>() : true;
            bool enableRawStream = streamConfig.contains("enable_raw_stream")
                ? streamConfig["enable_raw_stream"].get<bool>()
                : defaultEnableRawStream;
            
            // 為每個輸入源動態分配 VDEC 組（相同輸入源的流共用 VDEC 組）
            int vdecGroup, ivpsGroup;
            getNextAvailableGroup(baseGroup, vdecGroup, ivpsGroup);
            int mainVdecGroup = vdecGroup;  // 主碼流的 VDEC 組
            int mainIvpsGroup = ivpsGroup;  // 主碼流的 IVPS 組
            baseGroup = ivpsGroup + 1;  // 更新基礎組號，為下一個輸入源準備
            
            ALOGN("[Config] Allocated resources for input source '%s' (index %d): VDEC_Group=%d, Main_IVPS_Group=%d", 
                  inputSource.c_str(), inputSourceIndex, mainVdecGroup, mainIvpsGroup);
            
            // 配置主碼流（MediaMTX推送）
            StreamConfig rtspConfig;
            rtspConfig.streamId = streamIdBase++;
            rtspConfig.inputSource = inputSource;
            rtspConfig.inputCodec = inputCodec;
            rtspConfig.ivpsGroup = mainIvpsGroup;
            rtspConfig.vdecGroup = mainVdecGroup;  // 與同輸入源的 AI 流共用 VDEC 組
            rtspConfig.isMediaMTXOutput = true;
            // OSD 主碼流：為了確保在 AI 推理啟用後能正常畫框，main pipeline 必须先建好 OSD region
            // （而 raw 主碼流由后面 rawConfig：enableAI=false 保证不含 OSD）。
            rtspConfig.enableAI = true;
            // 為每個輸入源的主碼流分配不同的端口：基礎端口 + inputSourceIndex * 2
            // 這樣可以確保每個輸入源的主碼流對應正確的 MediaMTX 路徑（live1, live2, ...）
            uint16_t stream_port = mediamtx_base_port + inputSourceIndex * 2;
            rtspConfig.mediamtxEndpoint = mediamtxHost + ":" + std::to_string(stream_port);
            ALOGN("[Config] Stream %d (input source index %d) will push to MediaMTX: %s (expected path: live%d)", 
                  rtspConfig.streamId, inputSourceIndex, rtspConfig.mediamtxEndpoint.c_str(), inputSourceIndex + 1);
            
            if (streamConfig.contains("output_width")) {
                rtspConfig.outputWidth = streamConfig["output_width"];
            } else {
                rtspConfig.outputWidth = 1920;
            }
            
            if (streamConfig.contains("output_height")) {
                rtspConfig.outputHeight = streamConfig["output_height"];
            } else {
                rtspConfig.outputHeight = 1080;
            }
            
            if (streamConfig.contains("fps")) {
                rtspConfig.fps = streamConfig["fps"];
            } else {
                rtspConfig.fps = 30;
            }
            
            addStream(rtspConfig);
            ALOGN("[Config] Loaded main stream: stream_id=%d, source=%s, VDEC_Group=%d, IVPS_Group=%d, mediamtx_endpoint=%s", 
                  rtspConfig.streamId,
                  inputSource.c_str(),
                  rtspConfig.vdecGroup,
                  rtspConfig.ivpsGroup,
                  rtspConfig.mediamtxEndpoint.c_str());
            
            // 配置AI流（如果啟用）
            if (enableAI) {
                // 為 AI 流分配 IVPS 組（與主碼流共用 VDEC 組）
                // 主碼流已經被添加，所以 getNextAvailableGroup 會自動跳過已使用的組
                int aiIvpsGroup;
                int dummyVdecGroup;
                getNextAvailableGroup(baseGroup, dummyVdecGroup, aiIvpsGroup);
                baseGroup = aiIvpsGroup + 1;  // 更新基礎組號，為下一個輸入源準備
                // AI 流使用與主碼流相同的 VDEC 組（因為來自同一個輸入源）
                
                ALOGN("[Config] Allocated AI stream resources: VDEC_Group=%d (shared with main), AI_IVPS_Group=%d", 
                      mainVdecGroup, aiIvpsGroup);
                
                StreamConfig aiConfig;
                aiConfig.streamId = streamIdBase++;
                aiConfig.inputSource = inputSource;
                aiConfig.inputCodec = inputCodec;
                aiConfig.ivpsGroup = aiIvpsGroup;
                aiConfig.vdecGroup = mainVdecGroup;  // 與主碼流共用 VDEC 組（相同輸入源）
                aiConfig.enableAI = true;
                if (streamConfig.contains("plugin") && streamConfig["plugin"].is_string())
                    aiConfig.pluginPath = streamConfig["plugin"];
                
                // 模型配置：支援多模型（並行）或串行階段（models 數組可含 roi_from_previous）
                bool modelConfigured = false;
                if (streamConfig.contains("models") && streamConfig["models"].is_array() &&
                    !streamConfig["models"].empty()) {
                    aiConfig.modelStages.clear();
                    bool anyRoiFromPrevious = false;
                    for (const auto& modelEl : streamConfig["models"]) {
                        ModelStageConfig stage;
                        if (modelEl.contains("name")) stage.modelName = modelEl["name"];
                        if (modelEl.contains("plugin") && modelEl["plugin"].is_string())
                            stage.pluginPath = modelEl["plugin"];
                        else if (streamConfig.contains("plugin") && streamConfig["plugin"].is_string())
                            stage.pluginPath = streamConfig["plugin"];
                        if (modelEl.contains("path") && !modelEl["path"].is_null()) {
                            stage.modelPath = modelEl["path"];
                            if (stage.modelPath.find("/") == std::string::npos || stage.modelPath.find("../") == 0) {
                                if (!stage.modelName.empty()) {
                                    std::string fp = configService_.getModelPath(stage.modelName);
                                    if (!fp.empty()) stage.modelPath = fp;
                                }
                            }
                        } else if (!stage.modelName.empty()) {
                            stage.modelPath = configService_.getModelPath(stage.modelName);
                        }
                        if (modelEl.contains("conf_threshold")) stage.confThreshold = modelEl["conf_threshold"];
                        else if (streamConfig.contains("conf_thres")) stage.confThreshold = streamConfig["conf_thres"];
                        else if (config.contains("global_settings") && config["global_settings"].contains("default_conf_thres"))
                            stage.confThreshold = config["global_settings"]["default_conf_thres"];
                        else stage.confThreshold = configService_.getConfThreshold();
                        if (modelEl.contains("nms_threshold")) stage.nmsThreshold = modelEl["nms_threshold"];
                        else if (streamConfig.contains("nms_thres")) stage.nmsThreshold = streamConfig["nms_thres"];
                        else if (config.contains("global_settings") && config["global_settings"].contains("default_nms_thres"))
                            stage.nmsThreshold = config["global_settings"]["default_nms_thres"];
                        else stage.nmsThreshold = configService_.getNmsThreshold();
                        if (modelEl.contains("roi_from_previous") && modelEl["roi_from_previous"].get<bool>()) {
                            stage.roiFromPrevious = true;
                            anyRoiFromPrevious = true;
                        }
                        if (modelEl.contains("independent") && modelEl["independent"].get<bool>()) {
                            stage.independent = true;
                        }
                        if (modelEl.contains("params") && modelEl["params"].is_object()) {
                            stage.params = modelEl["params"];
                        }
                        if (!stage.modelPath.empty() && stage.modelPath != "none")
                            aiConfig.modelStages.push_back(stage);
                    }
                    {
                        bool hasBaseStage = false;
                        bool hasRoiStage = false;
                        for (const auto& s : aiConfig.modelStages) {
                            if (s.roiFromPrevious)
                                hasRoiStage = true;
                            else
                                hasBaseStage = true;
                        }
                        if (hasBaseStage && hasRoiStage && aiConfig.modelStages.size() > 1) {
                            std::stable_partition(aiConfig.modelStages.begin(),
                                                  aiConfig.modelStages.end(),
                                                  [](const ModelStageConfig& s) { return !s.roiFromPrevious; });
                        }
                    }
                    if (!aiConfig.modelStages.empty()) aiConfig.modelStages[0].roiFromPrevious = false;
                    if (anyRoiFromPrevious) aiConfig.aiPipelineMode = AIPipelineMode::Serial;
                    if (!aiConfig.modelStages.empty()) {
                        aiConfig.modelName = aiConfig.modelStages[0].modelName;
                        aiConfig.modelPath = aiConfig.modelStages[0].modelPath;
                        aiConfig.confThreshold = aiConfig.modelStages[0].confThreshold;
                        aiConfig.nmsThreshold = aiConfig.modelStages[0].nmsThreshold;
                        modelConfigured = true;
                        ALOGN("[Config] Using %zu model stages (mode=%s)", aiConfig.modelStages.size(),
                              aiConfig.aiPipelineMode == AIPipelineMode::Serial ? "serial" : "parallel");
                    }
                }
                // 如果沒有 models 數組，使用舊的 model_name 配置方式（向後兼容）
                if (!modelConfigured) {
                    if (streamConfig.contains("model_name")) {
                        aiConfig.modelName = streamConfig["model_name"];
                        aiConfig.modelPath = configService_.getModelPath(streamConfig["model_name"]);
                        modelConfigured = true;
                    } else if (config.contains("global_settings") && 
                              config["global_settings"].contains("default_model")) {
                        aiConfig.modelName = config["global_settings"]["default_model"];
                        aiConfig.modelPath = configService_.getModelPath(aiConfig.modelName);
                        modelConfigured = true;
                    }
                    
                    // 閾值配置（舊方式）
                    if (!modelConfigured || !streamConfig.contains("models")) {
                        if (streamConfig.contains("conf_thres")) {
                            aiConfig.confThreshold = streamConfig["conf_thres"];
                        } else if (config.contains("global_settings") && 
                                  config["global_settings"].contains("default_conf_thres")) {
                            aiConfig.confThreshold = config["global_settings"]["default_conf_thres"];
                        } else {
                            aiConfig.confThreshold = configService_.getConfThreshold();
                        }
                        
                        if (streamConfig.contains("nms_thres")) {
                            aiConfig.nmsThreshold = streamConfig["nms_thres"];
                        } else if (config.contains("global_settings") && 
                                  config["global_settings"].contains("default_nms_thres")) {
                            aiConfig.nmsThreshold = config["global_settings"]["default_nms_thres"];
                        } else {
                            aiConfig.nmsThreshold = configService_.getNmsThreshold();
                        }
                    }
                }
                
                // AI輸出尺寸
                if (streamConfig.contains("ai_output_width")) {
                    aiConfig.outputWidth = streamConfig["ai_output_width"];
                } else {
                    aiConfig.outputWidth = 640;
                }
                
                if (streamConfig.contains("ai_output_height")) {
                    aiConfig.outputHeight = streamConfig["ai_output_height"];
                } else {
                    aiConfig.outputHeight = 640;
                }
                
                if (streamConfig.contains("ai_fps")) {
                    aiConfig.fps = streamConfig["ai_fps"];
                } else {
                    // 預設降低 AI 支路 fps，避免拖累主碼流；可用 ai_fps 覆蓋
                    aiConfig.fps = std::min(rtspConfig.fps, 15);
                }
                
                addStream(aiConfig);
                ALOGN("[Config] Loaded AI stream: stream_id=%d, source=%s, model_name=%s, model_path=%s, conf_thres=%.2f, nms_thres=%.2f, VDEC_Group=%d, IVPS_Group=%d", 
                      aiConfig.streamId,
                      streamConfig["input_source"].get<std::string>().c_str(),
                      aiConfig.modelName.c_str(),
                      aiConfig.modelPath.c_str(),
                      aiConfig.confThreshold,
                      aiConfig.nmsThreshold,
                      aiConfig.vdecGroup,
                      aiConfig.ivpsGroup);
            }

            // 僅在啟用時記錄 raw 主碼流（端雲比對 / 雲端取未疊加畫面）；否則不佔用額外 RTP/編碼頻寬
            if (enableRawStream) {
                uint16_t rawPort = static_cast<uint16_t>(stream_port + 1);
                rawCandidates.push_back(RawCandidate{
                    inputSource,
                    rawPort,
                    inputSourceIndex,
                    rtspConfig.outputWidth,
                    rtspConfig.outputHeight,
                    rtspConfig.fps,
                    mainVdecGroup
                });
                ALOGN("[Config] enable_raw_stream=true for input index %d (expected live%d_raw), rawPort=%u",
                      inputSourceIndex, inputSourceIndex + 1, static_cast<unsigned>(rawPort));
            } else {
                ALOGN("[Config] enable_raw_stream=false for '%s' (skip live%d_raw; set enable_raw_stream to compare on cloud)",
                      inputSource.c_str(), inputSourceIndex + 1);
            }

            // 更新輸入源索引（在處理完一個輸入源的所有流之後）
            inputSourceIndex++;
        }

        // 建立所有 raw 主碼流（無 OSD）：
        // liveX_raw：推流端口是 osd 主碼流端口 + 1
        // liveX_osd：若你希望保留原本 liveX 名稱，mediamtx.yml 可用 redirect/liveX_osd->/liveX 做對應。
        for (const auto& cand : rawCandidates) {
            int dummyVdec = 0;
            int rawIvpsGroup = 0;
            getNextAvailableGroup(baseGroup, dummyVdec, rawIvpsGroup);
            baseGroup = rawIvpsGroup + 1;

            StreamConfig rawConfig;
            rawConfig.streamId = streamIdBase++;
            rawConfig.inputSource = cand.inputSource;
            rawConfig.inputCodec = "auto";
            rawConfig.ivpsGroup = rawIvpsGroup;
            rawConfig.vdecGroup = cand.sharedVdecGroup;
            rawConfig.isMediaMTXOutput = true;
            rawConfig.enableAI = false;  // Raw 主码流：不建 OSD region
            rawConfig.mediamtxEndpoint = mediamtxHost + ":" + std::to_string(cand.rawPort);
            rawConfig.outputWidth = cand.outputWidth;
            rawConfig.outputHeight = cand.outputHeight;
            rawConfig.fps = cand.fps;

            addStream(rawConfig);
            ALOGN("[Config] Loaded RAW main stream: stream_id=%d, source=%s, VDEC_Group=%d (shared with OSD), IVPS_Group=%d, mediamtx_endpoint=%s",
                  rawConfig.streamId,
                  cand.inputSource.c_str(),
                  rawConfig.vdecGroup,
                  rawConfig.ivpsGroup,
                  rawConfig.mediamtxEndpoint.c_str());
        }
        
        ALOGN("Successfully loaded %zu streams from config file", config["streams"].size());
        return true;
        
    } catch (const std::exception& e) {
        ALOGE("Failed to parse streams config: %s", e.what());
        return false;
    }
}

// 初始化 AI 流的 OSD 管理
void VideoStreamManager::initializeOSDForAIStream(int aiStreamId) {
    std::shared_lock<std::shared_mutex> streamLock(streamsMutex_); // 使用读锁
    
    auto it = aiStreamMap_.find(aiStreamId);
    if (it == aiStreamMap_.end() || it->second >= (int)streams_.size()) {
        ALOGW("[OSD] AI stream %d not found", aiStreamId);
        return;
    }
    
    VideoStream* aiStream = &streams_[it->second];
    if (!aiStream->isRunning() || !aiStream->getAIProcessor()) {
        ALOGW("[OSD] AI stream %d not ready (running=%d, hasProcessor=%d)", 
              aiStreamId, aiStream->isRunning() ? 1 : 0, aiStream->getAIProcessor() ? 1 : 0);
        return;
    }
    
    initializeOSDForStream(aiStreamId, aiStream);
}

// 參考 sample_multi_demux：初始化 OSD 管理（內部方法）
void VideoStreamManager::initializeOSDForStream(int aiStreamId, VideoStream* aiStream) {
    // 先收集需要 OSD 的 pipeline，避免持锁过久
    std::vector<pipeline_t*> pipes_need_osd;
    {
        std::shared_lock<std::shared_mutex> streamLock(streamsMutex_);
        std::string inputSource = aiStream->getInputSource();
        ALOGN("[OSD] Looking for main streams for AI stream %d (inputSource=%s)", 
              aiStreamId, inputSource.c_str());
        for (auto& stream : streams_) {
            ALOGN("[OSD] Checking stream %d: inputSource=%s, isMediaMTXOutput=%d, isRunning=%d, hasPipeline=%d", 
                  stream.getStreamId(), 
                  stream.getInputSource().c_str(),
                  stream.isMediaMTXOutput() ? 1 : 0,
                  stream.isRunning() ? 1 : 0,
                  stream.getPipeline() ? 1 : 0);
            if (stream.getStreamId() != aiStreamId &&
                stream.getInputSource() == inputSource &&
                stream.isMediaMTXOutput() &&
                stream.isRunning() &&
                stream.getPipeline()) {  // [修復] 確保 pipeline 指針不為空
                ALOGN("[OSD] Found main stream %d for AI stream %d (pipeid=%d)", 
                      stream.getStreamId(), aiStreamId, 
                      stream.getPipeline() ? stream.getPipeline()->pipeid : -1);
                pipes_need_osd.push_back(stream.getPipeline());
            }
        }
    }

    std::lock_guard<std::mutex> osdLock(osdMapMutex_);
    // 如果已經存在，先清理
    if (osdTargetMap_.find(aiStreamId) != osdTargetMap_.end()) {
        cleanupOSDForStream(aiStreamId);
    }
    
    if (pipes_need_osd.empty()) {
        ALOGW("[OSD] No pipelines need OSD for AI stream %d (inputSource=%s). "
              "This may happen if the main stream hasn't started yet or pipeline is not ready.", 
              aiStreamId, aiStream->getInputSource().c_str());
        return;
    }
    
    // 創建新的 OSD 管理結構
    OSDAssociatedModel* model = new OSDAssociatedModel();
    // 使用 weak_ptr 避免循環引用，同時允許檢查 aiProcessor 是否仍然有效
    std::shared_ptr<AIProcessor> aiProc = aiStream->getAIProcessor();
    model->aiProcessor = aiProc;
    model->pipes_need_osd = pipes_need_osd;
    
    // 獲取模型提供的 OSD 渲染器，如果沒有則使用默認渲染器
    // 多模型（安全帽+人形等）時合併結果會包含多種 label，需用「依 label 著色」的渲染器，否則
    // 僅依第一個模型選渲染器會導致第一模型為人形時全部畫成綠色（安全帽 no-helmet 也變綠）
    if (aiProc) {
        std::vector<std::string> modelPaths = aiStream->getModelPaths();
        bool anyHelmet = false;
        for (const std::string& p : modelPaths) {
            if (p.find("helmet") != std::string::npos) {
                anyHelmet = true;
                break;
            }
        }
        if (anyHelmet && modelPaths.size() > 1u) {
            model->osdRenderer = std::make_shared<DefaultOSDRenderer>();
            ALOGN("[OSD] Using helmet OSD renderer for AI stream %d (multi-model with helmet, label-based colors)", aiStreamId);
        } else {
            model->osdRenderer = aiProc->getOSDRenderer();
            if (!model->osdRenderer) {
                model->osdRenderer = std::make_shared<DefaultOSDRenderer>();
                ALOGN("[OSD] Using default OSD renderer for AI stream %d", aiStreamId);
            } else {
                ALOGN("[OSD] Using model-specific OSD renderer '%s' for AI stream %d", 
                      model->osdRenderer->getName().c_str(), aiStreamId);
            }
        }
    } else {
        model->osdRenderer = std::make_shared<DefaultOSDRenderer>();
    }
    
    // 啟動 OSD 更新線程
    model->osdThreadRunning.store(true, std::memory_order_release);
    model->osdUpdateThread = std::thread(osdUpdateThreadFunc, model);
    
    // 註冊到映射表
    osdTargetMap_[aiStreamId] = model;
    ALOGI("[OSD] Initialized OSD management for AI stream %d, %zu pipelines", 
          aiStreamId, model->pipes_need_osd.size());
}

// 清理 OSD 管理
void VideoStreamManager::cleanupOSDForStream(int aiStreamId) {
    OSDAssociatedModel* model = nullptr;
    
    // 1. 先從 map 中移除並獲取 model 指針（減少持鎖時間）
    {
        std::lock_guard<std::mutex> osdLock(osdMapMutex_);
        auto it = osdTargetMap_.find(aiStreamId);
        if (it != osdTargetMap_.end()) {
            model = it->second;
            osdTargetMap_.erase(it);
        }
    }
    
    // 2. 在鎖外安全停止線程並釋放資源
    if (model) {
        // 設置停止標誌並通知條件變量
        model->osdThreadRunning.store(false, std::memory_order_release);
        model->updateCv.notify_all();  // 喚醒 OSD 線程使其檢查停止標誌
        
        // 等待線程結束（線程內部有 50ms 超時機制，不會無限等待）
        if (model->osdUpdateThread.joinable()) {
            // 設置最長等待時間：500ms 應該足夠線程響應退出
            auto start = std::chrono::steady_clock::now();
            while (model->osdUpdateThread.joinable()) {
                // 嘗試 join（如果線程已結束會立即返回）
                // 由於 std::thread::join 沒有超時版本，我們依賴線程內部的超時機制
                model->osdUpdateThread.join();
                break;
            }
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (elapsed > std::chrono::milliseconds(100)) {
                ALOGW("[OSD] Thread join took %lld ms for stream %d", 
                      static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()),
                      aiStreamId);
            }
        }
        
        delete model;
        ALOGN("[OSD] Cleaned up OSD management for AI stream %d", aiStreamId);
    }
}

// 參考 sample_multi_demux：更新 AI 檢測結果
void VideoStreamManager::updateAIResult(int aiStreamId, const AI_RESULT_T* result) {
    if (!result) return;

    std::lock_guard<std::mutex> osdLock(osdMapMutex_);
    
    auto it = osdTargetMap_.find(aiStreamId);
    if (it != osdTargetMap_.end() && it->second) {
        OSDAssociatedModel* model = it->second;
        
        // 使用模型内部的锁保护结果数据的复制
        {
            std::lock_guard<std::mutex> resLock(model->resultMutex);
            // 深拷贝结果数据 (AI_RESULT_T 内部是固定大小数组，可以直接拷贝)
            memcpy(&model->latestResult, result, sizeof(AI_RESULT_T));
            model->shouldUpdateOSD.store(true, std::memory_order_relaxed);
        }
        // 通知 OSD 線程有新數據
        model->updateCv.notify_one();
        
        // 記錄檢測結果（每30幀記錄一次）
        static int log_count[64] = {0};
        if (++log_count[aiStreamId] % 300 == 0) {
            ALOGN("[OSD] Updated AI result for stream %d: nObjSize=%u, pipes_need_osd=%zu", 
                  aiStreamId, result->nObjSize, model->pipes_need_osd.size());
        }
    } else {
        // 記錄未找到 OSD 映射的情況（每100次記錄一次，避免日誌過多）
        static int warn_count[64] = {0};
        if (++warn_count[aiStreamId] % 100 == 0) {
            ALOGW("[OSD] No OSD mapping found for AI stream %d (updateAIResult called but OSD not initialized)", 
                  aiStreamId);
        }
    }
}

// 參考 sample_multi_demux：OSD 更新線程函數
void VideoStreamManager::osdUpdateThreadFunc(OSDAssociatedModel* model) {
    ALOGI("[OSD] OSD update thread started");
    
    // 设置线程名称方便调试
    pthread_setname_np(pthread_self(), "OSD_Update");
    
    while (model->osdThreadRunning.load(std::memory_order_relaxed)) {
        AI_RESULT_T result;
        bool hasUpdate = false;
        
        // 1. 使用條件變量等待更新通知或超時（替代輪詢 sleep）
        // [花屏修復] 將 OSD 更新節流到約 30fps（33ms），與主碼流幀率一致，避免 OSD 在 IVPS 合成
        // 一幀的過程中多次更新導致畫框處撕裂（原 16ms/60fps 會在一幀內更新 2 次）
        {
            std::unique_lock<std::mutex> lock(model->resultMutex);
            const int osd_wait_ms = 100;  // 約 10fps：顯著降低 OSD 對 IVPS/CPU 的壓力，優先保證主畫面流暢
            model->updateCv.wait_for(lock, std::chrono::milliseconds(osd_wait_ms), [model] {
                return model->shouldUpdateOSD.load(std::memory_order_relaxed) || 
                       !model->osdThreadRunning.load(std::memory_order_relaxed);
            });
            
            // 檢查是否需要退出
            if (!model->osdThreadRunning.load(std::memory_order_relaxed)) {
                ALOGI("[OSD] OSD update thread received stop signal");
                break;
            }
            
            if (model->shouldUpdateOSD.load(std::memory_order_relaxed)) {
                memcpy(&result, &model->latestResult, sizeof(AI_RESULT_T));
                model->shouldUpdateOSD.store(false, std::memory_order_relaxed);
                hasUpdate = true;
            }
        }
        
        if (!hasUpdate) {
            continue;
        }
        
        // 不再在此處 sleep：有更新時立即繪製可降低 OSD 延遲；下一輪會 wait_for 等待新結果，不會 busy-wait
        
        // 2. 检查是否有需要绘制的目标 Pipeline
        if (model->pipes_need_osd.empty()) {
            // 沒有目標 Pipeline，跳過此次更新（不需要額外 sleep，條件變量已處理等待）
            continue;
        }

        // 3. 获取AI模型输入尺寸（关键修复：恢复v0版本的逻辑）
        int ai_w = 640, ai_h = 640;
        std::shared_ptr<AIProcessor> aiProcessor = model->aiProcessor.lock();
        if (aiProcessor) {
            try {
                aiProcessor->getInputSize(&ai_w, &ai_h);
            } catch (...) {
                ai_w = 640;
                ai_h = 640;
            }
        }
        
        // 4. 准备绘制数据结构 (AX_IVPS_RGN_DISP_GROUP_T)
        // 我们假设所有关联的 Pipeline 分辨率相同（通常主码流都是 1920x1080）
        pipeline_t* firstPipe = model->pipes_need_osd[0];
        if (!firstPipe) continue;

        int dstW = firstPipe->m_ivps_attr.n_ivps_width; 
        int dstH = firstPipe->m_ivps_attr.n_ivps_height;
        
        // 防止除零错误
        if (dstW == 0 || dstH == 0) {
            dstW = 1920; 
            dstH = 1080;
        }

        // 5. 处理多区域OSD（恢复v0版本的多区域处理逻辑）
        for (pipeline_t* pipe : model->pipes_need_osd) {
            if (!pipe || !pipe->enable || pipe->m_ivps_attr.n_osd_rgn == 0) {
                static int skip_count = 0;
                if (++skip_count % 100 == 0) {
                    ALOGW("[OSD] Skipping pipeline update: pipe=%p, enable=%d, n_osd_rgn=%d", 
                          pipe, pipe ? pipe->enable : 0, pipe ? pipe->m_ivps_attr.n_osd_rgn : 0);
                }
                continue;
            }
            
            // 单区域处理
            if (pipe->m_ivps_attr.n_osd_rgn == 1) {
                IVPS_RGN_HANDLE regionHandle = pipe->m_ivps_attr.n_osd_rgn_chn[0];
                if (regionHandle == (IVPS_RGN_HANDLE)-1) continue;
                
                AX_IVPS_RGN_DISP_GROUP_T tDisp = {0};
                tDisp.tChnAttr.nZindex = 0;
                tDisp.tChnAttr.bSingleCanvas = AX_FALSE;
                tDisp.tChnAttr.nAlpha = 255;
                tDisp.tChnAttr.eFormat = AX_FORMAT_RGBA8888;
                tDisp.tChnAttr.nBitColor.bColorInvEn = AX_FALSE;
                tDisp.tChnAttr.nBitColor.nColor = 0xFF0000;
                tDisp.tChnAttr.nBitColor.nColorInv = 0xFF;
                tDisp.tChnAttr.nBitColor.nColorInvThr = 0xA0A0A0;
                
                // 调用渲染器，传递AI模型输入尺寸
                unsigned int numElements = model->osdRenderer->render(&result, dstW, dstH, &tDisp, AX_IVPS_REGION_MAX_DISP_NUM);
                tDisp.nNum = numElements;
                // 明確隱藏未使用的槽位，避免驅動讀到殘留資料造成花屏/色塊
                for (unsigned int k = numElements; k < AX_IVPS_REGION_MAX_DISP_NUM; k++) {
                    tDisp.arrDisp[k].bShow = AX_FALSE;
                    tDisp.arrDisp[k].eType = AX_IVPS_RGN_TYPE_RECT;
                }
                int ret = AX_IVPS_RGN_Update(regionHandle, &tDisp);
                if (ret != 0) {
                    static int error_count = 0;
                    if (++error_count % 100 == 0) {
                        ALOGE("[OSD] AX_IVPS_RGN_Update failed, ret=0x%x, handle=%d, pipeid=%d, numElements=%u",
                              ret, regionHandle, pipe->pipeid, numElements);
                    }
                } else {
                    // 記錄成功的 OSD 更新（每30次記錄一次，或當有檢測結果時）
                    static int success_log_count = 0;
                    if (++success_log_count % 300 == 0) {
                        ALOGN("[OSD] AX_IVPS_RGN_Update success: pipeid=%d, numElements=%u, nObjSize=%u",
                              pipe->pipeid, numElements, result.nObjSize);
                    }
                }
                // [花屏修復] 單區域更新後短暫讓出，降低與 IVPS 合成同一幀的競爭，減輕畫框撕裂
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } 
            // 多区域处理（恢复v0版本的多区域处理逻辑）
            else {
                unsigned int elementsPerRegion = AX_IVPS_REGION_MAX_DISP_NUM;
                unsigned int totalElements = result.nObjSize * 2;
                
                for (int rgnIdx = 0; rgnIdx < pipe->m_ivps_attr.n_osd_rgn; rgnIdx++) {
                    IVPS_RGN_HANDLE regionHandle = pipe->m_ivps_attr.n_osd_rgn_chn[rgnIdx];
                    if (regionHandle == (IVPS_RGN_HANDLE)-1) continue;
                    
                    AX_IVPS_RGN_DISP_GROUP_T tDisp = {0};
                    tDisp.tChnAttr.nZindex = rgnIdx;
                    tDisp.tChnAttr.bSingleCanvas = AX_FALSE;
                    tDisp.tChnAttr.nAlpha = 255;
                    tDisp.tChnAttr.eFormat = AX_FORMAT_RGBA8888;
                    tDisp.tChnAttr.nBitColor.bColorInvEn = AX_FALSE;
                    tDisp.tChnAttr.nBitColor.nColor = 0xFF0000;
                    tDisp.tChnAttr.nBitColor.nColorInv = 0xFF;
                    tDisp.tChnAttr.nBitColor.nColorInvThr = 0xA0A0A0;
                    
                    unsigned int startElement = rgnIdx * elementsPerRegion;
                    unsigned int endElement = std::min(startElement + elementsPerRegion, totalElements);
                    
                    if (startElement >= totalElements) {
                        tDisp.nNum = 0;
                        AX_IVPS_RGN_Update(regionHandle, &tDisp);
                        continue;
                    }
                    
                    unsigned int startObj = startElement / 2;
                    unsigned int endObj = std::min(startObj + (elementsPerRegion / 2), result.nObjSize);
                    
                    AI_RESULT_T subResult = {0};
                    subResult.nObjSize = endObj - startObj;
                    for (unsigned int i = startObj; i < endObj && i < result.nObjSize; i++) {
                        subResult.objects[i - startObj] = result.objects[i];
                    }
                    
                    // 调用渲染器，传递AI模型输入尺寸
                    unsigned int numElements = model->osdRenderer->render(&subResult, dstW, dstH, &tDisp, elementsPerRegion);
                    tDisp.nNum = numElements;
                    for (unsigned int k = numElements; k < AX_IVPS_REGION_MAX_DISP_NUM; k++) {
                        tDisp.arrDisp[k].bShow = AX_FALSE;
                        tDisp.arrDisp[k].eType = AX_IVPS_RGN_TYPE_RECT;
                    }
                    int ret = AX_IVPS_RGN_Update(regionHandle, &tDisp);
                    if (ret != 0) {
                        static int error_count = 0;
                        if (++error_count % 100 == 0) {
                            ALOGE("[OSD] AX_IVPS_RGN_Update failed for region %d, ret=0x%x, handle=%d, pipeid=%d, numElements=%u",
                                  rgnIdx, ret, regionHandle, pipe->pipeid, numElements);
                        }
                    } else {
                        // 記錄成功的 OSD 更新
                    static int success_log_count = 0;
                    if (++success_log_count % 300 == 0) {
                            ALOGN("[OSD] AX_IVPS_RGN_Update success for region %d: pipeid=%d, numElements=%u",
                                  rgnIdx, pipe->pipeid, numElements);
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        }
    }
    
    ALOGI("[OSD] OSD update thread stopped");
}


