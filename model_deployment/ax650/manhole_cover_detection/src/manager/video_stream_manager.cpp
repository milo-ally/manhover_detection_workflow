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

// 鏋勯€犲嚱鏁帮紝鍒濆鍖栭厤缃湇鍔″紩鐢?
VideoStreamManager::VideoStreamManager(ConfigService& configService)
    : configService_(configService) {
    // 瑷疆鍏ㄥ眬 VideoStreamManager 鎸囬嚌锛屼緵 AI 鍥炶浣跨敤
    VideoStream::setGlobalStreamManager(this);
}

// 鏋愭瀯鍑芥暟
VideoStreamManager::~VideoStreamManager() {
    // 鍏堝仠姝㈡墍鏈?OSD 绶氱▼
    {
        std::lock_guard<std::mutex> osdLock(osdMapMutex_);
        for (auto& pair : osdTargetMap_) {
            if (pair.second) {
                // 浣跨敤 atomic store 涓﹂€氱煡姊濅欢璁婇噺
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
    
    // 娓呴櫎鍏ㄥ眬鎸囬嚌
    VideoStream::setGlobalStreamManager(nullptr);
}

// 娣诲姞瑙嗛娴侊紝濡傛灉鍚敤AI锛屽垯璁板綍AI娴両D
void VideoStreamManager::addStream(const StreamConfig& config) {
    std::unique_lock<std::shared_mutex> lock(streamsMutex_);
    
    // 妫€鏌ユ槸鍚﹀凡瀛樺湪鐩稿悓ID鐨勬祦
    for (const auto& stream : streams_) {
        if (stream.getStreamId() == config.streamId) {
            ALOGW("Stream with ID %d already exists", config.streamId);
            return;
        }
    }
    
    streams_.emplace_back(config);
    
    // 鍗€鍒嗕富纰兼祦鍜?AI 娴侊細
    // 1. 涓荤⒓娴侊細enableAI=true 浣?isMediaMTXOutput=true锛屽彧鏄偤浜?OSD锛屼笉閫茶鎺ㄧ悊
    // 2. AI 娴侊細enableAI=true 浣?isMediaMTXOutput=false锛岄€茶 AI 鎺ㄧ悊
    if (config.enableAI && !config.isMediaMTXOutput) {
        // 閫欐槸 AI 娴侊紝瑷诲唺鍒?aiStreamMap_锛堝嵆浣垮垵濮嬫檪娌掓湁 modelPath锛?
        // 鍥犵偤妯″瀷鍙兘閫氶亷 Web 閰嶇疆鍕曟厠瑷疆
        aiStreamMap_[config.streamId] = streams_.size() - 1;
        ALOGI("AI stream registered with ID: %d (modelPath: %s)", 
              config.streamId, 
              config.modelPath.empty() ? "(empty, will be set via config)" : config.modelPath.c_str());
        
        VideoStream& newStream = streams_.back();
        // 澶氭ā鍨嬮殠娈碉紙涓﹁鎴栦覆琛岋級
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
            // 鍠ā鍨嬪悜寰屽吋瀹?
            std::string fullModelPath = config.modelPath;
            if (fullModelPath.find("/") == std::string::npos) {
                fullModelPath = configService_.getModelPath(config.modelName.empty() ? fullModelPath : config.modelName);
            }
            if (!fullModelPath.empty()) {
                ALOGN("[VideoStreamManager] Creating AIProcessor for stream %d...", config.streamId);
                nlohmann::json effParams = nlohmann::json::object();
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
        // 涓荤⒓娴侊細enableAI=true 浣嗘矑鏈?modelPath锛屽彧鏄偤浜?OSD
        ALOGI("Main stream %d registered with OSD support (no AI inference)", config.streamId);
    }
}

// 绉婚櫎鎸囧畾ID鐨勮棰戞祦
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

// 澶勭悊閰嶇疆鏇存柊锛屾牴鎹厤缃喅瀹氭槸鍚﹀惎鐢ˋI娴佹垨鏇存柊AI娴佸弬鏁?
void VideoStreamManager::handleConfigUpdate(const ConfigUpdate& update) {
    ALOGN("[VideoStreamManager] handleConfigUpdate called: model=%s, valid=%d, streamId=%d, cameraId=%d", 
          update.modelName.c_str(), update.valid ? 1 : 0, update.streamId, update.cameraId);
    
    // 鏀寔鎸夋祦鏇存柊锛氬鏋滄寚瀹氫簡 streamId锛屽彧鏇存柊瑭叉祦锛涘惁鍓囨洿鏂版墍鏈夋祦锛堝悜寰屽吋瀹癸級
    int targetStreamId = update.streamId;
    
    // 濡傛灉鎸囧畾浜?cameraId锛岄渶瑕佹槧灏勫埌灏嶆噳鐨?AI 娴?streamId
    // 鏀濆儚闋?ID 灏嶆噳涓荤⒓娴佺殑 streamId锛孉I 娴佺殑 streamId = 涓荤⒓娴?streamId + 1
    if (update.cameraId > 0 && targetStreamId < 0) {
        // cameraId 1 -> 涓荤⒓娴?streamId=1, AI 娴?streamId=2
        // cameraId 2 -> 涓荤⒓娴?streamId=3, AI 娴?streamId=4
        // 瑕忓緥锛欰I 娴?streamId = (cameraId - 1) * 2 + 2 = cameraId * 2
        targetStreamId = update.cameraId * 2;
        ALOGN("[VideoStreamManager] Mapped cameraId=%d to AI streamId=%d", update.cameraId, targetStreamId);
    }
    
    std::vector<int> streamIds;
    if (targetStreamId > 0) {
        // 鎸夋祦鏇存柊锛氬彧鏇存柊鎸囧畾鐨勬祦
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
        // 鍏ㄥ眬鏇存柊锛?dev/shm/ai_config.json 鐨勮垔鏍煎紡锛夛細
        // 涓嶈瑕嗚搵鍛戒护鍒?-m 鎸囧畾鐨勬ā鍨嬶紝閬垮厤鍟熷嫊寰屾ā鍨嬬媭鎱嬪け鎺с€?
        // UI 鑻ヨ鎺у埗锛屽繀闋堣蛋 cameraId/streamId 鐨勬寜娴佹洿鏂般€?
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
    
    // 鍦ㄥ惊鐜腑澶勭悊姣忎釜娴侊紝纭繚涓嶅湪鑰楁椂鎿嶄綔鏃舵寔鏈塎anager鐨勯攣
    if (!update.valid || update.modelName == "none") {
        // 绂佺敤鎸囧畾鐨?AI 娴?
        for (int streamId : streamIds) {
            enableAIStream(streamId, false);
        }
    } else {
        // 鏇存柊鎸囧畾鐨?AI 娴?
        for (int streamId : streamIds) {
            // 鎸夋祦鏇存柊锛坈ameraId/streamId锛夎鐐恒€孶I 鎺у埗銆嶏紝鍏佽ū瑕嗚搵鍛戒护鍒楁ā鍨?
            // 蹇呴爤鍦ㄨ鐢?updateAIStream 涔嬪墠娓呴櫎鍛戒护鍒楁瑷橈紝纰轰繚 updateAIStream 璁€鍙栨檪鐪嬪埌鐨勬槸娓呴櫎寰岀殑鐙€鎱?
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
                // 纰轰繚娓呴櫎鎿嶄綔瀹屾垚寰屽啀瑾跨敤 updateAIStream
                // 浣跨敤涓€鍊嬪皬鐨勫欢閬叉垨纰轰繚閹栭噵鏀惧緦鍐嶇辜绾?
            }
            // updateAIStream 鍏ч儴涓嶆渻鐛插彇 streamsMutex_锛屼絾 initializeOSDForStream 鏈?
            // 鎵€浠ラ€欒！涓嶉渶瑕佹寔鏈夐帠锛岄伩鍏嶆閹?
            if (!update.modelStages.empty()) {
                // 澶氭ā鍨嬮厤缃紙涓﹁/涓茶锛?
                ALOGN("[VideoStreamManager] Calling updateAIStreamWithStages for stream %d: %zu stages (mode=%s)", 
                      streamId, update.modelStages.size(),
                      update.aiPipelineMode == AIPipelineMode::Serial ? "serial" : "parallel");
                updateAIStreamWithStages(streamId, update.modelStages, update.aiPipelineMode);
            } else {
                // 鍠ā鍨嬮厤缃紙鍚戝緦鍏煎锛?
                ALOGN("[VideoStreamManager] Calling updateAIStream for stream %d: model=%s, path=%s", 
                      streamId, update.modelName.c_str(), update.modelPath.c_str());
                updateAIStream(streamId, update.modelPath, update.confThreshold, update.nmsThreshold, update.modelName);
            }
        }
    }
    ALOGN("[VideoStreamManager] handleConfigUpdate completed");
}

// 鍚敤鎴栫鐢ㄦ寚瀹欰I娴?
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

// 鏇存柊AI娴佺殑妯″瀷鍜岄槇鍊煎弬鏁?
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

    // 濡傛灉鍛戒护鍒楁瑷樺瓨鍦ㄤ笖妯″瀷璺緫鐩稿悓锛屽彧鏇存柊闁惧€硷紙閬垮厤閲嶈鍔犺級锛?
    // 閫欏€嬫鏌ユ噳瑭插湪 handleConfigUpdate 娓呴櫎妯欒涔嬪緦鍩疯锛屾墍浠ラ€氬父 isCommandLine 鎳夎┎鏄?false
    if (isCommandLine && !modelPath.empty() && modelPath == currentModelPath) {
        ALOGN("[VideoStreamManager] updateAIStream: Command-line model unchanged (path=%s), only updating thresholds", modelPath.c_str());
        targetStream->setAIEnabled(true);   
        targetStream->setThresholds(confThreshold, nmsThreshold);
        return;
    }
    
    // 濡傛灉鍛戒护鍒楁瑷樺瓨鍦ㄤ絾妯″瀷璺緫涓嶅悓锛岃鏄庨渶瑕佸垏鎻涙ā鍨嬶紝鎳夎┎娓呴櫎妯欒涓︽洿鏂?
    if (isCommandLine && !modelPath.empty() && modelPath != currentModelPath) {
        ALOGN("[VideoStreamManager] updateAIStream: Command-line model path changed (%s -> %s), clearing flag and updating", 
              currentModelPath.c_str(), modelPath.c_str());
        // 娓呴櫎鍛戒护鍒楁瑷橈紝鍏佽ū鏇存柊
        {
            std::shared_lock<std::shared_mutex> lock(streamsMutex_);
            auto it = aiStreamMap_.find(streamId);
            if (it != aiStreamMap_.end() && it->second < (int)streams_.size()) {
                streams_[it->second].clearCommandLineModelFlag();
            }
        }
        // 绻肩簩鍩疯寰岀簩鐨勬ā鍨嬫洿鏂伴倧杓?
    }
    
    std::string normalizedModelPath = modelPath;
    std::string normalizedCurrentPath = currentModelPath;
    
    // "../models/" 鐨勯暦搴︽槸 10锛屾墍浠ユ噳瑭蹭娇鐢?substr(10)
    if (normalizedModelPath.find("../models/") == 0) normalizedModelPath = normalizedModelPath.substr(10);
    if (normalizedCurrentPath.find("../models/") == 0) normalizedCurrentPath = normalizedCurrentPath.substr(10);

    if (modelPath.empty()) {
        enableAIStream(streamId, false);
        return;
    }

    std::unique_ptr<AIProcessor> newProcessor = nullptr;
    bool needModelReload = (normalizedModelPath != normalizedCurrentPath);
    
    ALOGN("[VideoStreamManager] updateAIStream: needModelReload=%d, normalizedCurrent=%s, normalizedNew=%s", 
          needModelReload ? 1 : 0, normalizedCurrentPat…8857 tokens truncated…aded %zu streams from config file", config["streams"].size());
        return true;
        
    } catch (const std::exception& e) {
        ALOGE("Failed to parse streams config: %s", e.what());
        return false;
    }
}

// 鍒濆鍖?AI 娴佺殑 OSD 绠＄悊
void VideoStreamManager::initializeOSDForAIStream(int aiStreamId) {
    std::shared_lock<std::shared_mutex> streamLock(streamsMutex_); // 浣跨敤璇婚攣
    
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

// 鍙冭€?sample_multi_demux锛氬垵濮嬪寲 OSD 绠＄悊锛堝収閮ㄦ柟娉曪級
void VideoStreamManager::initializeOSDForStream(int aiStreamId, VideoStream* aiStream) {
    // 鍏堟敹闆嗛渶瑕?OSD 鐨?pipeline锛岄伩鍏嶆寔閿佽繃涔?
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
                stream.getPipeline()) {  // [淇京] 纰轰繚 pipeline 鎸囬嚌涓嶇偤绌?
                ALOGN("[OSD] Found main stream %d for AI stream %d (pipeid=%d)", 
                      stream.getStreamId(), aiStreamId, 
                      stream.getPipeline() ? stream.getPipeline()->pipeid : -1);
                pipes_need_osd.push_back(stream.getPipeline());
            }
        }
    }

    std::lock_guard<std::mutex> osdLock(osdMapMutex_);
    // 濡傛灉宸茬稉瀛樺湪锛屽厛娓呯悊
    if (osdTargetMap_.find(aiStreamId) != osdTargetMap_.end()) {
        cleanupOSDForStream(aiStreamId);
    }
    
    if (pipes_need_osd.empty()) {
        ALOGW("[OSD] No pipelines need OSD for AI stream %d (inputSource=%s). "
              "This may happen if the main stream hasn't started yet or pipeline is not ready.", 
              aiStreamId, aiStream->getInputSource().c_str());
        return;
    }
    
    // 鍓靛缓鏂扮殑 OSD 绠＄悊绲愭
    OSDAssociatedModel* model = new OSDAssociatedModel();
    // 浣跨敤 weak_ptr 閬垮厤寰挵寮曠敤锛屽悓鏅傚厑瑷辨鏌?aiProcessor 鏄惁浠嶇劧鏈夋晥
    std::shared_ptr<AIProcessor> aiProc = aiStream->getAIProcessor();
    model->aiProcessor = aiProc;
    model->pipes_need_osd = pipes_need_osd;
    
    // 鐛插彇妯″瀷鎻愪緵鐨?OSD 娓叉煋鍣紝濡傛灉娌掓湁鍓囦娇鐢ㄩ粯瑾嶆覆鏌撳櫒
    // 澶氭ā鍨嬶紙瀹夊叏甯?浜哄舰绛夛級鏅傚悎浣电祼鏋滄渻鍖呭惈澶氱ó label锛岄渶鐢ㄣ€屼緷 label 钁楄壊銆嶇殑娓叉煋鍣紝鍚﹀墖
    // 鍍呬緷绗竴鍊嬫ā鍨嬮伕娓叉煋鍣ㄦ渻灏庤嚧绗竴妯″瀷鐐轰汉褰㈡檪鍏ㄩ儴鐣垚缍犺壊锛堝畨鍏ㄥ附 no-helmet 涔熻畩缍狅級
    if (aiProc) {
        model->osdRenderer = aiProc->getOSDRenderer();
        if (!model->osdRenderer) {
            model->osdRenderer = std::make_shared<DefaultOSDRenderer>();
        }
    } else {
        model->osdRenderer = std::make_shared<DefaultOSDRenderer>();
    }
    
    // 鍟熷嫊 OSD 鏇存柊绶氱▼
    model->osdThreadRunning.store(true, std::memory_order_release);
    model->osdUpdateThread = std::thread(osdUpdateThreadFunc, model);
    
    // 瑷诲唺鍒版槧灏勮〃
    osdTargetMap_[aiStreamId] = model;
    ALOGI("[OSD] Initialized OSD management for AI stream %d, %zu pipelines", 
          aiStreamId, model->pipes_need_osd.size());
}

// 娓呯悊 OSD 绠＄悊
void VideoStreamManager::cleanupOSDForStream(int aiStreamId) {
    OSDAssociatedModel* model = nullptr;
    
    // 1. 鍏堝緸 map 涓Щ闄や甫鐛插彇 model 鎸囬嚌锛堟笡灏戞寔閹栨檪闁擄級
    {
        std::lock_guard<std::mutex> osdLock(osdMapMutex_);
        auto it = osdTargetMap_.find(aiStreamId);
        if (it != osdTargetMap_.end()) {
            model = it->second;
            osdTargetMap_.erase(it);
        }
    }
    
    // 2. 鍦ㄩ帠澶栧畨鍏ㄥ仠姝㈢窔绋嬩甫閲嬫斁璩囨簮
    if (model) {
        // 瑷疆鍋滄妯欒獙涓﹂€氱煡姊濅欢璁婇噺
        model->osdThreadRunning.store(false, std::memory_order_release);
        model->updateCv.notify_all();  // 鍠氶啋 OSD 绶氱▼浣垮叾妾㈡煡鍋滄妯欒獙
        
        // 绛夊緟绶氱▼绲愭潫锛堢窔绋嬪収閮ㄦ湁 50ms 瓒呮檪姗熷埗锛屼笉鏈冪劇闄愮瓑寰咃級
        if (model->osdUpdateThread.joinable()) {
            // 瑷疆鏈€闀风瓑寰呮檪闁擄細500ms 鎳夎┎瓒冲绶氱▼闊挎噳閫€鍑?
            auto start = std::chrono::steady_clock::now();
            while (model->osdUpdateThread.joinable()) {
                // 鍢楄│ join锛堝鏋滅窔绋嬪凡绲愭潫鏈冪珛鍗宠繑鍥烇級
                // 鐢辨柤 std::thread::join 娌掓湁瓒呮檪鐗堟湰锛屾垜鍊戜緷璩寸窔绋嬪収閮ㄧ殑瓒呮檪姗熷埗
                model->osdUpdateThread.join();
                break;
            }
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (elapsed > std::chrono::milliseconds(100)) {
                ALOGW("[OSD] Thread join took %lld ms for stream %d", 
                      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(),
                      aiStreamId);
            }
        }
        
        delete model;
        ALOGN("[OSD] Cleaned up OSD management for AI stream %d", aiStreamId);
    }
}

// 鍙冭€?sample_multi_demux锛氭洿鏂?AI 妾㈡脯绲愭灉
void VideoStreamManager::updateAIResult(int aiStreamId, const AI_RESULT_T* result) {
    if (!result) return;

    std::lock_guard<std::mutex> osdLock(osdMapMutex_);
    
    auto it = osdTargetMap_.find(aiStreamId);
    if (it != osdTargetMap_.end() && it->second) {
        OSDAssociatedModel* model = it->second;
        
        // 浣跨敤妯″瀷鍐呴儴鐨勯攣淇濇姢缁撴灉鏁版嵁鐨勫鍒?
        {
            std::lock_guard<std::mutex> resLock(model->resultMutex);
            // 娣辨嫹璐濈粨鏋滄暟鎹?(AI_RESULT_T 鍐呴儴鏄浐瀹氬ぇ灏忔暟缁勶紝鍙互鐩存帴鎷疯礉)
            memcpy(&model->latestResult, result, sizeof(AI_RESULT_T));
            model->shouldUpdateOSD.store(true, std::memory_order_relaxed);
        }
        // 閫氱煡 OSD 绶氱▼鏈夋柊鏁告摎
        model->updateCv.notify_one();
        
        // 瑷橀寗妾㈡脯绲愭灉锛堟瘡30骞€瑷橀寗涓€娆★級
        static int log_count[64] = {0};
        if (++log_count[aiStreamId] % 300 == 0) {
            ALOGN("[OSD] Updated AI result for stream %d: nObjSize=%u, pipes_need_osd=%zu", 
                  aiStreamId, result->nObjSize, model->pipes_need_osd.size());
        }
    } else {
        // 瑷橀寗鏈壘鍒?OSD 鏄犲皠鐨勬儏娉侊紙姣?00娆¤閷勪竴娆★紝閬垮厤鏃ヨ獙閬庡锛?
        static int warn_count[64] = {0};
        if (++warn_count[aiStreamId] % 100 == 0) {
            ALOGW("[OSD] No OSD mapping found for AI stream %d (updateAIResult called but OSD not initialized)", 
                  aiStreamId);
        }
    }
}

// 鍙冭€?sample_multi_demux锛歄SD 鏇存柊绶氱▼鍑芥暩
void VideoStreamManager::osdUpdateThreadFunc(OSDAssociatedModel* model) {
    ALOGI("[OSD] OSD update thread started");
    
    // 璁剧疆绾跨▼鍚嶇О鏂逛究璋冭瘯
    pthread_setname_np(pthread_self(), "OSD_Update");
    
    while (model->osdThreadRunning.load(std::memory_order_relaxed)) {
        AI_RESULT_T result;
        bool hasUpdate = false;
        
        // 1. 浣跨敤姊濅欢璁婇噺绛夊緟鏇存柊閫氱煡鎴栬秴鏅傦紙鏇夸唬杓 sleep锛?
        // [鑺卞睆淇京] 灏?OSD 鏇存柊绡€娴佸埌绱?30fps锛?3ms锛夛紝鑸囦富纰兼祦骞€鐜囦竴鑷达紝閬垮厤 OSD 鍦?IVPS 鍚堟垚
        // 涓€骞€鐨勯亷绋嬩腑澶氭鏇存柊灏庤嚧鐣铏曟挄瑁傦紙鍘?16ms/60fps 鏈冨湪涓€骞€鍏ф洿鏂?2 娆★級
        {
            std::unique_lock<std::mutex> lock(model->resultMutex);
            const int osd_wait_ms = 100;  // 绱?10fps锛氶’钁楅檷浣?OSD 灏?IVPS/CPU 鐨勫鍔涳紝鍎厛淇濊瓑涓荤暙闈㈡祦鏆?
            model->updateCv.wait_for(lock, std::chrono::milliseconds(osd_wait_ms), [model] {
                return model->shouldUpdateOSD.load(std::memory_order_relaxed) || 
                       !model->osdThreadRunning.load(std::memory_order_relaxed);
            });
            
            // 妾㈡煡鏄惁闇€瑕侀€€鍑?
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
        
        // 涓嶅啀鍦ㄦ铏?sleep锛氭湁鏇存柊鏅傜珛鍗崇躬瑁藉彲闄嶄綆 OSD 寤堕伈锛涗笅涓€杓渻 wait_for 绛夊緟鏂扮祼鏋滐紝涓嶆渻 busy-wait
        
        // 2. 妫€鏌ユ槸鍚︽湁闇€瑕佺粯鍒剁殑鐩爣 Pipeline
        if (model->pipes_need_osd.empty()) {
            // 娌掓湁鐩 Pipeline锛岃烦閬庢娆℃洿鏂帮紙涓嶉渶瑕侀澶?sleep锛屾浠惰畩閲忓凡铏曠悊绛夊緟锛?
            continue;
        }

        // 3. 鑾峰彇AI妯″瀷杈撳叆灏哄锛堝叧閿慨澶嶏細鎭㈠v0鐗堟湰鐨勯€昏緫锛?
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
        
        // 4. 鍑嗗缁樺埗鏁版嵁缁撴瀯 (AX_IVPS_RGN_DISP_GROUP_T)
        // 鎴戜滑鍋囪鎵€鏈夊叧鑱旂殑 Pipeline 鍒嗚鲸鐜囩浉鍚岋紙閫氬父涓荤爜娴侀兘鏄?1920x1080锛?
        pipeline_t* firstPipe = model->pipes_need_osd[0];
        if (!firstPipe) continue;

        int dstW = firstPipe->m_ivps_attr.n_ivps_width; 
        int dstH = firstPipe->m_ivps_attr.n_ivps_height;
        
        // 闃叉闄ら浂閿欒
        if (dstW == 0 || dstH == 0) {
            dstW = 1920; 
            dstH = 1080;
        }

        // 5. 澶勭悊澶氬尯鍩烵SD锛堟仮澶峷0鐗堟湰鐨勫鍖哄煙澶勭悊閫昏緫锛?
        for (pipeline_t* pipe : model->pipes_need_osd) {
            if (!pipe || !pipe->enable || pipe->m_ivps_attr.n_osd_rgn == 0) {
                static int skip_count = 0;
                if (++skip_count % 100 == 0) {
                    ALOGW("[OSD] Skipping pipeline update: pipe=%p, enable=%d, n_osd_rgn=%d", 
                          pipe, pipe ? pipe->enable : 0, pipe ? pipe->m_ivps_attr.n_osd_rgn : 0);
                }
                continue;
            }
            
            // 鍗曞尯鍩熷鐞?
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
                
                // 璋冪敤娓叉煋鍣紝浼犻€扐I妯″瀷杈撳叆灏哄
                unsigned int numElements = model->osdRenderer->render(&result, dstW, dstH, &tDisp, AX_IVPS_REGION_MAX_DISP_NUM);
                tDisp.nNum = numElements;
                // 鏄庣⒑闅辫棌鏈娇鐢ㄧ殑妲戒綅锛岄伩鍏嶉﹨鍕曡畝鍒版畼鐣欒硣鏂欓€犳垚鑺卞睆/鑹插
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
                    // 瑷橀寗鎴愬姛鐨?OSD 鏇存柊锛堟瘡30娆¤閷勪竴娆★紝鎴栫暥鏈夋娓祼鏋滄檪锛?
                    static int success_log_count = 0;
                    if (++success_log_count % 300 == 0) {
                        ALOGN("[OSD] AX_IVPS_RGN_Update success: pipeid=%d, numElements=%u, nObjSize=%u",
                              pipe->pipeid, numElements, result.nObjSize);
                    }
                }
                // [鑺卞睆淇京] 鍠崁鍩熸洿鏂板緦鐭毇璁撳嚭锛岄檷浣庤垏 IVPS 鍚堟垚鍚屼竴骞€鐨勭鐖紝娓涜紩鐣鎾曡
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } 
            // 澶氬尯鍩熷鐞嗭紙鎭㈠v0鐗堟湰鐨勫鍖哄煙澶勭悊閫昏緫锛?
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
                    
                    // 璋冪敤娓叉煋鍣紝浼犻€扐I妯″瀷杈撳叆灏哄
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
                        // 瑷橀寗鎴愬姛鐨?OSD 鏇存柊
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




