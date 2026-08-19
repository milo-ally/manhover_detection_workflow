#include "video_stream_manager.h"
#include "../../utilities/json.hpp"

#include <fstream>
#include <algorithm>
#include <cstdlib>

namespace {

// 默认模型路径解析（与 config_service 保持一致；从 bin/ 启动时模型在 ../models/）
std::string resolve_model_path(const std::string& modelName) {
    if (modelName.empty() || modelName == "none") return "";
    if (modelName.find("/") != std::string::npos || modelName.find("\\\\") != std::string::npos)
        return modelName;
    if (modelName.find(".rknn") != std::string::npos)
        return "../models/" + modelName;
    return "../models/manhole-cover-yolo11s-production.rknn";
}

}  // namespace

VideoStreamManager::VideoStreamManager(ConfigService& configService)
    : configService_(configService) {}

VideoStreamManager::~VideoStreamManager() {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    for (auto& stream : streams_) {
        stream->stop();
    }
    streams_.clear();
    {
        std::lock_guard<std::mutex> osdLock(osdMapMutex_);
        for (auto& pair : osdTargetMap_) {
            delete pair.second;
        }
        osdTargetMap_.clear();
    }
}

void VideoStreamManager::addStream(const StreamConfig& config) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    streams_.push_back(std::make_unique<VideoStream>(config));
    aiStreamMap_[config.streamId] = static_cast<int>(streams_.size()) - 1;
}

void VideoStreamManager::removeStream(int streamId) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = aiStreamMap_.find(streamId);
    if (it == aiStreamMap_.end()) return;
    const int index = it->second;
    if (index >= 0 && index < static_cast<int>(streams_.size())) {
        streams_[index]->stop();
        streams_.erase(streams_.begin() + index);
    }
    aiStreamMap_.erase(it);
    cleanupOSDForStream(streamId);
}

void VideoStreamManager::initializeFromConfig(const ConfigService& configService) {
    // 兼容方法：当前配置由 loadStreamsFromConfig 直接加载
    (void)configService;
}

bool VideoStreamManager::loadStreamsFromConfig(const std::string& configPath,
                                               const std::string& mediamtxEndpoint,
                                               bool offlineMode,
                                               const std::string& outputPath) {
    runMode_ = offlineMode ? "offline" : "stream";

    std::ifstream file(configPath);
    if (!file.is_open()) {
        ALOGE("[VideoStreamManager] Cannot open config file: %s", configPath.c_str());
        return false;
    }
    nlohmann::json config;
    try {
        file >> config;
    } catch (const std::exception& e) {
        ALOGE("[VideoStreamManager] Invalid JSON in %s: %s", configPath.c_str(), e.what());
        return false;
    }

    // ===== MediaMTX 端點：命令行 > 配置文件 global_settings > 環境變量 > 默認 =====
    std::string mediamtxHost;
    std::string mediamtxPort;
    if (!mediamtxEndpoint.empty()) {
        size_t colon = mediamtxEndpoint.find(':');
        if (colon != std::string::npos) {
            mediamtxHost = mediamtxEndpoint.substr(0, colon);
            mediamtxPort = mediamtxEndpoint.substr(colon + 1);
        } else {
            mediamtxHost = mediamtxEndpoint;
        }
        ALOGN("[VideoStreamManager] Using MediaMTX endpoint from command line: %s:%s",
              mediamtxHost.c_str(), mediamtxPort.c_str());
    } else {
        if (config.contains("global_settings")) {
            auto& global = config["global_settings"];
            if (global.contains("mediamtx_host")) mediamtxHost = global["mediamtx_host"];
            if (global.contains("mediamtx_port")) mediamtxPort = global["mediamtx_port"];
        }
        const char* envHost = getenv("MEDIAMTX_HOST");
        const char* envPort = getenv("MEDIAMTX_RTP_PORT");
        if (envHost) mediamtxHost = envHost;
        if (envPort) mediamtxPort = envPort;
        if (mediamtxHost.empty()) mediamtxHost = "127.0.0.1";
        if (mediamtxPort.empty()) mediamtxPort = "8000";
    }

    // ===== 全局默认 =====
    float defaultConf = 0.25f;
    float defaultNms = 0.45f;
    std::string defaultModel = "manhole_cover";
    bool defaultEnableRawStream = false;
    if (config.contains("global_settings") && config["global_settings"].is_object()) {
        auto& global = config["global_settings"];
        if (global.contains("default_conf_thres")) defaultConf = global["default_conf_thres"];
        if (global.contains("default_nms_thres")) defaultNms = global["default_nms_thres"];
        if (global.contains("default_model")) defaultModel = global["default_model"];
        if (global.contains("enable_raw_stream")) defaultEnableRawStream = global["enable_raw_stream"].get<bool>();
    }

    if (!config.contains("streams") || !config["streams"].is_array()) {
        ALOGE("[VideoStreamManager] Invalid config format: missing 'streams' array");
        return false;
    }

    std::lock_guard<std::mutex> lock(streamsMutex_);
    streams_.clear();
    aiStreamMap_.clear();

    const size_t streamCount = config["streams"].size();
    int streamIdBase = 1;
    for (const auto& streamConfig : config["streams"]) {
        if (!streamConfig.contains("input_source") || !streamConfig["input_source"].is_string()) {
            ALOGW("[VideoStreamManager] Stream entry without input_source, skipped");
            continue;
        }

        StreamConfig sc;
        sc.streamId = streamConfig.contains("stream_id")
            ? streamConfig["stream_id"].get<int>() : streamIdBase++;
        sc.inputSource = streamConfig["input_source"].get<std::string>();
        sc.inputCodec = streamConfig.contains("input_codec")
            ? streamConfig["input_codec"].get<std::string>() : "h264";
        sc.enableAI = streamConfig.contains("enable_ai")
            ? streamConfig["enable_ai"].get<bool>() : true;
        sc.confThreshold = streamConfig.contains("conf_thres")
            ? streamConfig["conf_thres"].get<float>() : defaultConf;
        sc.nmsThreshold = streamConfig.contains("nms_thres")
            ? streamConfig["nms_thres"].get<float>() : defaultNms;
        sc.outputWidth = streamConfig.contains("output_width")
            ? streamConfig["output_width"].get<int>() : 1920;
        sc.outputHeight = streamConfig.contains("output_height")
            ? streamConfig["output_height"].get<int>() : 1080;
        sc.fps = streamConfig.contains("fps")
            ? streamConfig["fps"].get<int>() : 30;
        sc.aiOutputWidth = streamConfig.contains("ai_output_width")
            ? streamConfig["ai_output_width"].get<int>() : 640;
        sc.aiOutputHeight = streamConfig.contains("ai_output_height")
            ? streamConfig["ai_output_height"].get<int>() : 640;
        sc.aiFps = streamConfig.contains("ai_fps")
            ? streamConfig["ai_fps"].get<int>() : std::min(sc.fps, 15);
        sc.enableRawStream = streamConfig.contains("enable_raw_stream")
            ? streamConfig["enable_raw_stream"].get<bool>() : defaultEnableRawStream;
        if (streamConfig.contains("plugin") && streamConfig["plugin"].is_string()) {
            sc.pluginPath = streamConfig["plugin"].get<std::string>();
        }

        // ===== 模型：models[] 数组（多阶段） > model_name > global_settings.default_model =====
        if (streamConfig.contains("models") && streamConfig["models"].is_array() &&
            !streamConfig["models"].empty()) {
            for (const auto& modelEl : streamConfig["models"]) {
                ModelStageConfig stage;
                if (modelEl.contains("name")) stage.modelName = modelEl["name"].get<std::string>();
                if (modelEl.contains("plugin") && modelEl["plugin"].is_string())
                    stage.pluginPath = modelEl["plugin"].get<std::string>();
                else if (!sc.pluginPath.empty())
                    stage.pluginPath = sc.pluginPath;
                if (modelEl.contains("path") && !modelEl["path"].is_null()) {
                    stage.modelPath = modelEl["path"].get<std::string>();
                    if (stage.modelPath.find("/") == std::string::npos ||
                        stage.modelPath.find("../") == 0) {
                        if (!stage.modelName.empty()) {
                            std::string fp = resolve_model_path(stage.modelName);
                            if (!fp.empty()) stage.modelPath = fp;
                        }
                    }
                } else if (!stage.modelName.empty()) {
                    stage.modelPath = resolve_model_path(stage.modelName);
                }
                if (modelEl.contains("conf_threshold")) stage.confThreshold = modelEl["conf_threshold"];
                else stage.confThreshold = sc.confThreshold;
                if (modelEl.contains("nms_threshold")) stage.nmsThreshold = modelEl["nms_threshold"];
                else stage.nmsThreshold = sc.nmsThreshold;
                if (modelEl.contains("roi_from_previous") && modelEl["roi_from_previous"].get<bool>()) {
                    stage.roiFromPrevious = true;
                    sc.aiPipelineMode = AIPipelineMode::Serial;
                }
                if (modelEl.contains("independent") && modelEl["independent"].get<bool>()) {
                    stage.independent = true;
                }
                if (modelEl.contains("params") && modelEl["params"].is_object()) {
                    stage.params = modelEl["params"];
                }
                if (!stage.modelPath.empty() && stage.modelPath != "none") {
                    sc.modelStages.push_back(stage);
                }
            }
            if (!sc.modelStages.empty()) {
                sc.modelStages[0].roiFromPrevious = false;
                sc.modelName = sc.modelStages[0].modelName;
                sc.modelPath = sc.modelStages[0].modelPath;
                sc.confThreshold = sc.modelStages[0].confThreshold;
                sc.nmsThreshold = sc.modelStages[0].nmsThreshold;
            } else {
                sc.enableAI = false;
                ALOGW("[VideoStreamManager] Stream %d: no valid model stage, AI disabled", sc.streamId);
            }
        } else if (streamConfig.contains("model_name")) {
            sc.modelName = streamConfig["model_name"].get<std::string>();
            sc.modelPath = resolve_model_path(sc.modelName);
        } else if (!defaultModel.empty()) {
            sc.modelName = defaultModel;
            sc.modelPath = resolve_model_path(defaultModel);
        }

        if (sc.modelPath.empty()) {
            sc.enableAI = false;
            ALOGW("[VideoStreamManager] Stream %d: no model configured, AI disabled", sc.streamId);
        }

        // ===== 输出模式 =====
        if (offlineMode) {
            sc.isFileOutput = true;
            if (streamConfig.contains("output_path") && streamConfig["output_path"].is_string()) {
                sc.outputFilePath = streamConfig["output_path"].get<std::string>();
            } else if (!outputPath.empty()) {
                sc.outputFilePath = outputPath;
            }
            // 多路离线时避免共用同一输出文件
            if (streamCount > 1) {
                size_t pos = sc.outputFilePath.rfind('.');
                std::string suffix = "_" + std::to_string(sc.streamId);
                if (pos != std::string::npos) {
                    sc.outputFilePath = sc.outputFilePath.substr(0, pos) + suffix +
                                        sc.outputFilePath.substr(pos);
                } else {
                    sc.outputFilePath += suffix;
                }
            }
            ALOGN("[VideoStreamManager] Stream %d: offline output -> %s",
                  sc.streamId, sc.outputFilePath.c_str());
        } else {
            sc.isMediaMTXOutput = true;  // 兼容字段：输出最终进入 MediaMTX
            sc.rtspOutputUrl = "rtsp://127.0.0.1:8554/ai_out";
            if (streamCount > 1) {
                sc.rtspOutputUrl = "rtsp://127.0.0.1:8554/ai_out_" + std::to_string(sc.streamId);
            }
            ALOGN("[VideoStreamManager] Stream %d: stream output -> %s (MediaMTX %s:%s)",
                  sc.streamId, sc.rtspOutputUrl.c_str(), mediamtxHost.c_str(), mediamtxPort.c_str());
        }

        streams_.push_back(std::make_unique<VideoStream>(sc));
        aiStreamMap_[sc.streamId] = static_cast<int>(streams_.size()) - 1;
        ALOGN("[VideoStreamManager] Stream %d loaded: input=%s model=%s stages=%zu conf=%.3f nms=%.3f",
              sc.streamId, sc.inputSource.c_str(), sc.modelPath.c_str(),
              sc.modelStages.size(), sc.confThreshold, sc.nmsThreshold);
    }

    if (streams_.empty()) {
        ALOGE("[VideoStreamManager] No valid streams in config");
        return false;
    }
    ALOGN("[VideoStreamManager] Loaded %zu streams", streams_.size());
    return true;
}

void VideoStreamManager::handleConfigUpdate(const ConfigUpdate& update) {
    VideoStream* stream = nullptr;
    if (update.streamId < 0) {
        ALOGN("[VideoStreamManager] Applying global config update to %zu streams", streams_.size());
        for (auto& s : streams_) {
            if (update.valid && !update.modelPath.empty() && update.modelPath != "none") {
                s->setModelPath(update.modelPath);
                s->setModelName(update.modelName);
            }
            if (update.confThreshold > 0.0f) {
                s->setThresholds(update.confThreshold, update.nmsThreshold);
            }
        }
        return;
    }
    stream = getStream(update.streamId);
    if (!stream) return;
    ALOGN("[VideoStreamManager] Applying config update to stream %d", update.streamId);
    if (update.valid && !update.modelPath.empty() && update.modelPath != "none") {
        stream->setModelPath(update.modelPath);
        stream->setModelName(update.modelName);
    }
    if (update.confThreshold > 0.0f) {
        stream->setThresholds(update.confThreshold, update.nmsThreshold);
    }
}

void VideoStreamManager::enableAIStream(int streamId, bool enable) {
    VideoStream* stream = getStream(streamId);
    if (stream) stream->setAIEnabled(enable);
}

void VideoStreamManager::updateAIStream(int streamId, const std::string& modelPath,
                                        float confThreshold, float nmsThreshold,
                                        const std::string& modelName) {
    VideoStream* stream = getStream(streamId);
    if (!stream) return;
    stream->setModelPath(modelPath);
    if (!modelName.empty()) stream->setModelName(modelName);
    stream->setThresholds(confThreshold, nmsThreshold);
}

void VideoStreamManager::updateAIStreamWithStages(int streamId,
                                                  const std::vector<ModelStageConfig>& stages,
                                                  AIPipelineMode mode) {
    VideoStream* stream = getStream(streamId);
    if (stream) stream->setModelStages(stages, mode);
}

VideoStream* VideoStreamManager::getStream(int streamId) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = aiStreamMap_.find(streamId);
    if (it == aiStreamMap_.end()) return nullptr;
    const int index = it->second;
    if (index < 0 || index >= static_cast<int>(streams_.size())) return nullptr;
    return streams_[index].get();
}

void VideoStreamManager::addMultiStreamConfig(const std::vector<StreamConfig>& configs) {
    for (const auto& config : configs) {
        addStream(config);
    }
}

void VideoStreamManager::updateStreamModel(int streamId, const std::string& modelPath) {
    VideoStream* stream = getStream(streamId);
    if (stream) stream->setModelPath(modelPath);
}

void VideoStreamManager::updateAIResult(int aiStreamId, const AI_RESULT_T* result) {
    std::lock_guard<std::mutex> osdLock(osdMapMutex_);
    auto it = osdTargetMap_.find(aiStreamId);
    if (it == osdTargetMap_.end()) return;
    OSDAssociatedModel* model = it->second;
    std::lock_guard<std::mutex> resultLock(model->resultMutex);
    if (result) {
        model->latestResult = *result;
    }
}

void VideoStreamManager::initializeOSDForAIStream(int aiStreamId) {
    VideoStream* aiStream = getStream(aiStreamId);
    if (!aiStream) return;
    initializeOSDForStream(aiStreamId, aiStream);
}

void VideoStreamManager::initializeOSDForStream(int aiStreamId, VideoStream* aiStream) {
    std::lock_guard<std::mutex> osdLock(osdMapMutex_);
    auto existing = osdTargetMap_.find(aiStreamId);
    if (existing != osdTargetMap_.end()) {
        delete existing->second;
        osdTargetMap_.erase(existing);
    }
    OSDAssociatedModel* model = new OSDAssociatedModel();
    model->aiProcessor = aiStream->getAIProcessor();
    model->osdRenderer = std::make_shared<DefaultOSDRenderer>();
    osdTargetMap_[aiStreamId] = model;
    ALOGN("[OSD] Initialized OSD management for AI stream %d", aiStreamId);
}

void VideoStreamManager::cleanupOSDForStream(int aiStreamId) {
    std::lock_guard<std::mutex> osdLock(osdMapMutex_);
    auto it = osdTargetMap_.find(aiStreamId);
    if (it != osdTargetMap_.end()) {
        delete it->second;
        osdTargetMap_.erase(it);
    }
}

void VideoStreamManager::notifyAIError(int streamId, const std::string& error) {
    ALOGE("AI error on stream %d: %s", streamId, error.c_str());
}

bool VideoStreamManager::allEnded() const {
    if (streams_.empty()) return true;
    for (const auto& stream : streams_) {
        if (!stream->isEnded()) return false;
    }
    return true;
}
