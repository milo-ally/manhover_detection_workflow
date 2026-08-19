#include "video_stream_manager.h"
#include "../../utilities/json.hpp"

#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <memory>

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
    (void)configService;
}

bool VideoStreamManager::loadStreamsFromConfig(const std::string& configPath,
                                               const std::string& mediamtxEndpoint,
                                               bool offlineMode,
                                               const std::string& outputPath) {
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
    if (config.contains("global_settings") && config["global_settings"].is_object()) {
        auto& global = config["global_settings"];
        if (global.contains("default_conf_thres")) defaultConf = global["default_conf_thres"];
        if (global.contains("default_nms_thres")) defaultNms = global["default_nms_thres"];
        if (global.contains("default_model")) defaultModel = global["default_model"];
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

        // ===== 解析公共字段 =====
        const std::string inputSource = streamConfig["input_source"].get<std::string>();
        const std::string inputCodec = streamConfig.contains("input_codec")
            ? streamConfig["input_codec"].get<std::string>() : "h264";
        const float confThres = streamConfig.contains("conf_thres")
            ? streamConfig["conf_thres"].get<float>() : defaultConf;
        const float nmsThres = streamConfig.contains("nms_thres")
            ? streamConfig["nms_thres"].get<float>() : defaultNms;
        const int outWidth = streamConfig.contains("output_width")
            ? streamConfig["output_width"].get<int>() : 1920;
        const int outHeight = streamConfig.contains("output_height")
            ? streamConfig["output_height"].get<int>() : 1080;
        const int fps = streamConfig.contains("fps")
            ? streamConfig["fps"].get<int>() : 30;
        const int aiW = streamConfig.contains("ai_output_width")
            ? streamConfig["ai_output_width"].get<int>() : 640;
        const int aiH = streamConfig.contains("ai_output_height")
            ? streamConfig["ai_output_height"].get<int>() : 640;
        const int aiFps = streamConfig.contains("ai_fps")
            ? streamConfig["ai_fps"].get<int>() : std::min(fps, 15);
        const int bitrateKbps = streamConfig.contains("bitrate_kbps")
            ? streamConfig["bitrate_kbps"].get<int>() : 4000;
        std::string streamPlugin;
        if (streamConfig.contains("plugin") && streamConfig["plugin"].is_string()) {
            streamPlugin = streamConfig["plugin"].get<std::string>();
        }

        // ===== 模型（models[] / model_name / default_model）=====
        std::vector<ModelStageConfig> stages;
        std::string modelName, modelPath;
        AIPipelineMode mode = AIPipelineMode::Parallel;
        if (streamConfig.contains("models") && streamConfig["models"].is_array() &&
            !streamConfig["models"].empty()) {
            for (const auto& modelEl : streamConfig["models"]) {
                ModelStageConfig stage;
                if (modelEl.contains("name")) stage.modelName = modelEl["name"].get<std::string>();
                if (modelEl.contains("plugin") && modelEl["plugin"].is_string())
                    stage.pluginPath = modelEl["plugin"].get<std::string>();
                else if (!streamPlugin.empty())
                    stage.pluginPath = streamPlugin;
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
                else stage.confThreshold = confThres;
                if (modelEl.contains("nms_threshold")) stage.nmsThreshold = modelEl["nms_threshold"];
                else stage.nmsThreshold = nmsThres;
                if (modelEl.contains("roi_from_previous") && modelEl["roi_from_previous"].get<bool>()) {
                    stage.roiFromPrevious = true;
                    mode = AIPipelineMode::Serial;
                }
                if (modelEl.contains("independent") && modelEl["independent"].get<bool>()) {
                    stage.independent = true;
                }
                if (modelEl.contains("params") && modelEl["params"].is_object()) {
                    stage.params = modelEl["params"];
                }
                if (!stage.modelPath.empty() && stage.modelPath != "none") {
                    stages.push_back(stage);
                }
            }
            if (!stages.empty()) {
                stages[0].roiFromPrevious = false;
                modelName = stages[0].modelName;
                modelPath = stages[0].modelPath;
            }
        } else if (streamConfig.contains("model_name")) {
            modelName = streamConfig["model_name"].get<std::string>();
            modelPath = resolve_model_path(modelName);
        } else {
            modelName = defaultModel;
            modelPath = resolve_model_path(defaultModel);
        }

        // ===== 每路输入拆两条流（对齐 AX650 主码流 + AI 流）=====
        // 共享：主码流解码帧 -> AI 流；AI 推理结果 -> 主码流 OSD
        auto broker = std::make_shared<FrameBroker>();
        auto aiResult = std::make_shared<SharedAIResult>();

        // --- 主码流（输出流：RTP->MediaMTX 或离线文件）---
        StreamConfig scMain;
        scMain.streamId = streamIdBase++;
        scMain.inputSource = inputSource;
        scMain.inputCodec = inputCodec;
        scMain.enableAI = false;  // 主码流不做推理，只负责 解码->缩放->画框->编码->推流
        scMain.outputWidth = outWidth;
        scMain.outputHeight = outHeight;
        scMain.fps = fps;
        scMain.bitrateKbps = bitrateKbps;
        scMain.mediamtxEndpoint = mediamtxHost + ":" + mediamtxPort;
        if (offlineMode) {
            scMain.isFileOutput = true;
            if (streamConfig.contains("output_path") && streamConfig["output_path"].is_string()) {
                scMain.outputFilePath = streamConfig["output_path"].get<std::string>();
            } else if (!outputPath.empty()) {
                scMain.outputFilePath = outputPath;
            }
            if (streamCount > 1) {
                size_t pos = scMain.outputFilePath.rfind('.');
                std::string suffix = "_" + std::to_string(scMain.streamId);
                if (pos != std::string::npos) {
                    scMain.outputFilePath = scMain.outputFilePath.substr(0, pos) + suffix +
                                            scMain.outputFilePath.substr(pos);
                } else {
                    scMain.outputFilePath += suffix;
                }
            }
            // 与 AX650 一致：输出以 .mp4 结尾时先写 raw H.264，主程序结束后 ffmpeg 封装
            if (scMain.outputFilePath.size() > 4 &&
                scMain.outputFilePath.compare(scMain.outputFilePath.size() - 4, 4, ".mp4") == 0) {
                scMain.outputFilePath += ".tmp.h264";
            }
            ALOGN("[VideoStreamManager] Main stream %d: offline raw output -> %s",
                  scMain.streamId, scMain.outputFilePath.c_str());
        } else {
            scMain.isMediaMTXOutput = true;
            ALOGN("[VideoStreamManager] Main stream %d: RTP -> MediaMTX %s:%s",
                  scMain.streamId, mediamtxHost.c_str(), mediamtxPort.c_str());
        }

        // --- AI 流（推理流：broker -> RGA 640 -> RKNN -> SharedAIResult）---
        StreamConfig scAi;
        scAi.streamId = streamIdBase++;
        scAi.inputSource = inputSource;
        scAi.inputCodec = inputCodec;
        scAi.enableAI = true;
        scAi.modelPath = modelPath;
        scAi.modelName = modelName;
        scAi.pluginPath = streamPlugin;
        scAi.confThreshold = confThres;
        scAi.nmsThreshold = nmsThres;
        scAi.aiOutputWidth = aiW;
        scAi.aiOutputHeight = aiH;
        scAi.aiFps = aiFps;
        scAi.modelStages = stages;
        scAi.aiPipelineMode = mode;

        auto mainStream = std::make_unique<VideoStream>(scMain);
        auto aiStream = std::make_unique<VideoStream>(scAi);
        mainStream->attachFrameBroker(broker);
        mainStream->attachAIResult(aiResult);
        aiStream->attachFrameBroker(broker);
        aiStream->attachAIResult(aiResult);

        streams_.push_back(std::move(mainStream));
        aiStreamMap_[scMain.streamId] = static_cast<int>(streams_.size()) - 1;
        streams_.push_back(std::move(aiStream));
        aiStreamMap_[scAi.streamId] = static_cast<int>(streams_.size()) - 1;

        ALOGN("[VideoStreamManager] Input loaded: main=%d ai=%d input=%s model=%s stages=%zu",
              scMain.streamId, scAi.streamId, inputSource.c_str(), modelPath.c_str(),
              stages.size());
    }

    if (streams_.empty()) {
        ALOGE("[VideoStreamManager] No valid streams in config");
        return false;
    }
    ALOGN("[VideoStreamManager] Loaded %zu streams (%zu inputs)",
          streams_.size(), streams_.size() / 2);
    return true;
}

void VideoStreamManager::handleConfigUpdate(const ConfigUpdate& update) {
    auto applyToStream = [&](const std::unique_ptr<VideoStream>& stream) {
        if (!stream->isAIStream()) return;  // 只更新 AI 流的模型/阈值
        if (update.valid && !update.modelPath.empty() && update.modelPath != "none") {
            stream->setModelPath(update.modelPath);
            stream->setModelName(update.modelName);
        }
        if (update.confThreshold > 0.0f) {
            stream->setThresholds(update.confThreshold, update.nmsThreshold);
        }
    };

    if (update.streamId < 0) {
        ALOGN("[VideoStreamManager] Applying global config update to %zu streams", streams_.size());
        for (auto& s : streams_) applyToStream(s);
        return;
    }
    for (auto& s : streams_) {
        if (s->getStreamId() == update.streamId) {
            ALOGN("[VideoStreamManager] Applying config update to stream %d", update.streamId);
            applyToStream(s);
            break;
        }
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
    if (!result) return;
    std::lock_guard<std::mutex> osdLock(osdMapMutex_);
    auto it = osdTargetMap_.find(aiStreamId);
    if (it == osdTargetMap_.end()) return;
    OSDAssociatedModel* model = it->second;
    std::lock_guard<std::mutex> resultLock(model->resultMutex);
    model->latestResult = *result;
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
    bool anyMain = false;
    for (const auto& stream : streams_) {
        if (stream->isMainStream()) {
            anyMain = true;
            if (!stream->isEnded()) return false;
        }
    }
    return anyMain;
}
