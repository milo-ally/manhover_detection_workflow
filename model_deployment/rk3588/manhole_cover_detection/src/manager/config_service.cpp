#include "config_service.h"
#include "../../utilities/sample_log.h"
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>

ConfigService::ConfigService(const std::string& configPath)
    : configPath_(configPath) {}

ConfigService::~ConfigService() {
    stopMonitoring();
}

void ConfigService::startMonitoring() {
    if (running_) return;
    shutdownRequested_ = false;
    running_ = true;
    monitorThread_ = std::thread(&ConfigService::monitorConfig, this);
}

void ConfigService::stopMonitoring() {
    if (!running_) return;
    shutdownRequested_ = true;
    running_ = false;
    if (monitorThread_.joinable()) {
        monitorThread_.join();
    }
}

void ConfigService::registerConfigListener(std::function<void(const ConfigUpdate&)> listener) {
    std::lock_guard<std::mutex> lock(listenersMutex_);
    listeners_.push_back(std::move(listener));
    ALOGN("[Config] Registered config listener, total listeners: %zu", listeners_.size());
}

std::string ConfigService::getCurrentModel() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return currentModelName_;
}

float ConfigService::getConfThreshold() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return confThreshold_;
}

float ConfigService::getNmsThreshold() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return nmsThreshold_;
}

bool ConfigService::isConfigValid() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return configValid_;
}

std::string ConfigService::getModelPath(const std::string& modelName) const {
    if (modelName.empty() || modelName == "none") return "";
    // 本部署只有一个模型：保留原 ConfigService API，但把所有逻辑名解析到井盖模型。
    if (modelName.find("/") != std::string::npos || modelName.find("\\\\") != std::string::npos)
        return modelName;
    if (modelName.find(".rknn") != std::string::npos)
        return "../models/" + modelName;
    return "../models/manhole-cover-yolo11s-production.rknn";
}

void ConfigService::monitorConfig() {
    struct timespec lastMtim;
    lastMtim.tv_sec = 0;
    lastMtim.tv_nsec = 0;
    off_t lastSize = 0;

    ALOGN("[Config] monitorConfig thread started, watching: %s", configPath_.c_str());

    while (running_ && !shutdownRequested_) {
        struct stat st;
        if (stat(configPath_.c_str(), &st) == 0) {
            bool changed = false;
            if (st.st_mtim.tv_sec != lastMtim.tv_sec || st.st_mtim.tv_nsec != lastMtim.tv_nsec) {
                changed = true;
            }
            if (st.st_size != lastSize) {
                changed = true;
            }
            if (changed) {
                ALOGN("[Config] File changed detected: size=%ld, mtime=(%ld,%ld)",
                      (long)st.st_size, (long)st.st_mtim.tv_sec, (long)st.st_mtim.tv_nsec);
                bool configLoaded = loadConfig();
                if (configLoaded) {
                    lastSize = st.st_size;
                    lastMtim = st.st_mtim;
                    ALOGN("[Config] loadConfig succeeded, metadata cache updated");
                }
            }
        } else {
            // 配置文件被删除：通知监听者清空配置
            if (lastMtim.tv_sec != 0 || lastMtim.tv_nsec != 0) {
                lastMtim.tv_sec = 0;
                lastMtim.tv_nsec = 0;
                lastSize = 0;
                ConfigUpdate update;
                {
                    std::lock_guard<std::mutex> lock(configMutex_);
                    currentModelName_ = "none";
                    configValid_ = false;
                    update.modelName = "none";
                    update.modelPath = "";
                    update.confThreshold = confThreshold_;
                    update.nmsThreshold = nmsThreshold_;
                    update.valid = false;
                    update.streamId = -1;
                }
                std::vector<std::function<void(const ConfigUpdate&)>> listenersCopy;
                {
                    std::lock_guard<std::mutex> lock(listenersMutex_);
                    listenersCopy = listeners_;
                }
                for (const auto& listener : listenersCopy) {
                    listener(update);
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

bool ConfigService::loadConfig() {
    ALOGN("[Config] loadConfig: Attempting to load config from: %s", configPath_.c_str());
    std::ifstream file(configPath_);
    if (!file.is_open()) {
        ALOGW("[Config] Config file not found: %s", configPath_.c_str());
        std::lock_guard<std::mutex> lock(configMutex_);
        if (currentModelName_ != "none") {
            currentModelName_ = "none";
            configValid_ = false;
            return true;  // 触发更新
        }
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    if (fileSize == 0) {
        ALOGW("[Config] Config file empty (size=0), keeping previous config");
        return false;
    }

    try {
        nlohmann::json config;
        file >> config;
        ALOGN("[Config] Successfully parsed JSON config");

        std::lock_guard<std::mutex> lock(configMutex_);
        bool hasStreamConfig = false;

        if (config.contains("streams") && config["streams"].is_array()) {
            hasStreamConfig = true;
            size_t streamCount = config["streams"].size();
            ALOGN("[Config] Found streams array with %zu entries", streamCount);

            for (const auto& streamConfig : config["streams"]) {
                if (!streamConfig.contains("stream_id")) continue;
                int streamId = streamConfig["stream_id"];

                ConfigUpdate streamUpdate;
                streamUpdate.streamId = streamId;
                streamUpdate.confThreshold = streamConfig.contains("conf_thres")
                    ? streamConfig["conf_thres"].get<float>() : confThreshold_;
                streamUpdate.nmsThreshold = streamConfig.contains("nms_thres")
                    ? streamConfig["nms_thres"].get<float>() : nmsThreshold_;

                if (streamConfig.contains("models") && streamConfig["models"].is_array() &&
                    !streamConfig["models"].empty()) {
                    streamUpdate.modelStages.clear();
                    for (const auto& modelEl : streamConfig["models"]) {
                        ModelStageConfig stage;
                        if (modelEl.contains("name")) stage.modelName = modelEl["name"];
                        if (modelEl.contains("plugin") && modelEl["plugin"].is_string())
                            stage.pluginPath = modelEl["plugin"];
                        else if (streamConfig.contains("plugin") && streamConfig["plugin"].is_string())
                            stage.pluginPath = streamConfig["plugin"];
                        if (modelEl.contains("path") && !modelEl["path"].is_null()) {
                            stage.modelPath = modelEl["path"];
                            if (stage.modelPath.find("/") == std::string::npos ||
                                stage.modelPath.find("../") == 0) {
                                if (!stage.modelName.empty()) {
                                    std::string fp = getModelPath(stage.modelName);
                                    if (!fp.empty()) stage.modelPath = fp;
                                }
                            }
                        } else if (!stage.modelName.empty()) {
                            stage.modelPath = getModelPath(stage.modelName);
                        }
                        if (modelEl.contains("conf_threshold"))
                            stage.confThreshold = modelEl["conf_threshold"];
                        else
                            stage.confThreshold = streamUpdate.confThreshold;
                        if (modelEl.contains("nms_threshold"))
                            stage.nmsThreshold = modelEl["nms_threshold"];
                        else
                            stage.nmsThreshold = streamUpdate.nmsThreshold;
                        if (modelEl.contains("roi_from_previous") && modelEl["roi_from_previous"].get<bool>()) {
                            stage.roiFromPrevious = true;
                            streamUpdate.aiPipelineMode = AIPipelineMode::Serial;
                        }
                        if (modelEl.contains("params") && modelEl["params"].is_object()) {
                            stage.params = modelEl["params"];
                        }
                        if (!stage.modelPath.empty() && stage.modelPath != "none") {
                            streamUpdate.modelStages.push_back(stage);
                        }
                    }
                    if (!streamUpdate.modelStages.empty()) {
                        streamUpdate.modelName = streamUpdate.modelStages[0].modelName;
                        streamUpdate.modelPath = streamUpdate.modelStages[0].modelPath;
                        streamUpdate.valid = true;
                    } else {
                        streamUpdate.valid = false;
                        streamUpdate.modelName = "none";
                    }
                } else if (streamConfig.contains("model_name")) {
                    std::string newModel = streamConfig["model_name"];
                    streamUpdate.modelName = newModel;
                    streamUpdate.modelPath = (newModel != "none") ? getModelPath(newModel) : "";
                    streamUpdate.valid = (newModel != "none");
                } else {
                    streamUpdate.modelName = "none";
                    streamUpdate.valid = false;
                }

                ALOGN("[Config] Stream-specific update: streamId=%d, model=%s, path=%s, valid=%d, stages=%zu",
                      streamId, streamUpdate.modelName.c_str(), streamUpdate.modelPath.c_str(),
                      streamUpdate.valid ? 1 : 0, streamUpdate.modelStages.size());

                std::vector<std::function<void(const ConfigUpdate&)>> listenersCopy;
                {
                    std::lock_guard<std::mutex> lock(listenersMutex_);
                    listenersCopy = listeners_;
                }
                for (const auto& listener : listenersCopy) {
                    listener(streamUpdate);
                }
            }
        } else if (config.contains("model_name")) {
            // 全局配置模式（向后兼容）：更新所有流
            ALOGN("[Config] Using global config mode (model_name found)");
            std::string newModel = config["model_name"];
            std::string modelPath = getModelPath(newModel);
            if (newModel != "none" && !modelPath.empty() &&
                access(modelPath.c_str(), F_OK) != 0) {
                ALOGE("[Config] Model file not found: %s. Keeping previous model: %s",
                      modelPath.c_str(), currentModelName_.c_str());
            } else {
                if (newModel != currentModelName_) {
                    currentModelName_ = newModel;
                }
            }
        } else {
            if (currentModelName_ != "none") {
                currentModelName_ = "none";
            }
        }

        // 解析全局阈值
        if (config.contains("conf_thres")) {
            float newConf = config["conf_thres"].get<float>();
            if (newConf < 0.0f) newConf = 0.01f;
            if (newConf > 1.0f) newConf = 1.0f;
            confThreshold_ = newConf;
        }
        if (config.contains("nms_thres")) {
            float newNms = config["nms_thres"].get<float>();
            if (newNms < 0.0f) newNms = 0.01f;
            if (newNms > 1.0f) newNms = 1.0f;
            nmsThreshold_ = newNms;
        }

        configValid_ = true;

        if (hasStreamConfig) {
            ALOGN("[Config] Stream config processed, returning true");
            return true;
        }
        return true;
    } catch (const std::exception& e) {
        ALOGE("[Config] Parse error: %s - keeping previous config", e.what());
        return false;
    }
}
