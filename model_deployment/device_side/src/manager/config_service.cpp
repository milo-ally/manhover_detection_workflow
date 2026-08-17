#include "config_service.h"
#include "../../utilities/sample_log.h"
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>

// 构造函数，初始化配置文件路径
ConfigService::ConfigService(const std::string& configPath) 
    : configPath_(configPath) {}

// 析构函数，停止监控线程
ConfigService::~ConfigService() {
    stopMonitoring();
}

// 启动配置文件监控线程
void ConfigService::startMonitoring() {
    if (running_) return; 
    shutdownRequested_ = false;
    running_ = true;
    monitorThread_ = std::thread(&ConfigService::monitorConfig, this);
}

// 停止配置文件监控线程
void ConfigService::stopMonitoring() {
    if (!running_) return; 
    shutdownRequested_ = true;
    running_ = false;
    if (monitorThread_.joinable()) {
        monitorThread_.join();
    }
}

// 注册配置变更监听器
void ConfigService::registerConfigListener(std::function<void(const ConfigUpdate&)> listener) {
    std::lock_guard<std::mutex> lock(listenersMutex_);
    listeners_.push_back(std::move(listener));
    ALOGN("[Config] Registered config listener, total listeners: %zu", listeners_.size());
}

// 获取当前模型名称
std::string ConfigService::getCurrentModel() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return currentModelName_;
}

// 获取置信度阈值
float ConfigService::getConfThreshold() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return confThreshold_;
}

// 获取NMS阈值
float ConfigService::getNmsThreshold() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return nmsThreshold_;
}

// 判断配置是否有效
bool ConfigService::isConfigValid() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return configValid_;
}

// 配置文件监控线程函数
void ConfigService::monitorConfig() {
    // st_mtime 只有「秒」精度；前端快速連點兩次可能落在同一秒，
    // 造成第二次更新不觸發 reload。改用 (sec,nsec) 進行判斷。
    struct timespec lastMtim;
    lastMtim.tv_sec = 0;
    lastMtim.tv_nsec = 0;
    off_t lastSize = 0;  // 記錄上次文件大小，用於檢測文件變化
    
    ALOGN("[Config] monitorConfig thread started, watching: %s", configPath_.c_str());

    while (running_ && !shutdownRequested_) {
        struct stat st;
        if (stat(configPath_.c_str(), &st) == 0) {
            // 配置文件存在，判断是否被修改
            // POSIX: st_mtim 提供奈秒；若環境不支援，至少 st_mtime 仍可用（但我們優先用 st_mtim）
            bool changed = false;
#if defined(__APPLE__)
            // macOS uses st_mtimespec
            // 只檢測變化，不立即更新 lastMtim（在 loadConfig() 成功後再更新）
            if (st.st_mtimespec.tv_sec != lastMtim.tv_sec || st.st_mtimespec.tv_nsec != lastMtim.tv_nsec) {
                changed = true;
                ALOGN("[Config] File mtime changed: (%ld,%ld) -> (%ld,%ld)", 
                      (long)lastMtim.tv_sec, (long)lastMtim.tv_nsec,
                      (long)st.st_mtimespec.tv_sec, (long)st.st_mtimespec.tv_nsec);
            }
#else
            if (st.st_mtim.tv_sec != lastMtim.tv_sec || st.st_mtim.tv_nsec != lastMtim.tv_nsec) {
                changed = true;
                ALOGN("[Config] File mtime changed: (%ld,%ld) -> (%ld,%ld)", 
                      (long)lastMtim.tv_sec, (long)lastMtim.tv_nsec,
                      (long)st.st_mtim.tv_sec, (long)st.st_mtim.tv_nsec);
            }
#endif

            // 再加一層保險：有些系統/檔案系統時間戳仍可能不變，st_size 變化也視為更新
            // 只檢測變化，不立即更新 lastSize（在 loadConfig() 成功後再更新）
            if (st.st_size != lastSize) {
                ALOGN("[Config] File size changed: %ld -> %ld", (long)lastSize, (long)st.st_size);
                changed = true;
            }
            
            // 時間戳與大小都未變時：不讀檔、不打 log，直接 sleep，減少 I/O 與日誌對長時間運行的影響
            static std::string lastContent;
            static bool lastContentInitialized = false;
            if (!changed) {
                // 不讀取文件，不輸出任何日誌，直接 sleep 後繼續
            } else {
                // mtime 或 size 有變化：讀取文件內容比較，避免 SFTP 覆蓋未改 mtime 的情況
                std::ifstream checkFile(configPath_);
                if (checkFile.is_open()) {
                    try {
                        std::string fileContent((std::istreambuf_iterator<char>(checkFile)),
                                                std::istreambuf_iterator<char>());
                        
                        if (!lastContentInitialized) {
                            lastContent = fileContent;
                            lastContentInitialized = true;
                            ALOGN("[Config] Initialized file content cache, size=%zu bytes", fileContent.size());
                        } else if (fileContent != lastContent) {
                            changed = true;
                            ALOGN("[Config] File content changed (content comparison), triggering reload");
                            ALOGN("[Config] Old content size=%zu, new content size=%zu", lastContent.size(), fileContent.size());
                        }
                        // 若 mtime/size 變但內容相同，仍保持 changed=true，後續會 reload（安全）
                    } catch (const std::exception& e) {
                        ALOGW("[Config] Error reading file content: %s", e.what());
                    } catch (...) {
                        ALOGW("[Config] Unknown error reading file content");
                    }
                }
            }

            if (changed) {
                ALOGN("[Config] File changed detected: size=%ld, mtime=(%ld,%ld)", 
                      (long)st.st_size, (long)st.st_mtim.tv_sec, (long)st.st_mtim.tv_nsec);
                
                // 在 loadConfig() 之前保存當前文件內容，以便成功後更新 lastContent
                std::string currentFileContent;
                {
                    std::ifstream saveFile(configPath_);
                    if (saveFile.is_open()) {
                        try {
                            currentFileContent = std::string((std::istreambuf_iterator<char>(saveFile)),
                                                            std::istreambuf_iterator<char>());
                        } catch (...) {
                            // 忽略讀取錯誤
                        }
                    }
                }
                
                bool configLoaded = loadConfig();
                ALOGN("[Config] loadConfig returned: %d", configLoaded ? 1 : 0);
                if (configLoaded) {
                    // 只有在 loadConfig() 成功後才更新 lastContent 和 lastSize，確保狀態一致
                    if (!currentFileContent.empty()) {
                        lastContent = currentFileContent;
                        ALOGN("[Config] Updated file content cache after successful loadConfig");
                    }
                    // 更新 lastSize 和 lastMtim，確保下次檢測時不會重複觸發
                    lastSize = st.st_size;
#if defined(__APPLE__)
                    lastMtim = st.st_mtimespec;
#else
                    lastMtim = st.st_mtim;
#endif
                    ALOGN("[Config] Updated file metadata cache: size=%ld, mtime=(%ld,%ld)", 
                          (long)lastSize, (long)lastMtim.tv_sec, (long)lastMtim.tv_nsec);
                    // loadConfig 會自動處理按流配置（在 loadConfig 內部直接觸發更新）
                    // 這裡只需要處理全局配置模式
                    // 檢查配置文件是否包含 streams 數組
                    bool isStreamConfig = false;
                    {
                        std::ifstream checkFile(configPath_);
                        if (checkFile.is_open()) {
                            try {
                                nlohmann::json checkConfig;
                                checkFile >> checkConfig;
                                isStreamConfig = checkConfig.contains("streams") && 
                                                checkConfig["streams"].is_array() && 
                                                checkConfig["streams"].size() > 0;
                                ALOGN("[Config] Re-checked config file: isStreamConfig=%d", isStreamConfig ? 1 : 0);
                            } catch (...) {
                                // 忽略解析錯誤
                            }
                        }
                    }
                    
                    // 只有在全局配置模式下才觸發全局更新
                    // 按流配置模式已在 loadConfig 中直接觸發更新
                    if (!isStreamConfig) {
                        ALOGN("[Config] Triggering global config update");
                        ConfigUpdate update;
                        {
                            std::lock_guard<std::mutex> lock(configMutex_);
                            update.modelName = currentModelName_;
                            update.modelPath = (currentModelName_ != "none") ? 
                                              getModelPath(currentModelName_) : "";
                            update.confThreshold = confThreshold_;
                            update.nmsThreshold = nmsThreshold_;
                            update.valid = configValid_;
                            update.streamId = -1;  // 全局更新
                            update.cameraId = -1;
                        }
                        
                        // 通知所有监听器（需要复制listeners_以避免在回调中修改）
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
            }
        } else {
            // 配置文件不存在
            if (lastMtim.tv_sec != 0 || lastMtim.tv_nsec != 0) {
                // 文件被删除，通知监听器
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
                    update.streamId = -1;  // 全局更新
                    update.cameraId = -1;
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
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 休眠500ms
    }
}

// 加载配置文件，解析JSON内容
bool ConfigService::loadConfig() {
    ALOGN("[Config] loadConfig: Attempting to load config from: %s", configPath_.c_str());
    std::ifstream file(configPath_);
    if (!file.is_open()) {
        // 配置文件不存在
        ALOGW("[Config] Config file not found: %s", configPath_.c_str());
        std::lock_guard<std::mutex> lock(configMutex_);
        if (currentModelName_ != "none") {
            currentModelName_ = "none";
            configValid_ = false;
            return true; // 触发更新
        }
        return false;
    }
    // 空檔或寫入中（先 truncate 再 write）會導致 parse 失敗，直接跳過並保留原配置
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
        
        // [Debug] 輸出配置文件的實際內容（前500字符）
        std::string configDump = config.dump();
        size_t dumpLen = configDump.length() > 500 ? 500 : configDump.length();
        ALOGN("[Config] Config content (first %zu chars): %s", dumpLen, configDump.substr(0, dumpLen).c_str());
        
        std::lock_guard<std::mutex> lock(configMutex_);
        bool configChanged = false;
        bool hasStreamConfig = false;  // 標記是否使用按流配置
        
        // 支持按流配置：優先解析 streams 數組，向後兼容全局 model_name
        ALOGN("[Config] Checking config structure: has streams=%d, has model_name=%d", 
              config.contains("streams") ? 1 : 0, config.contains("model_name") ? 1 : 0);
        if (config.contains("streams") && config["streams"].is_array()) {
            hasStreamConfig = true;
            size_t streamCount = config["streams"].size();
            ALOGN("[Config] Found streams array with %zu entries", streamCount);
            // 按流配置模式：每個流可以有不同的模型
            // 格式1（舊）：{"streams": [{"stream_id": 2, "model_name": "helmet", ...}, ...]}
            // 格式2（新）：{"streams": [{"stream_id": 2, "models": [{"name": "...", "roi_from_previous": true}, ...]}, ...]}
            for (const auto& streamConfig : config["streams"]) {
                if (!streamConfig.contains("stream_id")) continue;
                int streamId = streamConfig["stream_id"];
                
                ConfigUpdate streamUpdate;
                streamUpdate.streamId = streamId;
                streamUpdate.cameraId = -1;
                streamUpdate.confThreshold = streamConfig.contains("conf_thres") ? 
                                             streamConfig["conf_thres"].get<float>() : confThreshold_;
                streamUpdate.nmsThreshold = streamConfig.contains("nms_thres") ? 
                                            streamConfig["nms_thres"].get<float>() : nmsThreshold_;
                
                // 優先處理 models 數組格式（支持串行/並行）
                if (streamConfig.contains("models") && streamConfig["models"].is_array() && 
                    !streamConfig["models"].empty()) {
                    streamUpdate.modelStages.clear();
                    bool anyRoiFromPrevious = false;
                    for (const auto& modelEl : streamConfig["models"]) {
                        ModelStageConfig stage;
                        if (modelEl.contains("name")) stage.modelName = modelEl["name"];
                        if (modelEl.contains("path") && !modelEl["path"].is_null()) {
                            stage.modelPath = modelEl["path"];
                            if (stage.modelPath.find("/") == std::string::npos || stage.modelPath.find("../") == 0) {
                                if (!stage.modelName.empty()) {
                                    std::string fp = getModelPath(stage.modelName);
                                    if (!fp.empty()) stage.modelPath = fp;
                                }
                            }
                        } else if (!stage.modelName.empty()) {
                            stage.modelPath = getModelPath(stage.modelName);
                        }
                        if (modelEl.contains("conf_threshold")) stage.confThreshold = modelEl["conf_threshold"];
                        else stage.confThreshold = streamUpdate.confThreshold;
                        if (modelEl.contains("nms_threshold")) stage.nmsThreshold = modelEl["nms_threshold"];
                        else stage.nmsThreshold = streamUpdate.nmsThreshold;
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
                        if (!stage.modelPath.empty() && stage.modelPath != "none") {
                            if (access(stage.modelPath.c_str(), F_OK) != 0) {
                                ALOGE("[Config] Model file not found for stream %d stage: %s", streamId, stage.modelPath.c_str());
                                continue;
                            }
                            streamUpdate.modelStages.push_back(stage);
                        }
                    }
                    {
                        bool hasBaseStage = false;
                        bool hasRoiStage = false;
                        for (const auto& s : streamUpdate.modelStages) {
                            if (s.roiFromPrevious)
                                hasRoiStage = true;
                            else
                                hasBaseStage = true;
                        }
                        if (hasBaseStage && hasRoiStage && streamUpdate.modelStages.size() > 1) {
                            std::stable_partition(streamUpdate.modelStages.begin(),
                                                  streamUpdate.modelStages.end(),
                                                  [](const ModelStageConfig& s) { return !s.roiFromPrevious; });
                        }
                    }
                    // 管線第一個階段不可能有「前序 ROI」；誤標會導致首輪無全圖推理、合併結果為空
                    if (!streamUpdate.modelStages.empty()) streamUpdate.modelStages[0].roiFromPrevious = false;
                    if (anyRoiFromPrevious) streamUpdate.aiPipelineMode = AIPipelineMode::Serial;
                    if (!streamUpdate.modelStages.empty()) {
                        streamUpdate.modelName = streamUpdate.modelStages[0].modelName;
                        streamUpdate.modelPath = streamUpdate.modelStages[0].modelPath;
                        streamUpdate.valid = true;
                        ALOGN("[Config] Stream %d: parsed %zu model stages (mode=%s)", streamId, 
                              streamUpdate.modelStages.size(), 
                              streamUpdate.aiPipelineMode == AIPipelineMode::Serial ? "serial" : "parallel");
                    } else {
                        streamUpdate.valid = false;
                        streamUpdate.modelName = "none";
                    }
                } else if (streamConfig.contains("model_name")) {
                    // 向後兼容：單一 model_name
                    std::string newModel = streamConfig["model_name"];
                    std::string modelPath = getModelPath(newModel);
                    if (newModel != "none" && !modelPath.empty() && access(modelPath.c_str(), F_OK) != 0) {
                        ALOGE("[Config] Model file not found for stream %d: %s", streamId, modelPath.c_str());
                        continue;
                    }
                    streamUpdate.modelName = newModel;
                    streamUpdate.modelPath = (newModel != "none") ? modelPath : "";
                    streamUpdate.valid = (newModel != "none");
                } else {
                    streamUpdate.modelName = "none";
                    streamUpdate.valid = false;
                }
                
                ALOGN("[Config] Stream-specific update: streamId=%d, model=%s, path=%s, valid=%d, stages=%zu", 
                      streamId, streamUpdate.modelName.c_str(), streamUpdate.modelPath.c_str(), 
                      streamUpdate.valid ? 1 : 0, streamUpdate.modelStages.size());
                
                // 通知監聽器（按流更新）
                std::vector<std::function<void(const ConfigUpdate&)>> listenersCopy;
                {
                    std::lock_guard<std::mutex> lock(listenersMutex_);
                    listenersCopy = listeners_;
                }
                for (const auto& listener : listenersCopy) {
                    listener(streamUpdate);
                }
                configChanged = true;
            }
        } else if (config.contains("model_name")) {
            // 全局配置模式（向後兼容）：更新所有流
            ALOGN("[Config] Using global config mode (model_name found)");
            std::string newModel = config["model_name"];
            // 增加校验：只有当模型文件确实存在(或是none)时才接受配置
            std::string modelPath = getModelPath(newModel);
            if (newModel != "none" && !modelPath.empty() && access(modelPath.c_str(), F_OK) != 0) {
                ALOGE("[Config] Model file not found: %s. Keeping previous model: %s", 
                      modelPath.c_str(), currentModelName_.c_str());
            } else {
                if (newModel != currentModelName_) {
                    currentModelName_ = newModel;
                    configChanged = true;
                }
            }
        } else {
            if (currentModelName_ != "none") {
                currentModelName_ = "none";
                configChanged = true;
            }
        }
        
        // 解析置信度阈值
        if (config.contains("conf_thres")) {
            float newConf = config["conf_thres"].get<float>();
            // 增加范围校验 Clamp (0.01 ~ 1.0)
            if (newConf < 0.0f) newConf = 0.01f;
            if (newConf > 1.0f) newConf = 1.0f;
            
            if (newConf != confThreshold_) {
                confThreshold_ = newConf;
                configChanged = true;
            }
        }
        
        // 解析NMS阈值
        if (config.contains("nms_thres")) {
            float newNms = config["nms_thres"].get<float>();
            // 增加范围校验
            if (newNms < 0.0f) newNms = 0.01f;
            if (newNms > 1.0f) newNms = 1.0f;

            if (newNms != nmsThreshold_) {
                nmsThreshold_ = newNms;
                configChanged = true;
            }
        }
        
        configValid_ = true;
        
        // 如果使用按流配置，標記為已處理（不需要在 monitorConfig 中再次觸發全局更新）
        if (hasStreamConfig) {
            // 按流配置已在上面直接觸發更新，不需要返回 configChanged
            // 但需要返回 true 表示配置已加載（即使內容相同，也要返回 true 以確保監聽器被調用）
            ALOGN("[Config] Stream config processed, returning true (configChanged=%d)", configChanged ? 1 : 0);
            return true;  // 總是返回 true，確保按流更新被觸發
        }
        
        return configChanged;
    } catch (const std::exception& e) {
        ALOGE("[Config] Parse error: %s - keeping previous config", e.what());
        return false;
    }
}

// 根据模型名称获取模型文件路径（公开方法）
std::string ConfigService::getModelPath(const std::string& modelName) const {
    if (modelName == "none" || modelName.empty()) {
        return "";
    }
    
    std::string modelPath = modelName;
    if (modelPath.find(".axmodel") == std::string::npos) {
        modelPath += ".axmodel";
    }
    
    if (modelPath.find("/") != std::string::npos) {
        return modelPath;
    }
    
    // 特殊处理helmet模型（優先檢查完整文件名）
    if (modelPath == "yolo11_helmet.axmodel" || 
        modelPath == "lyg_helmet.axmodel" || 
        (modelPath.find("helmet") != std::string::npos && modelPath.find("pose") == std::string::npos)) {
        return "../models/yolo11_helmet.axmodel";
    }
    
    // 特殊处理跌倒检测模型（優先檢查完整文件名）
    if (modelPath == "yolo11_pose_cut.axmodel" || 
        modelPath.find("pose") != std::string::npos || 
        modelPath.find("fall") != std::string::npos) {
        return "../models/yolo11_pose_cut.axmodel";
    }

    // 人員聚集：與人員偵測共用同一模型文件（human_detection.axmodel），插件由 modelName 區分
    if (modelPath.find("crowd") != std::string::npos || modelPath.find("human_group") != std::string::npos) {
        return "../models/yolo11_human_detection.axmodel";
    }

    // 人員偵測
    if (modelPath.find("human") != std::string::npos) {
        return "../models/yolo11_human_detection.axmodel";
    }

    // 人臉檢測
    if (modelPath.find("face_detector") != std::string::npos ||
        (modelPath.find("face") != std::string::npos &&
         modelPath.find("arcface") == std::string::npos &&
         modelPath.find("recognition") == std::string::npos)) {
        return "../models/face_detector_cut.axmodel";
    }

    // 人臉辨識（ArcFace）
    if (modelPath.find("arcface") != std::string::npos ||
        modelPath.find("face_rec") != std::string::npos ||
        modelPath.find("recognition") != std::string::npos) {
        return "../models/arcface_model2.axmodel";
    }
    
    return "../models/" + modelPath;
}

