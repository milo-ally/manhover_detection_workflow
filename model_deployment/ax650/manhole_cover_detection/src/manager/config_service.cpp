#include "config_service.h"
#include "../../utilities/sample_log.h"
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>

// 鏋勯€犲嚱鏁帮紝鍒濆鍖栭厤缃枃浠惰矾寰?
ConfigService::ConfigService(const std::string& configPath) 
    : configPath_(configPath) {}

// 鏋愭瀯鍑芥暟锛屽仠姝㈢洃鎺х嚎绋?
ConfigService::~ConfigService() {
    stopMonitoring();
}

// 鍚姩閰嶇疆鏂囦欢鐩戞帶绾跨▼
void ConfigService::startMonitoring() {
    if (running_) return; 
    shutdownRequested_ = false;
    running_ = true;
    monitorThread_ = std::thread(&ConfigService::monitorConfig, this);
}

// 鍋滄閰嶇疆鏂囦欢鐩戞帶绾跨▼
void ConfigService::stopMonitoring() {
    if (!running_) return; 
    shutdownRequested_ = true;
    running_ = false;
    if (monitorThread_.joinable()) {
        monitorThread_.join();
    }
}

// 娉ㄥ唽閰嶇疆鍙樻洿鐩戝惉鍣?
void ConfigService::registerConfigListener(std::function<void(const ConfigUpdate&)> listener) {
    std::lock_guard<std::mutex> lock(listenersMutex_);
    listeners_.push_back(std::move(listener));
    ALOGN("[Config] Registered config listener, total listeners: %zu", listeners_.size());
}

// 鑾峰彇褰撳墠妯″瀷鍚嶇О
std::string ConfigService::getCurrentModel() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return currentModelName_;
}

// 鑾峰彇缃俊搴﹂槇鍊?
float ConfigService::getConfThreshold() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return confThreshold_;
}

// 鑾峰彇NMS闃堝€?
float ConfigService::getNmsThreshold() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return nmsThreshold_;
}

// 鍒ゆ柇閰嶇疆鏄惁鏈夋晥
bool ConfigService::isConfigValid() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return configValid_;
}

// 閰嶇疆鏂囦欢鐩戞帶绾跨▼鍑芥暟
void ConfigService::monitorConfig() {
    // st_mtime 鍙湁銆岀銆嶇簿搴︼紱鍓嶇蹇€熼€ｉ粸鍏╂鍙兘钀藉湪鍚屼竴绉掞紝
    // 閫犳垚绗簩娆℃洿鏂颁笉瑙哥櫦 reload銆傛敼鐢?(sec,nsec) 閫茶鍒ゆ柗銆?
    struct timespec lastMtim;
    lastMtim.tv_sec = 0;
    lastMtim.tv_nsec = 0;
    off_t lastSize = 0;  // 瑷橀寗涓婃鏂囦欢澶у皬锛岀敤鏂兼娓枃浠惰畩鍖?
    
    ALOGN("[Config] monitorConfig thread started, watching: %s", configPath_.c_str());

    while (running_ && !shutdownRequested_) {
        struct stat st;
        if (stat(configPath_.c_str(), &st) == 0) {
            // 閰嶇疆鏂囦欢瀛樺湪锛屽垽鏂槸鍚﹁淇敼
            // POSIX: st_mtim 鎻愪緵濂堢锛涜嫢鐠板涓嶆敮鎻达紝鑷冲皯 st_mtime 浠嶅彲鐢紙浣嗘垜鍊戝劒鍏堢敤 st_mtim锛?
            bool changed = false;
#if defined(__APPLE__)
            // macOS uses st_mtimespec
            // 鍙娓畩鍖栵紝涓嶇珛鍗虫洿鏂?lastMtim锛堝湪 loadConfig() 鎴愬姛寰屽啀鏇存柊锛?
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

            // 鍐嶅姞涓€灞や繚闅細鏈変簺绯荤当/妾旀绯荤当鏅傞枔鎴充粛鍙兘涓嶈畩锛宻t_size 璁婂寲涔熻鐐烘洿鏂?
            // 鍙娓畩鍖栵紝涓嶇珛鍗虫洿鏂?lastSize锛堝湪 loadConfig() 鎴愬姛寰屽啀鏇存柊锛?
            if (st.st_size != lastSize) {
                ALOGN("[Config] File size changed: %ld -> %ld", (long)lastSize, (long)st.st_size);
                changed = true;
            }
            
            // 鏅傞枔鎴宠垏澶у皬閮芥湭璁婃檪锛氫笉璁€妾斻€佷笉鎵?log锛岀洿鎺?sleep锛屾笡灏?I/O 鑸囨棩瑾屽皪闀锋檪闁撻亱琛岀殑褰遍熆
            static std::string lastContent;
            static bool lastContentInitialized = false;
            if (!changed) {
                // 涓嶈畝鍙栨枃浠讹紝涓嶈几鍑轰换浣曟棩瑾岋紝鐩存帴 sleep 寰岀辜绾?
            } else {
                // mtime 鎴?size 鏈夎畩鍖栵細璁€鍙栨枃浠跺収瀹规瘮杓冿紝閬垮厤 SFTP 瑕嗚搵鏈敼 mtime 鐨勬儏娉?
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
                        // 鑻?mtime/size 璁婁絾鍏у鐩稿悓锛屼粛淇濇寔 changed=true锛屽緦绾屾渻 reload锛堝畨鍏級
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
                
                // 鍦?loadConfig() 涔嬪墠淇濆瓨鐣跺墠鏂囦欢鍏у锛屼互渚挎垚鍔熷緦鏇存柊 lastContent
                std::string currentFileContent;
                {
                    std::ifstream saveFile(configPath_);
                    if (saveFile.is_open()) {
                        try {
                            currentFileContent = std::string((std::istreambuf_iterator<char>(saveFile)),
                                                            std::istreambuf_iterator<char>());
                        } catch (...) {
                            // 蹇界暐璁€鍙栭尟瑾?
                        }
                    }
                }
                
                bool configLoaded = loadConfig();
                ALOGN("[Config] loadConfig returned: %d", configLoaded ? 1 : 0);
                if (configLoaded) {
                    // 鍙湁鍦?loadConfig() 鎴愬姛寰屾墠鏇存柊 lastContent 鍜?lastSize锛岀⒑淇濈媭鎱嬩竴鑷?
                    if (!currentFileContent.empty()) {
                        lastContent = currentFileContent;
                        ALOGN("[Config] Updated file content cache after successful loadConfig");
                    }
                    // 鏇存柊 lastSize 鍜?lastMtim锛岀⒑淇濅笅娆℃娓檪涓嶆渻閲嶈瑙哥櫦
                    lastSize = st.st_size;
#if defined(__APPLE__)
                    lastMtim = st.st_mtimespec;
#else
                    lastMtim = st.st_mtim;
#endif
                    ALOGN("[Config] Updated file metadata cache: size=%ld, mtime=(%ld,%ld)", 
                          (long)lastSize, (long)lastMtim.tv_sec, (long)lastMtim.tv_nsec);
                    // loadConfig 鏈冭嚜鍕曡檿鐞嗘寜娴侀厤缃紙鍦?loadConfig 鍏ч儴鐩存帴瑙哥櫦鏇存柊锛?
                    // 閫欒！鍙渶瑕佽檿鐞嗗叏灞€閰嶇疆妯″紡
                    // 妾㈡煡閰嶇疆鏂囦欢鏄惁鍖呭惈 streams 鏁哥祫
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
                                // 蹇界暐瑙ｆ瀽閷
                            }
                        }
                    }
                    
                    // 鍙湁鍦ㄥ叏灞€閰嶇疆妯″紡涓嬫墠瑙哥櫦鍏ㄥ眬鏇存柊
                    // 鎸夋祦閰嶇疆妯″紡宸插湪 loadConfig 涓洿鎺ヨЦ鐧兼洿鏂?
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
                            update.streamId = -1;  // 鍏ㄥ眬鏇存柊
                            update.cameraId = -1;
                        }
                        
                        // 閫氱煡鎵€鏈夌洃鍚櫒锛堥渶瑕佸鍒秎isteners_浠ラ伩鍏嶅湪鍥炶皟涓慨鏀癸級
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
            // 閰嶇疆鏂囦欢涓嶅瓨鍦?
            if (lastMtim.tv_sec != 0 || lastMtim.tv_nsec != 0) {
                // 鏂囦欢琚垹闄わ紝閫氱煡鐩戝惉鍣?
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
                    update.streamId = -1;  // 鍏ㄥ眬鏇存柊
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
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 浼戠湢500ms
    }
}

// 鍔犺浇閰嶇疆鏂囦欢锛岃В鏋怞SON鍐呭
bool ConfigService::loadConfig() {
    ALOGN("[Config] loadConfig: Attempting to load config from: %s", configPath_.c_str());
    std::ifstream file(configPath_);
    if (!file.is_open()) {
        // 閰嶇疆鏂囦欢涓嶅瓨鍦?
        ALOGW("[Config] Config file not found: %s", configPath_.c_str());
        std::lock_guard<std::mutex> lock(configMutex_);
        if (currentModelName_ != "none") {
            currentModelName_ = "none";
            configValid_ = false;
            return true; // 瑙﹀彂鏇存柊
        }
        return false;
    }
    // 绌烘獢鎴栧鍏ヤ腑锛堝厛 truncate 鍐?write锛夋渻灏庤嚧 parse 澶辨晽锛岀洿鎺ヨ烦閬庝甫淇濈暀鍘熼厤缃?
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
        
        // [Debug] 杓稿嚭閰嶇疆鏂囦欢鐨勫闅涘収瀹癸紙鍓?00瀛楃锛?
        std::string configDump = config.dump();
        size_t dumpLen = configDump.length() > 500 ? 500 : configDump.length();
        ALOGN("[Config] Config content (first %zu chars): %s", dumpLen, configDump.substr(0, dumpLen).c_str());
        
        std::lock_guard<std::mutex> lock(configMutex_);
        bool configChanged = false;
        bool hasStreamConfig = false;  // 妯欒鏄惁浣跨敤鎸夋祦閰嶇疆
        
        // 鏀寔鎸夋祦閰嶇疆锛氬劒鍏堣В鏋?streams 鏁哥祫锛屽悜寰屽吋瀹瑰叏灞€ model_name
        ALOGN("[Config] Checking config structure: has streams=%d, has model_name=%d", 
              config.contains("streams") ? 1 : 0, config.contains("model_name") ? 1 : 0);
        if (config.contains("streams") && config["streams"].is_array()) {
            hasStreamConfig = true;
            size_t streamCount = config["streams"].size();
            ALOGN("[Config] Found streams array with %zu entries", streamCount);
            // 鎸夋祦閰嶇疆妯″紡锛氭瘡鍊嬫祦鍙互鏈変笉鍚岀殑妯″瀷
            // 鏍煎紡1锛堣垔锛夛細{"streams": [{"stream_id": 2, "model_name": "helmet", ...}, ...]}
            // 鏍煎紡2锛堟柊锛夛細{"streams": [{"stream_id": 2, "models": [{"name": "...", "roi_from_previous": true}, ...]}, ...]}
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
                
                // 鍎厛铏曠悊 models 鏁哥祫鏍煎紡锛堟敮鎸佷覆琛?涓﹁锛?
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
                    // 绠＄窔绗竴鍊嬮殠娈典笉鍙兘鏈夈€屽墠搴?ROI銆嶏紱瑾ゆ鏈冨皫鑷撮杓劇鍏ㄥ湒鎺ㄧ悊銆佸悎浣电祼鏋滅偤绌?
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
                    // 鍚戝緦鍏煎锛氬柈涓€ model_name
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
                
                // 閫氱煡鐩ｈ伣鍣紙鎸夋祦鏇存柊锛?
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
            // 鍏ㄥ眬閰嶇疆妯″紡锛堝悜寰屽吋瀹癸級锛氭洿鏂版墍鏈夋祦
            ALOGN("[Config] Using global config mode (model_name found)");
            std::string newModel = config["model_name"];
            // 澧炲姞鏍￠獙锛氬彧鏈夊綋妯″瀷鏂囦欢纭疄瀛樺湪(鎴栨槸none)鏃舵墠鎺ュ彈閰嶇疆
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
        
        // 瑙ｆ瀽缃俊搴﹂槇鍊?
        if (config.contains("conf_thres")) {
            float newConf = config["conf_thres"].get<float>();
            // 澧炲姞鑼冨洿鏍￠獙 Clamp (0.01 ~ 1.0)
            if (newConf < 0.0f) newConf = 0.01f;
            if (newConf > 1.0f) newConf = 1.0f;
            
            if (newConf != confThreshold_) {
                confThreshold_ = newConf;
                configChanged = true;
            }
        }
        
        // 瑙ｆ瀽NMS闃堝€?
        if (config.contains("nms_thres")) {
            float newNms = config["nms_thres"].get<float>();
            // 澧炲姞鑼冨洿鏍￠獙
            if (newNms < 0.0f) newNms = 0.01f;
            if (newNms > 1.0f) newNms = 1.0f;

            if (newNms != nmsThreshold_) {
                nmsThreshold_ = newNms;
                configChanged = true;
            }
        }
        
        configValid_ = true;
        
        // 濡傛灉浣跨敤鎸夋祦閰嶇疆锛屾瑷樼偤宸茶檿鐞嗭紙涓嶉渶瑕佸湪 monitorConfig 涓啀娆¤Ц鐧煎叏灞€鏇存柊锛?
        if (hasStreamConfig) {
            // 鎸夋祦閰嶇疆宸插湪涓婇潰鐩存帴瑙哥櫦鏇存柊锛屼笉闇€瑕佽繑鍥?configChanged
            // 浣嗛渶瑕佽繑鍥?true 琛ㄧず閰嶇疆宸插姞杓夛紙鍗充娇鍏у鐩稿悓锛屼篃瑕佽繑鍥?true 浠ョ⒑淇濈洠鑱藉櫒琚鐢級
            ALOGN("[Config] Stream config processed, returning true (configChanged=%d)", configChanged ? 1 : 0);
            return true;  // 绺芥槸杩斿洖 true锛岀⒑淇濇寜娴佹洿鏂拌瑙哥櫦
        }
        
        return configChanged;
    } catch (const std::exception& e) {
        ALOGE("[Config] Parse error: %s - keeping previous config", e.what());
        return false;
    }
}

// 鏍规嵁妯″瀷鍚嶇О鑾峰彇妯″瀷鏂囦欢璺緞锛堝叕寮€鏂规硶锛?
std::string ConfigService::getModelPath(const std::string& modelName) const {
    if (modelName.empty() || modelName == "none") return "";
    // This deployment contains one model. Preserve the original ConfigService
    // API, but resolve every logical name to the manhole-cover model.
    if (modelName.find("/") != std::string::npos || modelName.find("\\\\") != std::string::npos)
        return modelName;
    if (modelName.find(".axmodel") != std::string::npos)
        return "../models/" + modelName;
    return "../models/yolo11s-manhole-detection.axmodel";
}

