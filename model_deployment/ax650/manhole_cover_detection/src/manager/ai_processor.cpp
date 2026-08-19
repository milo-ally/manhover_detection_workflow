#include "ai_processor.h"
#include "../../utilities/sample_log.h"
#include "../include/osd_renderer_interface.h"
#include <dlfcn.h>
#include <unistd.h>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include "ax_sys_api.h"

// 构造函数：根据模型路径加载AI模型；modelName 用於區分 crowd vs human（共用同一 path 時）
AIProcessor::AIProcessor(const std::string& modelPath, const std::string& modelName,
                         const nlohmann::json& modelParams) {
    if (!modelPath.empty()) {
        loadModel(modelPath, modelName, modelParams);
    }
}

// 析构函数：卸载AI模型，释放资源
AIProcessor::~AIProcessor() {
    unloadModel();
}

// 加载AI模型插件库和模型实例；modelName 用於插件選擇（crowd 與 human 共用 path 時）
bool AIProcessor::loadModel(const std::string& modelPath, const std::string& modelName,
                            const nlohmann::json& modelParams) {
    std::lock_guard<std::mutex> lock(modelMutex_);
    
    // 先卸载已加载的模型
    if (model_) {
        unloadModel();
    }

    if (modelPath.empty()) {
        return false;
    }

    modelName_ = modelName;
    modelParams_ = modelParams.is_object() ? modelParams : nlohmann::json::object();
    osdRenderer_.reset();
    // 插件/OSD 選擇：有 modelName 時用 modelName（人員聚集與人員偵測共用 path）
    const std::string& pluginHint = modelName_.empty() ? modelPath : modelName_;
    applyModelParamsToEnv(pluginHint, modelParams_);

    // 根据 pluginHint 自动选择插件库
    std::string pluginPath = "./libyolo_plugin.so";  // 默认使用yolo_plugin（向后兼容）
    
    if (pluginHint.find("pose") != std::string::npos || 
        pluginHint.find("fall") != std::string::npos ||
        pluginHint.find("yolo11_pose") != std::string::npos) {
        pluginPath = "./libfall_plugin.so";
        ALOGN("[AIProcessor] Using fall detection plugin for model: %s", modelPath.c_str());
    } else if (pluginHint.find("helmet") != std::string::npos) {
        pluginPath = "./libhelmet_plugin.so";
        ALOGN("[AIProcessor] Using helmet detection plugin for model: %s", modelPath.c_str());
    } else if (pluginHint.find("fire") != std::string::npos || pluginHint.find("smoke") != std::string::npos) {
        pluginPath = "./libsmoke_fire_plugin.so";
        ALOGN("[AIProcessor] Using smoke/fire detection plugin for model: %s", modelPath.c_str());
    } else if (pluginHint.find("plate") != std::string::npos) {
        pluginPath = "./libplate_detection_plugin.so";
        ALOGN("[AIProcessor] Using plate detection plugin for model: %s", modelPath.c_str());
    } else if (pluginHint.find("crowd") != std::string::npos || pluginHint.find("human_group") != std::string::npos) {
        pluginPath = "./libcrowd_plugin.so";
        ALOGN("[AIProcessor] Using crowd (person aggregation) plugin for model: %s", modelPath.c_str());
    } else if (pluginHint.find("face_rec") != std::string::npos ||
               pluginHint.find("arcface") != std::string::npos ||
               pluginHint.find("recognition") != std::string::npos) {
        pluginPath = "./libface_recognition_plugin.so";
        ALOGN("[AIProcessor] Using face recognition plugin for model: %s", modelPath.c_str());
    } else if (pluginHint.find("face") != std::string::npos) {
        pluginPath = "./libface_detection_plugin.so";
        ALOGN("[AIProcessor] Using face detection plugin for model: %s", modelPath.c_str());
    } else if (pluginHint.find("human") != std::string::npos) {
        pluginPath = "./libhuman_detection_plugin.so";
        ALOGN("[AIProcessor] Using human detection plugin for model: %s", modelPath.c_str());
    } else if (modelPath.find("behavior") != std::string::npos) {
        pluginPath = "./libbehavior_plugin.so";
        ALOGN("[AIProcessor] Using behavior detection plugin for model: %s", modelPath.c_str());
    } else {
        pluginPath = "./libhelmet_plugin.so";
        ALOGN("[AIProcessor] Using manhole-cover plugin for model: %s", modelPath.c_str());
    }

    // 檢查模型文件是否存在（人員聚集插件會自行 fallback 到人員偵測模型，故跳過檢查）
    // Configuration takes precedence; keep the manhole plugin as the default.
    if (modelParams_.contains("plugin") && modelParams_["plugin"].is_string() &&
        !modelParams_["plugin"].get<std::string>().empty()) {
        pluginPath = modelParams_["plugin"].get<std::string>();
    } else {
        pluginPath = "./libmanhole_plugin.so";
    }
    ALOGN("[AIProcessor] Loading plugin: %s", pluginPath.c_str());
    const bool isCrowdPlugin = false;
    if (!isCrowdPlugin) {
        std::ifstream test_file(modelPath);
        if (!test_file.good()) {
            ALOGE("Model file does not exist: %s", modelPath.c_str());
            ALOGE("Please check if the model file exists and the path is correct.");
            return false;
        }
        test_file.close();
    }

    // 加载插件库（动态库）
    dlerror();
    void* handle = dlopen(pluginPath.c_str(), RTLD_LAZY);
    if (!handle) {
        ALOGE("dlopen failed for %s: %s", pluginPath.c_str(), dlerror());
        // 如果指定的插件加载失败，尝试使用默认插件
        if (pluginPath != "./libyolo_plugin.so") {
            ALOGW("Trying fallback to ./bin/libmanhole_plugin.so");
            handle = dlopen("./bin/libmanhole_plugin.so", RTLD_LAZY);
            if (!handle) {
                ALOGE("Fallback dlopen also failed: %s", dlerror());
                return false;
            }
        } else {
            return false;
        }
    }

    // 获取创建模型实例的函数指针
    CreateAIModelFunc create = (CreateAIModelFunc)dlsym(handle, "CreateAIModel");
    if (!create) {
        ALOGE("dlsym CreateAIModel failed: %s", dlerror());
        dlclose(handle);
        return false;
    }

    // 创建模型实例并初始化
    ALOGN("[AIProcessor] Creating model instance...");
    IAIModel* newModel = create();
    if (!newModel) {
        ALOGE("Failed to create model instance");
        dlclose(handle);
        return false;
    }
    
    ALOGN("[AIProcessor] Initializing model: %s", modelPath.c_str());
    int initRet = newModel->Init(modelPath.c_str());
    if (initRet != 0) {
        ALOGE("Model Init failed: %s, ret=%d", modelPath.c_str(), initRet);
        // 初始化失败时销毁模型实例
        DestroyAIModelFunc destroy = (DestroyAIModelFunc)dlsym(handle, "DestroyAIModel");
        if (destroy) destroy(newModel);
        dlclose(handle);
        return false;
    }
    
    ALOGN("[AIProcessor] Model initialized successfully");

    // 保存模型实例和插件句柄
    model_ = newModel;
    pluginHandle_ = handle;
    modelPath_ = modelPath;
    if (!modelName.empty()) modelName_ = modelName;

    // 获取模型输入尺寸（注意：此時已經持有 modelMutex_ 鎖，所以不能調用 getInputSize）
    // getInputSize 內部也會嘗試獲取鎖，會導致死鎖
    // 直接從 model_ 獲取輸入尺寸
    int w = 640, h = 640;
    if (model_ && newModel) {
        model_->GetInputSize(&w, &h);
    }
    ALOGN("AI Model Loaded: %s, %dx%d", modelPath.c_str(), w, h);

    return true;
}

void AIProcessor::applyModelParamsToEnv(const std::string& modelHint, const nlohmann::json& modelParams) {
    if (!modelParams.is_object()) return;
    const auto lowerHint = [&]() {
        std::string s = modelHint;
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(::tolower(c)); });
        return s;
    }();
    const bool isFaceDet = lowerHint.find("face_detector") != std::string::npos ||
                           (lowerHint.find("face") != std::string::npos &&
                            lowerHint.find("arcface") == std::string::npos &&
                            lowerHint.find("recognition") == std::string::npos);
    const bool isFaceRec = lowerHint.find("arcface") != std::string::npos ||
                           lowerHint.find("face_rec") != std::string::npos ||
                           lowerHint.find("recognition") != std::string::npos;

    auto setFloat = [&](const char* key, const char* envKey) {
        if (!modelParams.contains(key)) return;
        try {
            const float v = modelParams[key].get<float>();
            std::string sv = std::to_string(v);
            setenv(envKey, sv.c_str(), 1);
            ALOGN("[AIProcessor] Applied model param %s=%.4f -> %s", key, v, envKey);
        } catch (...) {
            ALOGW("[AIProcessor] Ignore invalid float param: %s", key);
        }
    };
    auto setInt = [&](const char* key, const char* envKey) {
        if (!modelParams.contains(key)) return;
        try {
            const int v = modelParams[key].get<int>();
            std::string sv = std::to_string(v);
            setenv(envKey, sv.c_str(), 1);
            ALOGN("[AIProcessor] Applied model param %s=%d -> %s", key, v, envKey);
        } catch (...) {
            ALOGW("[AIProcessor] Ignore invalid int param: %s", key);
        }
    };

    // 通用 conf/nms（大多數 YOLO 類插件可直接讀這組）
    setFloat("conf_threshold", "MODEL_CONF_THRESH");
    setFloat("nms_threshold", "MODEL_NMS_THRESH");

    if (isFaceDet) {
        setFloat("conf_threshold", "FACE_DET_CONF");
        setFloat("nms_threshold", "FACE_DET_NMS");
    }
    if (isFaceRec) {
        setFloat("face_rec_threshold", "FACE_REC_THRESHOLD");
        setFloat("face_rec_margin", "FACE_REC_MARGIN");
        setFloat("face_rec_min_quality", "FACE_REC_MIN_QUALITY");
        setInt("face_rec_min_face_size", "FACE_REC_MIN_FACE_SIZE");
        setInt("face_rec_skip_margin", "FACE_REC_SKIP_MARGIN");
        setInt("face_rec_debug", "FACE_REC_DEBUG");
    }

    // 各插件專用（避免同進程多模型時互相覆蓋）
    if (lowerHint.find("helmet") != std::string::npos) {
        setFloat("conf_threshold", "HELMET_CONF_THRESH");
        setFloat("nms_threshold", "HELMET_NMS_THRESH");
    } else if (lowerHint.find("human") != std::string::npos &&
               lowerHint.find("crowd") == std::string::npos &&
               lowerHint.find("group") == std::string::npos) {
        setFloat("conf_threshold", "HUMAN_CONF_THRESH");
        setFloat("nms_threshold", "HUMAN_NMS_THRESH");
    } else if (lowerHint.find("crowd") != std::string::npos || lowerHint.find("human_group") != std::string::npos) {
        setFloat("conf_threshold", "CROWD_CONF_THRESH");
        setFloat("nms_threshold", "CROWD_NMS_THRESH");
    } else if (lowerHint.find("smoke") != std::string::npos || lowerHint.find("fire") != std::string::npos) {
        setFloat("conf_threshold", "SMOKE_FIRE_CONF_THRESH");
        setFloat("nms_threshold", "SMOKE_FIRE_NMS_THRESH");
    } else if (lowerHint.find("plate") != std::string::npos) {
        setFloat("conf_threshold", "PLATE_CONF_THRESH");
        setFloat("nms_threshold", "PLATE_NMS_THRESH");
    } else if (lowerHint.find("behavior") != std::string::npos) {
        setFloat("conf_threshold", "BEHAVIOR_CONF_THRESH");
        setFloat("nms_threshold", "BEHAVIOR_NMS_THRESH");
    } else if (lowerHint.find("construction") != std::string::npos || lowerHint.find("site") != std::string::npos) {
        setFloat("conf_threshold", "CONSTRUCTION_CONF_THRESH");
        setFloat("nms_threshold", "CONSTRUCTION_NMS_THRESH");
    } else if (lowerHint.find("fall") != std::string::npos || lowerHint.find("pose") != std::string::npos) {
        setFloat("conf_threshold", "FALL_CONF_THRESH");
        setFloat("nms_threshold", "FALL_NMS_THRESH");
    }
}

// 卸载AI模型和插件库，释放资源
void AIProcessor::unloadModel() {
    std::lock_guard<std::mutex> lock(modelMutex_);
    
    if (model_) {
        // 先等待一小段时间，确保没有正在执行的推理
        usleep(50 * 1000);  // 50ms
        
        model_->Deinit(); // 模型反初始化

        if (pluginHandle_) {
            // 调用插件中的销毁函数释放模型实例
            DestroyAIModelFunc destroy = (DestroyAIModelFunc)dlsym(pluginHandle_, "DestroyAIModel");
            if (destroy) destroy(model_);
        }

        model_ = nullptr;
    }

    if (pluginHandle_) {
        dlclose(pluginHandle_); // 卸载动态库
        pluginHandle_ = nullptr;
    }

    ALOGN("AI Model Unloaded");
}

// 对输入帧进行推理处理
bool AIProcessor::processFrame(const AX_VIDEO_FRAME_T* frame, AI_RESULT_T* result) {
    if (!frame || !result) return false;
    
    // 线程安全地获取模型指针
    IAIModel* currentModel = nullptr;
    {
        std::lock_guard<std::mutex> lock(modelMutex_);
        currentModel = model_;
        if (!currentModel) {
            return false;
        }
    }
    
    // 注意：这里不持有锁，让卸载过程可以进行
    // 但我们已经保存了模型指针，只要在推理完成前不释放就可以
    
    // 处理内存映射
    AX_BOOL bMapped = AX_FALSE;
    AX_VIDEO_FRAME_T tFrame = *frame;
    
    if (!tFrame.u64VirAddr[0] && tFrame.u64PhyAddr[0]) {
        tFrame.u64VirAddr[0] = (AX_U64)AX_SYS_Mmap((AX_U64)tFrame.u64PhyAddr[0], tFrame.u32FrameSize);
        bMapped = AX_TRUE;
    }

    int ret = currentModel->Inference(&tFrame, result);
    
    if (bMapped && tFrame.u64VirAddr[0]) {
        AX_SYS_Munmap((void*)tFrame.u64VirAddr[0], tFrame.u32FrameSize);
    }
    
    return (ret == 0);
}

// 设置置信度和NMS阈值
void AIProcessor::setThresholds(float conf, float nms) {
    std::lock_guard<std::mutex> lock(modelMutex_);
    confThreshold_ = conf;
    nmsThreshold_ = nms;
    ALOGD("AI thresholds updated: conf=%.2f, nms=%.2f", conf, nms);
}

// 获取模型输入尺寸
void AIProcessor::getInputSize(int* w, int* h) const {
    // [Optim] 移除函数入口/锁获取等高频 Debug 日志，这些日志在每帧渲染时会刷屏影响性能
    std::lock_guard<std::mutex> lock(modelMutex_);
    if (model_ && w && h) {
        model_->GetInputSize(w, h);
    } else {
        // 仅在异常情况打印警告，且限制频率（简单的频率控制）
        static int warn_count = 0;
        if (warn_count++ % 1000 == 0) {
             ALOGW("[AIProcessor] Model not available when getting input size, defaulting to 640x640");
        }
        if (w) *w = 640;
        if (h) *h = 640;
    }
}

// 获取当前模型路径
std::string AIProcessor::getModelPath() const { 
    std::lock_guard<std::mutex> lock(modelMutex_);
    return modelPath_; 
}

// 獲取 OSD 渲染器（每個模型可以有自己的渲染器）
std::shared_ptr<IOSDRenderer> AIProcessor::getOSDRenderer() {
    std::lock_guard<std::mutex> lock(modelMutex_);
    return nullptr;
#if 0
    // 如果已經有渲染器，直接返回
    if (osdRenderer_) {
        return osdRenderer_;
    }
    
    // 根據模型路徑或 modelName 創建對應的渲染器（人員聚集與人員偵測共用 path，用 modelName_ 區分）
    const std::string& osdHint = modelName_.empty() ? modelPath_ : modelName_;
    if (osdHint.find("crowd") != std::string::npos || osdHint.find("human_group") != std::string::npos) {
        // 人員聚集：群組框 + 成員數
        osdRenderer_ = std::make_shared<CrowdDetectionOSDRenderer>();
    } else if (modelPath_.find("fall") != std::string::npos || 
        modelPath_.find("pose") != std::string::npos) {
        // 跌倒檢測模型使用專門的渲染器（支持骨架繪製）
        osdRenderer_ = std::make_shared<FallDetectionOSDRenderer>();
    } else if (modelPath_.find("helmet") != std::string::npos) {
        // 安全帽檢測模型使用專門的渲染器（根據類別使用不同顏色）
        osdRenderer_ = std::make_shared<HelmetDetectionOSDRenderer>();
    } else if (modelPath_.find("fire") != std::string::npos || modelPath_.find("smoke") != std::string::npos) {
        // 煙火檢測模型使用專門的渲染器
        osdRenderer_ = std::make_shared<SmokeFireDetectionOSDRenderer>();
    } else if (modelPath_.find("behavior") != std::string::npos) {
        // 行為檢測模型使用專門的渲染器
        osdRenderer_ = std::make_shared<BehaviorDetectionOSDRenderer>();
    } else if (osdHint.find("face_rec") != std::string::npos ||
               osdHint.find("arcface") != std::string::npos ||
               osdHint.find("recognition") != std::string::npos) {
        osdRenderer_ = std::make_shared<FaceRecognitionOSDRenderer>(modelParams_);
    } else if (osdHint.find("face") != std::string::npos) {
        osdRenderer_ = std::make_shared<FaceDetectionOSDRenderer>();
    } else if (osdHint.find("human") != std::string::npos) {
        // 人員偵測：每個目標框 + 額外顯示人數
        osdRenderer_ = std::make_shared<HumanDetectionOSDRenderer>();
    }
    
    // 其他模型返回 nullptr，表示使用默認渲染器
    return osdRenderer_;
#endif
}

