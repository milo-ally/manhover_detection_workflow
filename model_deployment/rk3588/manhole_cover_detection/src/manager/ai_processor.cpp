#include "ai_processor.h"
#include "../osd_renderer_interface.h"
#include "../../utilities/sample_log.h"
#include <dlfcn.h>
#include <unistd.h>
#include <fstream>
#include <cstdlib>
#include <algorithm>

AIProcessor::AIProcessor(const std::string& modelPath, const std::string& modelName,
                         const nlohmann::json& modelParams) {
    if (!modelPath.empty()) {
        loadModel(modelPath, modelName, modelParams);
    }
}

AIProcessor::~AIProcessor() {
    unloadModel();
}

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
    const std::string& pluginHint = modelName_.empty() ? modelPath : modelName_;
    applyModelParamsToEnv(pluginHint, modelParams_);
    osdRenderer_.reset();

    // 插件路径：配置 plugin 字段优先，否则默认 ./libmanhole_plugin.so
    std::string pluginPath = "./libmanhole_plugin.so";
    if (modelParams_.contains("plugin") && modelParams_["plugin"].is_string() &&
        !modelParams_["plugin"].get<std::string>().empty()) {
        pluginPath = modelParams_["plugin"].get<std::string>();
    }
    ALOGN("[AIProcessor] Loading plugin: %s", pluginPath.c_str());

    // 检查模型文件是否存在
    {
        std::ifstream test_file(modelPath);
        if (!test_file.good()) {
            ALOGE("Model file does not exist: %s", modelPath.c_str());
            ALOGE("Please check if the model file exists and the path is correct.");
            return false;
        }
    }

    // 加载插件库（动态库）
    dlerror();
    void* handle = dlopen(pluginPath.c_str(), RTLD_LAZY);
    if (!handle) {
        ALOGE("dlopen failed for %s: %s", pluginPath.c_str(), dlerror());
        // 指定的插件加载失败时回退到 ./bin/libmanhole_plugin.so
        ALOGW("Trying fallback to ./bin/libmanhole_plugin.so");
        handle = dlopen("./bin/libmanhole_plugin.so", RTLD_LAZY);
        if (!handle) {
            ALOGE("Fallback dlopen also failed: %s", dlerror());
            return false;
        }
    }

    CreateAIModelFunc create = (CreateAIModelFunc)dlsym(handle, "CreateAIModel");
    if (!create) {
        ALOGE("dlsym CreateAIModel failed: %s", dlerror());
        dlclose(handle);
        return false;
    }

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
        DestroyAIModelFunc destroy = (DestroyAIModelFunc)dlsym(handle, "DestroyAIModel");
        if (destroy) destroy(newModel);
        dlclose(handle);
        return false;
    }
    ALOGN("[AIProcessor] Model initialized successfully");

    model_ = newModel;
    pluginHandle_ = handle;
    modelPath_ = modelPath;
    if (!modelName.empty()) modelName_ = modelName;

    int w = 640, h = 640;
    if (model_) {
        model_->GetInputSize(&w, &h);
    }
    ALOGN("AI Model Loaded: %s, %dx%d", modelPath.c_str(), w, h);
    return true;
}

void AIProcessor::applyModelParamsToEnv(const std::string& modelHint, const nlohmann::json& modelParams) {
    if (!modelParams.is_object()) return;
    const auto lowerHint = [&]() {
        std::string s = modelHint;
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(::tolower(c)); });
        return s;
    }();

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

    // 通用 conf/nms（插件可读 MODEL_CONF_THRESH / MODEL_NMS_THRESH）
    setFloat("conf_threshold", "MODEL_CONF_THRESH");
    setFloat("nms_threshold", "MODEL_NMS_THRESH");

    // 井盖插件专用环境变量（插件优先读 MANHOLE_*，未设置时回退 MODEL_*）
    if (lowerHint.find("manhole") != std::string::npos ||
        lowerHint.find("cover") != std::string::npos) {
        setFloat("conf_threshold", "MANHOLE_CONF_THRESH");
        setFloat("nms_threshold", "MANHOLE_NMS_THRESH");
    }
}

void AIProcessor::unloadModel() {
    std::lock_guard<std::mutex> lock(modelMutex_);
    if (model_) {
        usleep(50 * 1000);  // 50ms，确保没有正在执行的推理
        model_->Deinit();
        if (pluginHandle_) {
            DestroyAIModelFunc destroy = (DestroyAIModelFunc)dlsym(pluginHandle_, "DestroyAIModel");
            if (destroy) destroy(model_);
        }
        model_ = nullptr;
    }
    if (pluginHandle_) {
        dlclose(pluginHandle_);
        pluginHandle_ = nullptr;
    }
    ALOGN("AI Model Unloaded");
}

bool AIProcessor::processFrame(const AI_FRAME_T* frame, AI_RESULT_T* result) {
    if (!frame || !result) return false;
    IAIModel* currentModel = nullptr;
    {
        std::lock_guard<std::mutex> lock(modelMutex_);
        currentModel = model_;
        if (!currentModel) {
            return false;
        }
    }
    int ret = currentModel->Inference(frame, result);
    return (ret == 0);
}

void AIProcessor::setThresholds(float conf, float nms) {
    std::lock_guard<std::mutex> lock(modelMutex_);
    confThreshold_ = conf;
    nmsThreshold_ = nms;
    ALOGD("AI thresholds updated: conf=%.2f, nms=%.2f", conf, nms);
}

void AIProcessor::getInputSize(int* w, int* h) const {
    std::lock_guard<std::mutex> lock(modelMutex_);
    if (model_ && w && h) {
        model_->GetInputSize(w, h);
    } else {
        if (w) *w = 640;
        if (h) *h = 640;
    }
}

std::string AIProcessor::getModelPath() const {
    std::lock_guard<std::mutex> lock(modelMutex_);
    return modelPath_;
}

std::shared_ptr<IOSDRenderer> AIProcessor::getOSDRenderer() {
    std::lock_guard<std::mutex> lock(modelMutex_);
    // 当前插件不提供专用渲染器，返回 nullptr -> 使用默认渲染器
    return osdRenderer_;
}
