#ifndef VIDEO_STREAM_H
#define VIDEO_STREAM_H

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <functional>
#include <unistd.h>
#include "common_pipeline.h"
#include "../ai_interface.h"
#include "osd_renderer.h"
#include "ai_pipeline_config.h"

// Forward declarations
class AIProcessor;
class ConfigService;
class VideoStreamManager;  // 前向聲明，避免循環依賴
class InferenceManager;

// Stream configuration parameters
struct StreamConfig {
    int streamId;
    std::string inputSource;
    // 輸入編碼：h264 / h265 / auto（預設）。auto 會先試 h264，連續失敗後自動切到 h265。
    std::string inputCodec = "auto";
    bool enableAI = false;
    std::string modelPath;  // 單模型時使用；多模型時由 modelStages 覆蓋
    std::string modelName;  // 單模型時使用
    bool isCommandLineModel = false;  // 標記模型是否由命令行指定（不應被配置文件覆蓋）
    float confThreshold = 0.45f;
    float nmsThreshold = 0.45f;
    int ivpsGroup;
    int vdecGroup;
    int outputWidth = 1920;
    int outputHeight = 1080;
    int fps = 30;
    std::string rtspEndpoint = "axstream";
    bool isRTSPOutput = false;
    bool isMediaMTXOutput = false;
    std::string mediamtxEndpoint = "127.0.0.1:8000";
    // （預留）可在需要时对 MediaMTX VENC channel 做映射/覆盖。
    // 多模型：同一流並行多模型或串行階段
    AIPipelineMode aiPipelineMode = AIPipelineMode::Parallel;
    std::vector<ModelStageConfig> modelStages;  // 非空時優先於單一 modelPath/modelName
};

class VideoStream {
public:
    explicit VideoStream(const StreamConfig& config);
    ~VideoStream();
    
    // 禁止複製，允許移動
    VideoStream(const VideoStream&) = delete;
    VideoStream& operator=(const VideoStream&) = delete;
    VideoStream(VideoStream&&) noexcept;
    VideoStream& operator=(VideoStream&&) noexcept;

    // Lifecycle management
    bool start();
    void stop();
    bool isRunning() const;

    // Configuration updates
    void updateConfig(const StreamConfig& newConfig);
    void setAIEnabled(bool enable);
    void clearCommandLineModelFlag();  // 清除命令行模型標記，允許 Web 配置覆蓋
    void clearOSD();  // 清除 OSD 顯示
    
    // Frame processing interface
    void processFrame(pipeline_buffer_t* buffer);
    void setAIProcessor(std::unique_ptr<AIProcessor> processor);
    /** 設置多個 AI 處理器（並行多模型或串行階段）；空時等效於無模型 */
    void setAIProcessors(std::vector<std::unique_ptr<AIProcessor>> processors);

    void onCallbackEnter();
    void onCallbackExit();

    // Accessors
    int getStreamId() const { return config_.streamId; }
    int getIvpsGroup() const { return config_.ivpsGroup; }
    int getVdecGroup() const { return config_.vdecGroup; }
    bool isAIEnabled() const { return config_.enableAI; }
    /** 返回第一個 AI 處理器（OSD/輸入尺寸等兼容單模型邏輯）；多模型時取首個 */
    std::shared_ptr<AIProcessor> getAIProcessor() const { 
        std::lock_guard<std::mutex> lock(stateMutex_);
        return aiProcessors_.empty() ? nullptr : aiProcessors_[0]; 
    }
    /** 當前流是否使用多模型（並行或串行） */
    bool hasMultipleProcessors() const {
        std::lock_guard<std::mutex> lock(stateMutex_);
        return aiProcessors_.size() > 1u;
    }
    std::string getModelPath() const;
    /** 返回當前流所有模型路徑（多模型時為 modelStages 的路徑列表，單模型時為單一元素），用於 OSD 渲染器選擇 */
    std::vector<std::string> getModelPaths() const;
    std::string getModelName() const { return config_.modelName; }
    bool isCommandLineModel() const { return config_.isCommandLineModel; }  // 檢查模型是否由命令行指定
    std::string getInputSource() const { return config_.inputSource; }
    bool isMediaMTXOutput() const { return config_.isMediaMTXOutput; }  // 檢查是否是推送到 MediaMTX 的流
    pipeline_t* getPipeline() { return &pipeline_; }  // 獲取pipeline指針，用於多流處理
    void setThresholds(float conf, float nms);
    /** 更新當前模型路徑（切換模型後必須調用，否則 getModelPath() 仍為舊值，導致後續切換被誤判為「無需重載」） */
    void setModelPath(const std::string& path);
    /** 更新當前模型名稱（與 setModelPath 搭配，用於 crowd vs human 共用 path 時區分插件） */
    void setModelName(const std::string& name);
    /** 更新多模型配置（modelStages 和 aiPipelineMode） */
    void setModelStages(const std::vector<ModelStageConfig>& stages, AIPipelineMode mode);

private:
    void resolveInputCodecConfig();
    bool tryAutoCodecFallbackLocked();

    // Pipeline management
    bool createProcessingPipeline();
    void destroyProcessingPipeline();
    void configureIVPS();
    void configureVDEC();
    void configureVENC();

    StreamConfig config_;
    pipeline_t pipeline_;
    bool running_ = false;
    
    std::vector<std::shared_ptr<AIProcessor>> aiProcessors_;  // 並行多模型或串行階段
    std::unique_ptr<OSDRenderer> osdRenderer_;  // Per-stream OSD
    class AIWorker;
    std::unique_ptr<AIWorker> aiWorker_;  // 每路 AI 固定 worker，避免每幀新建線程導致 ~10min 後斷流
    std::unique_ptr<InferenceManager> inferenceManager_;  // 任務圖 / 流水線推理調度

    mutable std::mutex stateMutex_;
    std::function<void(pipeline_buffer_t*)> outputCallback_;

    std::atomic<int> errorCount_{0};  // 添加错误计数器
    
    // 緩存模型輸入尺寸，避免每幀都調用 getInputSize
    int cachedInputWidth_ = 0;
    int cachedInputHeight_ = 0;
    
    // AI inference callback
    static void aiInferenceCallback(pipeline_buffer_t* buf);
    /** 將一幀提交給本流專用 worker 線程執行推理（避免每幀新建線程導致長時間運行後資源耗盡）。佇列滿則丟幀並返回 false */
    bool submitAIFrame(std::vector<uint8_t> frameData, uint32_t w, uint32_t h, uint32_t stridePix, uint32_t sz);

    // 允許輔助函數在非成員上下文中存取推理相關私有狀態
    friend bool runAIInference(VideoStream* stream, AX_VIDEO_FRAME_T* tFrame, AI_RESULT_T* stResult);
    friend void deliverWorkerInferenceResult(VideoStream* s, int streamId, bool processSuccess, AI_RESULT_T& stResult,
                                             double inferenceTimeMs);
    friend class AIWorker;

    // Thread safety for callback synchronization
    std::atomic<int> activeCallbacks_{0}; // Counter for active AI inference callbacks
    std::mutex stopMutex_;               // Mutex for the condition variable
    std::condition_variable stopCv_;     // CV to signal when all callbacks are done
    std::mutex pipelineMutex_;
    pipeline_input_e preferredInputType_ = pi_vdec_h264;
    bool autoCodecFallbackEnabled_ = true;
    bool autoCodecFallbackSwitched_ = false;
    int consecutiveInputErrors_ = 0;
    
public:
    // 設置全局 VideoStreamManager 指針（由 VideoStreamManager 調用）
    static void setGlobalStreamManager(VideoStreamManager* manager);

    void incrementErrorCount() {
        errorCount_.fetch_add(1, std::memory_order_relaxed);
    }
    
    int getErrorCount() const {
        return errorCount_.load(std::memory_order_relaxed);
    }
};

// Custom implementation of make_unique for C++11
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

#endif // VIDEO_STREAM_H

