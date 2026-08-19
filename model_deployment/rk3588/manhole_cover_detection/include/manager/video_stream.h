#ifndef VIDEO_STREAM_H
#define VIDEO_STREAM_H

// RK3588 版视频流。与 AX650 的 include/manager/video_stream.h 同构：
// 方法名/成员名一致（start/stop/updateConfig/setAIProcessors/osdRenderer_/
// inferenceManager_ 等）；仅底层不同——AX650 走 VDEC/IVPS/VENC 硬件流水线，
// RK3588 用 OpenCV VideoCapture + FFmpeg h264_rkmpp。

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "../ai_interface.h"
#include "ai_pipeline_config.h"
#include "ai_processor.h"
#include "osd_renderer.h"
#include "inference_manager.h"

class VideoStream {
public:
    explicit VideoStream(const StreamConfig& config);
    ~VideoStream();

    VideoStream(const VideoStream&) = delete;
    VideoStream& operator=(const VideoStream&) = delete;

    // Lifecycle management
    bool start();
    void stop();
    bool isRunning() const { return running_.load(); }
    bool isEnded() const { return ended_.load(); }  // 离线输入读到 EOF

    // Configuration updates
    void updateConfig(const StreamConfig& newConfig);
    void setAIEnabled(bool enable);
    void clearOSD();

    // Frame processing interface
    void processFrame(const AI_FRAME_T* frame, AI_RESULT_T* result);

    void setAIProcessor(std::unique_ptr<AIProcessor> processor);
    void setAIProcessors(std::vector<std::unique_ptr<AIProcessor>> processors);

    // Accessors
    int getStreamId() const { return config_.streamId; }
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
    bool isCommandLineModel() const { return config_.isCommandLineModel; }
    void setCommandLineModelFlag(bool flag) { config_.isCommandLineModel = flag; }
    std::string getInputSource() const { return config_.inputSource; }
    bool isMediaMTXOutput() const { return config_.isMediaMTXOutput; }
    bool isFileOutput() const { return config_.isFileOutput; }
    void setThresholds(float conf, float nms);
    void setModelPath(const std::string& path);
    void setModelName(const std::string& name);
    void setModelStages(const std::vector<ModelStageConfig>& stages, AIPipelineMode mode);

private:
    void runLoop();
    bool openOutput();
    void closeOutput();
    void rebuildProcessors(const std::vector<ModelStageConfig>& stages,
                           const std::string& singlePath, const std::string& singleName);

    StreamConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> ended_{false};
    std::thread workerThread_;

    std::vector<std::shared_ptr<AIProcessor>> aiProcessors_;  // 並行多模型或串行階段
    std::unique_ptr<InferenceManager> inferenceManager_;      // 任務圖 / 流水線推理調度
    std::unique_ptr<OSDRenderer> osdRenderer_;                // Per-stream OSD

    cv::VideoCapture capture_;
    cv::VideoWriter writer_;
    FILE* streamPipe_ = nullptr;  // ffmpeg RTSP 管道

    mutable std::mutex stateMutex_;
    std::atomic<long> frameCount_{0};
};

#endif // VIDEO_STREAM_H
