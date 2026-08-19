#ifndef VIDEO_STREAM_H
#define VIDEO_STREAM_H

// 档位2 版视频流。与 AX650 的 include/manager/video_stream.h 同构；
// 每路输入拆两条流（与 AX650 一致）：
//   主码流（isMainStream = isMediaMTXOutput || isFileOutput）：
//     H264Demux -> MPP 解码(NV12) -> RGA 缩放/转BGR -> CPU 画框(OSD 降级)
//     -> RGA 转NV12 -> MPP 编码 -> ffmpeg -c copy RTSP(MediaMTX) / raw 文件
//   AI 流（isAIStream = enableAI && !isMainStream）：
//     FrameBroker 取 NV12 -> RGA 640x640 -> RKNN 插件 -> SharedAIResult
// 对应硬件：VDEC=MPP mpi_dec、IVPS=RGA、VENC=MPP mpi_enc（IVPS OSD region 降级为 CPU 画框）

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <cstdio>

#include "ai_interface.h"
#include "ai_pipeline_config.h"
#include "ai_processor.h"
#include "osd_renderer.h"
#include "inference_manager.h"
#include "rk_media.h"
#include "h264_demux.h"
#include "frame_broker.h"

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
    bool isEnded() const { return ended_.load(); }  // 输入 EOF / 出错

    // Configuration updates
    void updateConfig(const StreamConfig& newConfig);
    void setAIEnabled(bool enable);
    void clearOSD();

    // Frame processing interface（AI 流单帧推理，兼容接口）
    void processFrame(const AI_FRAME_T* frame, AI_RESULT_T* result);

    void setAIProcessor(std::unique_ptr<AIProcessor> processor);
    void setAIProcessors(std::vector<std::unique_ptr<AIProcessor>> processors);

    // 主码流 <-> AI 流共享（由 VideoStreamManager 装配）
    void attachFrameBroker(std::shared_ptr<FrameBroker> broker) { broker_ = std::move(broker); }
    void attachAIResult(std::shared_ptr<SharedAIResult> result) { aiResult_ = std::move(result); }

    // Accessors（与 ax650 同名）
    int getStreamId() const { return config_.streamId; }
    bool isAIEnabled() const { return config_.enableAI; }
    bool isMainStream() const { return config_.isMediaMTXOutput || config_.isFileOutput; }
    bool isAIStream() const { return config_.enableAI && !isMainStream(); }
    std::shared_ptr<AIProcessor> getAIProcessor() const {
        std::lock_guard<std::mutex> lock(stateMutex_);
        return aiProcessors_.empty() ? nullptr : aiProcessors_[0];
    }
    bool hasMultipleProcessors() const {
        std::lock_guard<std::mutex> lock(stateMutex_);
        return aiProcessors_.size() > 1u;
    }
    std::string getModelPath() const;
    std::vector<std::string> getModelPaths() const;
    std::string getModelName() const { return config_.modelName; }
    bool isCommandLineModel() const { return config_.isCommandLineModel; }
    void setCommandLineModelFlag(bool flag) { config_.isCommandLineModel = flag; }
    std::string getInputSource() const { return config_.inputSource; }
    bool isMediaMTXOutput() const { return config_.isMediaMTXOutput; }
    bool isFileOutput() const { return config_.isFileOutput; }
    std::string getOutputFilePath() const { return config_.outputFilePath; }
    int getFps() const { return config_.fps; }
    void setThresholds(float conf, float nms);
    void setModelPath(const std::string& path);
    void setModelName(const std::string& name);
    void setModelStages(const std::vector<ModelStageConfig>& stages, AIPipelineMode mode);

private:
    // 主码流线程：demux -> decoder -> onDecodedFrame -> rga/osd/enc -> rtp/文件
    void mainLoop();
    // AI 流线程：broker -> rga 640 -> plugin -> SharedAIResult
    void aiLoop();

    void onDecodedFrame(const RkNv12Frame& frame);
    void pushToOutput(const uint8_t* data, size_t size);
    void openOutput();
    void closeOutput();
    void rebuildProcessors(const std::vector<ModelStageConfig>& stages,
                           const std::string& singlePath, const std::string& singleName);

    StreamConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> ended_{false};
    std::thread workerThread_;

    std::vector<std::shared_ptr<AIProcessor>> aiProcessors_;
    std::unique_ptr<InferenceManager> inferenceManager_;
    std::unique_ptr<OSDRenderer> osdRenderer_;

    // 共享（manager 装配）
    std::shared_ptr<FrameBroker> broker_;
    std::shared_ptr<SharedAIResult> aiResult_;

    // 主码流硬件资源（档位2：MPP/RGA）
    std::unique_ptr<RkDecoder> decoder_;
    std::unique_ptr<RkEncoder> encoder_;
    std::unique_ptr<H264Demux> demux_;
    FILE* fileOut_ = nullptr;      // offline：raw H.264 文件
    FILE* rtspPipe_ = nullptr;     // stream：ffmpeg RTSP 子进程 stdin（-c copy）

    // 中间缓冲（主码流：NV12->BGR->画框->NV12；AI 流：NV12->640 BGR）
    std::vector<uint8_t> nv12Tmp_;   // resize 中间帧
    std::vector<uint8_t> bgrBuf_;    // 画框缓冲（输出尺寸 BGR）
    std::vector<uint8_t> nv12Out_;   // 编码输入（输出尺寸 NV12）
    std::vector<uint8_t> aiBgrBuf_;  // AI 支路 640x640 BGR
    int outStride_ = 0;
    bool encoderReady_ = false;

    mutable std::mutex stateMutex_;
    std::atomic<long> frameCount_{0};
};

#endif // VIDEO_STREAM_H
