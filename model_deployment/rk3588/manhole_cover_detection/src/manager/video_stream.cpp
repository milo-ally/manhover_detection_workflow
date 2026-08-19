#include "video_stream.h"
#include "../../utilities/sample_log.h"

#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <cmath>

namespace {

bool is_rtsp_url(const std::string& value) {
    return value.rfind("rtsp://", 0) == 0 || value.rfind("rtsps://", 0) == 0;
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') quoted += "'\\''";
        else quoted += c;
    }
    return quoted + "'";
}

}  // namespace

VideoStream::VideoStream(const StreamConfig& config) : config_(config) {
    osdRenderer_ = std::make_unique<OSDRenderer>();
    inferenceManager_ = std::make_unique<InferenceManager>();
}

VideoStream::~VideoStream() {
    stop();
}

bool VideoStream::start() {
    if (running_.load()) return true;
    ended_.store(false);

    // 打开输入：RTSP 走 TCP（SSH 隧道只转发 TCP），本地文件直接打开
    if (is_rtsp_url(config_.inputSource)) {
        setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 1);
        if (!capture_.open(config_.inputSource, cv::CAP_FFMPEG)) {
            ALOGE("[VideoStream] Stream %d: cannot open RTSP input: %s",
                  config_.streamId, config_.inputSource.c_str());
            return false;
        }
    } else {
        if (!capture_.open(config_.inputSource)) {
            ALOGE("[VideoStream] Stream %d: cannot open file input: %s",
                  config_.streamId, config_.inputSource.c_str());
            return false;
        }
    }

    // 创建 AI 处理器（启用 AI 时），支持多模型阶段
    if (config_.enableAI) {
        if (!config_.modelStages.empty()) {
            rebuildProcessors(config_.modelStages, "", "");
        } else {
            rebuildProcessors({}, config_.modelPath, config_.modelName);
        }
        if (aiProcessors_.empty()) {
            ALOGE("[VideoStream] Stream %d: no AI model loaded", config_.streamId);
            return false;
        }
        inferenceManager_->configure(aiProcessors_, config_.modelStages, config_.aiPipelineMode);
        osdRenderer_->init();
        ALOGN("[VideoStream] Stream %d: %zu AI processor(s) ready",
              config_.streamId, aiProcessors_.size());
    }

    if (!openOutput()) {
        ALOGE("[VideoStream] Stream %d: cannot open output", config_.streamId);
        return false;
    }

    running_.store(true);
    workerThread_ = std::thread(&VideoStream::runLoop, this);
    ALOGN("[VideoStream] Stream %d started: input=%s, ai=%d",
          config_.streamId, config_.inputSource.c_str(), config_.enableAI ? 1 : 0);
    return true;
}

void VideoStream::rebuildProcessors(const std::vector<ModelStageConfig>& stages,
                                    const std::string& singlePath, const std::string& singleName) {
    std::vector<std::unique_ptr<AIProcessor>> newProcessors;
    auto build = [&](const std::string& path, const std::string& name, const std::string& plugin,
                     float conf, float nms) {
        nlohmann::json params = nlohmann::json::object();
        params["conf_threshold"] = conf;
        params["nms_threshold"] = nms;
        if (!plugin.empty()) params["plugin"] = plugin;
        auto proc = std::make_unique<AIProcessor>(path, name, params);
        if (!proc->isModelLoaded()) {
            ALOGE("[VideoStream] Stream %d: model failed to load: %s",
                  config_.streamId, path.c_str());
            return;
        }
        newProcessors.push_back(std::move(proc));
    };

    if (!stages.empty()) {
        for (const auto& stage : stages) {
            std::string plugin = !stage.pluginPath.empty() ? stage.pluginPath : config_.pluginPath;
            build(stage.modelPath, stage.modelName, plugin, stage.confThreshold, stage.nmsThreshold);
        }
    } else if (!singlePath.empty()) {
        build(singlePath, singleName, config_.pluginPath, config_.confThreshold, config_.nmsThreshold);
    }

    std::lock_guard<std::mutex> lock(stateMutex_);
    aiProcessors_.clear();
    for (auto& p : newProcessors) {
        aiProcessors_.push_back(std::move(p));
    }
}

bool VideoStream::openOutput() {
    const double fps = capture_.get(cv::CAP_PROP_FPS);
    const int width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double effFps = (fps > 0.0 && std::isfinite(fps)) ? fps : 25.0;

    if (config_.isFileOutput && !config_.outputFilePath.empty()) {
        writer_.open(config_.outputFilePath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                     effFps, cv::Size(width, height));
        if (!writer_.isOpened()) {
            ALOGE("[VideoStream] Stream %d: cannot open output video %s (check codec support)",
                  config_.streamId, config_.outputFilePath.c_str());
            return false;
        }
        ALOGN("[VideoStream] Stream %d: output_mode=file %s (%dx%d, %.2f fps)",
              config_.streamId, config_.outputFilePath.c_str(), width, height, effFps);
    } else if (is_rtsp_url(config_.rtspOutputUrl)) {
        std::ostringstream command;
        command << "ffmpeg -loglevel warning -f rawvideo -pix_fmt bgr24"
                << " -s " << width << "x" << height
                << " -r " << effFps << " -i pipe:0 -an"
                << " -c:v h264_rkmpp -pix_fmt yuv420p -f rtsp"
                << " -rtsp_transport tcp " << shell_quote(config_.rtspOutputUrl);
        ALOGN("[VideoStream] Stream %d: output_mode=rtsp command=%s",
              config_.streamId, command.str().c_str());
        streamPipe_ = popen(command.str().c_str(), "w");
        if (!streamPipe_) {
            ALOGE("[VideoStream] Stream %d: cannot start ffmpeg RTSP output: %s",
                  config_.streamId, config_.rtspOutputUrl.c_str());
            return false;
        }
    } else {
        ALOGE("[VideoStream] Stream %d: no valid output configured", config_.streamId);
        return false;
    }
    return true;
}

void VideoStream::closeOutput() {
    if (streamPipe_) {
        pclose(streamPipe_);
        streamPipe_ = nullptr;
    }
    if (writer_.isOpened()) {
        writer_.release();
    }
}

void VideoStream::runLoop() {
    cv::Mat frame;
    while (running_.load()) {
        if (!capture_.read(frame)) {
            ALOGN("[VideoStream] Stream %d: input ended or failed: %s",
                  config_.streamId, config_.inputSource.c_str());
            ended_.store(true);
            break;
        }

        if (config_.enableAI && !aiProcessors_.empty()) {
            AI_RESULT_T result;
            memset(&result, 0, sizeof(result));
            AI_FRAME_T aiFrame;
            aiFrame.data = frame.data;
            aiFrame.width = frame.cols;
            aiFrame.height = frame.rows;
            aiFrame.stride = static_cast<int>(frame.step);
            aiFrame.format = AI_FRAME_FORMAT_BGR24;

            bool ok = false;
            if (hasMultipleProcessors()) {
                ok = inferenceManager_->run(&aiFrame, &result);
            } else if (aiProcessors_[0]) {
                ok = aiProcessors_[0]->processFrame(&aiFrame, &result);
            }
            if (ok && result.nObjSize > 0 && osdRenderer_) {
                osdRenderer_->update(&result, frame.cols, frame.rows, frame.cols, frame.rows, frame);
            }
        }

        // 输出：RTSP 管道或文件
        if (streamPipe_) {
            if (!frame.isContinuous()) {
                frame = frame.clone();
            }
            const size_t bytes = frame.total() * frame.elemSize();
            if (fwrite(frame.data, 1, bytes, streamPipe_) != bytes) {
                ALOGE("[VideoStream] Stream %d: ffmpeg RTSP output pipe closed",
                      config_.streamId);
                ended_.store(true);
                break;
            }
            fflush(streamPipe_);
        } else if (writer_.isOpened()) {
            writer_.write(frame);
        }

        const long count = frameCount_.fetch_add(1) + 1;
        if (count % 100 == 1) {
            ALOGN("[VideoStream] Stream %d: frame=%ld", config_.streamId, count);
        }
    }
    ALOGN("[VideoStream] Stream %d: worker exited, frames=%ld",
          config_.streamId, frameCount_.load());
}

void VideoStream::stop() {
    if (!running_.load() && !workerThread_.joinable()) return;
    running_.store(false);
    if (workerThread_.joinable()) {
        workerThread_.join();
    }
    closeOutput();
    if (capture_.isOpened()) {
        capture_.release();
    }
    std::lock_guard<std::mutex> lock(stateMutex_);
    aiProcessors_.clear();
}

void VideoStream::updateConfig(const StreamConfig& newConfig) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    const bool inputChanged = (newConfig.inputSource != config_.inputSource);
    config_ = newConfig;
    if (inputChanged) {
        ALOGW("[VideoStream] Stream %d: input source changed, restart required", config_.streamId);
    }
}

void VideoStream::setAIEnabled(bool enable) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.enableAI = enable;
}

void VideoStream::clearOSD() {
    if (osdRenderer_) {
        osdRenderer_->clear();
    }
}

void VideoStream::processFrame(const AI_FRAME_T* frame, AI_RESULT_T* result) {
    if (!frame || !result || aiProcessors_.empty()) return;
    if (hasMultipleProcessors()) {
        inferenceManager_->run(const_cast<AI_FRAME_T*>(frame), result);
    } else if (aiProcessors_[0]) {
        aiProcessors_[0]->processFrame(frame, result);
    }
}

void VideoStream::setAIProcessor(std::unique_ptr<AIProcessor> processor) {
    std::vector<std::unique_ptr<AIProcessor>> vec;
    if (processor) vec.push_back(std::move(processor));
    setAIProcessors(std::move(vec));
}

void VideoStream::setAIProcessors(std::vector<std::unique_ptr<AIProcessor>> processors) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    aiProcessors_.clear();
    for (auto& p : processors) {
        aiProcessors_.push_back(std::move(p));
    }
    if (aiProcessors_.size() > 1) {
        config_.aiPipelineMode = AIPipelineMode::Parallel;
    }
    inferenceManager_->configure(aiProcessors_, config_.modelStages, config_.aiPipelineMode);
}

void VideoStream::setThresholds(float conf, float nms) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.confThreshold = conf;
    config_.nmsThreshold = nms;
    setenv("MANHOLE_CONF_THRESH", std::to_string(conf).c_str(), 1);
    setenv("MANHOLE_NMS_THRESH", std::to_string(nms).c_str(), 1);
    setenv("MODEL_CONF_THRESH", std::to_string(conf).c_str(), 1);
    setenv("MODEL_NMS_THRESH", std::to_string(nms).c_str(), 1);
    ALOGN("[VideoStream] Stream %d: thresholds updated conf=%.3f nms=%.3f",
          config_.streamId, conf, nms);
}

void VideoStream::setModelPath(const std::string& path) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.modelPath = path;
}

void VideoStream::setModelName(const std::string& name) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.modelName = name;
}

void VideoStream::setModelStages(const std::vector<ModelStageConfig>& stages, AIPipelineMode mode) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.modelStages = stages;
    config_.aiPipelineMode = mode;
}

std::string VideoStream::getModelPath() const {
    std::lock_guard<std::mutex> lock(stateMutex_);
    return config_.modelPath;
}

std::vector<std::string> VideoStream::getModelPaths() const {
    std::lock_guard<std::mutex> lock(stateMutex_);
    std::vector<std::string> paths;
    if (!config_.modelStages.empty()) {
        for (const auto& stage : config_.modelStages) {
            paths.push_back(stage.modelPath);
        }
    } else if (!config_.modelPath.empty()) {
        paths.push_back(config_.modelPath);
    }
    return paths;
}
