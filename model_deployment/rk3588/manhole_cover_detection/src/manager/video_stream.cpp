#include "video_stream.h"
#include "../../utilities/sample_log.h"

#include <opencv2/imgproc.hpp>

#include <cstring>
#include <cstdlib>
#include <sstream>

namespace {

inline int align16(int v) { return (v + 15) & ~15; }

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

    if (isMainStream()) {
        // ===== 主码流：demux + MPP 解码 + RGA + OSD + MPP 编码 + RTP/文件 =====
        demux_ = std::make_unique<H264Demux>();
        decoder_ = std::make_unique<RkDecoder>();
        encoder_ = std::make_unique<RkEncoder>();
        decoder_->setFrameCallback([this](const RkNv12Frame& f) { onDecodedFrame(f); });
        encoder_->setPacketCallback([this](const uint8_t* d, size_t s) { pushToOutput(d, s); });

        openOutput();
        if (!decoder_->init()) {
            ALOGE("[VideoStream] Stream %d: MPP decoder init failed", config_.streamId);
            return false;
        }
        // OSD 渲染器必须先 init（initialized_ 默认 false，未 init 时 update 直接返回、不画框）
        if (!osdRenderer_->init()) {
            ALOGE("[VideoStream] Stream %d: OSD renderer init failed", config_.streamId);
            return false;
        }
        ALOGN("[VideoStream] Stream %d: main stream start, input=%s, out=%dx%d@%d bps=%dk",
              config_.streamId, config_.inputSource.c_str(),
              config_.outputWidth, config_.outputHeight, config_.fps, config_.bitrateKbps);
    } else if (isAIStream()) {
        // ===== AI 流：broker -> RGA 640 -> 插件推理 =====
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
        aiBgrBuf_.resize(config_.aiOutputWidth * config_.aiOutputHeight * 3);
        ALOGN("[VideoStream] Stream %d: AI stream start, ai=%dx%d, %zu processor(s)",
              config_.streamId, config_.aiOutputWidth, config_.aiOutputHeight,
              aiProcessors_.size());
    } else {
        ALOGE("[VideoStream] Stream %d: no role configured (main/AI)", config_.streamId);
        return false;
    }

    running_.store(true);
    workerThread_ = std::thread(isMainStream() ? &VideoStream::mainLoop
                                               : &VideoStream::aiLoop, this);
    return true;
}

// ============================ 主码流 ============================

void VideoStream::mainLoop() {
    if (!demux_->open(config_.inputSource)) {
        ALOGE("[VideoStream] Stream %d: cannot open input: %s",
              config_.streamId, config_.inputSource.c_str());
        ended_.store(true);
        return;
    }
    demux_->setPacketCallback([this](const uint8_t* d, size_t s) {
        return decoder_->sendPacket(d, s);
    });

    while (running_.load() && !demux_->eof()) {
        if (!demux_->readOnce()) {
            // EOF 或回调停止
        }
    }
    ALOGN("[VideoStream] Stream %d: main loop exited, frames=%ld",
          config_.streamId, frameCount_.load());
    ended_.store(true);
}

// 解码帧回调（主码流线程内同步执行）
void VideoStream::onDecodedFrame(const RkNv12Frame& frame) {
    if (!frame.valid()) return;

    // 1) 共享给 AI 流（latest-frame 语义）
    if (broker_) broker_->publish(frame.data, frame.width, frame.height, frame.stride);

    // 2) 首帧初始化编码器（编码尺寸/码率来自配置）
    if (!encoderReady_) {
        outStride_ = align16(config_.outputWidth);
        bgrBuf_.resize(static_cast<size_t>(config_.outputWidth) * config_.outputHeight * 3);
        nv12Out_.resize(static_cast<size_t>(outStride_) * config_.outputHeight * 3 / 2);
        if (!encoder_->init(config_.outputWidth, config_.outputHeight, outStride_,
                            config_.fps, config_.bitrateKbps)) {
            ALOGE("[VideoStream] Stream %d: MPP encoder init failed", config_.streamId);
            ended_.store(true);
            return;
        }
        std::vector<uint8_t> hdr;
        if (encoder_->getHeader(hdr) && !hdr.empty()) {
            // 把 SPS/PPS 单独写一次到流最前（若为空则跳过，依赖 EACH_IDR 帧内联）
            pushToOutput(hdr.data(), hdr.size());
            ALOGN("[VideoStream] Stream %d: wrote encoder header %zu bytes",
                  config_.streamId, hdr.size());
        }
        encoderReady_ = true;
    }

    // 3) RGA：NV12 -> 输出尺寸 NV12
    nv12Tmp_.resize(static_cast<size_t>(config_.outputWidth) * config_.outputHeight * 3 / 2);
    if (!rga_ops::resizeNv12(frame.data, frame.width, frame.height, frame.stride,
                             nv12Tmp_.data(), config_.outputWidth, config_.outputHeight,
                             config_.outputWidth)) {
        return;
    }
    // 4) RGA：NV12 -> BGR（画框缓冲）
    if (!rga_ops::nv12ToBgr(nv12Tmp_.data(), config_.outputWidth, config_.outputHeight,
                            config_.outputWidth, bgrBuf_.data())) {
        return;
    }
    // 5) OSD（IVPS OSD region 降级）：主码流编码线程拉取 AI 结果并 CPU 画框
    if (aiResult_) {
        AI_RESULT_T result;
        if (aiResult_->get(result) && osdRenderer_) {
            cv::Mat bgr(config_.outputHeight, config_.outputWidth, CV_8UC3, bgrBuf_.data());
            osdRenderer_->update(&result, config_.outputWidth, config_.outputHeight,
                                 config_.outputWidth, config_.outputHeight, bgr);
        }
    }
    // 6) RGA：BGR -> NV12（编码输入）
    if (!rga_ops::bgrToNv12(bgrBuf_.data(), config_.outputWidth, config_.outputHeight,
                            nv12Out_.data(), outStride_)) {
        return;
    }
    // 7) MPP 编码 -> 回调 pushToOutput -> RTP/文件
    RkNv12Frame enc;
    enc.data = nv12Out_.data();
    enc.width = config_.outputWidth;
    enc.height = config_.outputHeight;
    enc.stride = outStride_;
    enc.size = nv12Out_.size();
    if (!encoder_->encodeFrame(enc)) {
        ALOGE("[VideoStream] Stream %d: encode frame failed", config_.streamId);
        return;
    }

    const long count = frameCount_.fetch_add(1) + 1;
    if (count % 300 == 1) {
        ALOGN("[VideoStream] Stream %d: frame=%ld", config_.streamId, count);
    }
}

void VideoStream::openOutput() {
    if (config_.isFileOutput && !config_.outputFilePath.empty()) {
        fileOut_ = fopen(config_.outputFilePath.c_str(), "wb");
        if (!fileOut_) {
            ALOGE("[VideoStream] Stream %d: cannot open output file: %s",
                  config_.streamId, config_.outputFilePath.c_str());
        }
        ALOGN("[VideoStream] Stream %d: output_mode=file %s",
              config_.streamId, config_.outputFilePath.c_str());
        return;
    }
    if (config_.isMediaMTXOutput && !config_.mediamtxEndpoint.empty()) {
        std::string host = config_.mediamtxEndpoint;
        uint16_t port = 8000;
        size_t colon = host.find(':');
        if (colon != std::string::npos) {
            port = static_cast<uint16_t>(atoi(host.substr(colon + 1).c_str()));
            host = host.substr(0, colon);
        }
        if (rtp_pusher_init(&rtpPusher_, host.c_str(), port, 0) == 0) {
            rtpReady_ = true;
            ALOGN("[VideoStream] Stream %d: output_mode=rtp MediaMTX %s:%u",
                  config_.streamId, host.c_str(), port);
        } else {
            ALOGE("[VideoStream] Stream %d: rtp_pusher_init failed: %s:%u",
                  config_.streamId, host.c_str(), port);
        }
    }
}

void VideoStream::pushToOutput(const uint8_t* data, size_t size) {
    // 诊断：打印前 4 个包的 size 和前 4 字节（应为 00 00 00 01 起始码）
    static int pktLog = 0;
    if (pktLog < 4 && data && size > 0) {
        const unsigned char* d = reinterpret_cast<const unsigned char*>(data);
        ALOGN("[VideoStream] Stream %d: out pkt[%d] size=%zu head=%02x %02x %02x %02x",
              config_.streamId, pktLog++, size,
              size > 0 ? d[0] : 0, size > 1 ? d[1] : 0,
              size > 2 ? d[2] : 0, size > 3 ? d[3] : 0);
    }
    if (fileOut_) {
        fwrite(data, 1, size, fileOut_);
        return;
    }
    if (rtpReady_) {
        pushH264ToRtp(data, size);
    }
}

void VideoStream::pushH264ToRtp(const uint8_t* data, size_t size) {
    // 按起始码切分 NAL，逐个推 RTP（rtp_pusher 内部处理 RTP 分片）
    const uint64_t pts = static_cast<uint64_t>(frameCount_.load()) * 1000000 / config_.fps;
    size_t pos = 0;
    while (pos + 4 <= size) {
        size_t start = pos;
        while (pos + 4 <= size) {
            if (data[pos] == 0 && data[pos + 1] == 0 && data[pos + 2] == 1) {
                pos += 3;
                break;
            }
            if (data[pos] == 0 && data[pos + 1] == 0 && data[pos + 2] == 0 && data[pos + 3] == 1) {
                pos += 4;
                break;
            }
            ++pos;
        }
        if (pos > size) break;
        // [start, pos) 是一个 NAL（含起始码）
        rtp_pusher_push_nalu(&rtpPusher_, data + start, static_cast<uint32_t>(pos - start), pts);
        if (pos + 4 > size) break;
    }
}

// ============================ AI 流 ============================

void VideoStream::aiLoop() {
    Nv12FrameData frame;
    std::vector<uint8_t> nv12Tmp;
    const int aiW = config_.aiOutputWidth;
    const int aiH = config_.aiOutputHeight;
    nv12Tmp.resize(static_cast<size_t>(aiW) * aiH * 3 / 2);

    while (running_.load()) {
        if (!broker_ || !broker_->consumeLatest(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        // RGA：NV12 -> 640x640
        if (!rga_ops::resizeNv12(frame.data.data(), frame.width, frame.height, frame.stride,
                                 nv12Tmp.data(), aiW, aiH, aiW)) {
            continue;
        }
        // RGA：NV12 -> BGR 640
        if (!rga_ops::nv12ToBgr(nv12Tmp.data(), aiW, aiH, aiW, aiBgrBuf_.data())) {
            continue;
        }
        // 插件推理（BGR24 AI_FRAME_T，640x640 时 letterbox 为恒等）
        AI_RESULT_T result;
        memset(&result, 0, sizeof(result));
        AI_FRAME_T f;
        f.data = aiBgrBuf_.data();
        f.width = aiW;
        f.height = aiH;
        f.stride = aiW * 3;
        f.format = AI_FRAME_FORMAT_BGR24;

        bool ok = false;
        if (hasMultipleProcessors()) {
            ok = inferenceManager_->run(&f, &result);
        } else if (aiProcessors_[0]) {
            ok = aiProcessors_[0]->processFrame(&f, &result);
        }
        if (ok && aiResult_) {
            aiResult_->set(result);
        }
        const long count = frameCount_.fetch_add(1) + 1;
        if (ok && result.nObjSize > 0 && count % 300 == 1) {
            ALOGN("[VideoStream] Stream %d: AI detections=%u", config_.streamId, result.nObjSize);
        }
    }
    ALOGN("[VideoStream] Stream %d: AI loop exited, frames=%ld",
          config_.streamId, frameCount_.load());
}

// ============================ 通用 ============================

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

void VideoStream::stop() {
    if (!running_.load() && !workerThread_.joinable()) return;
    running_.store(false);
    if (workerThread_.joinable()) {
        workerThread_.join();
    }
    if (decoder_) decoder_->deinit();
    if (encoder_) encoder_->deinit();
    closeOutput();
    std::lock_guard<std::mutex> lock(stateMutex_);
    aiProcessors_.clear();
}

void VideoStream::closeOutput() {
    if (fileOut_) {
        fclose(fileOut_);
        fileOut_ = nullptr;
    }
    if (rtpReady_) {
        rtp_pusher_deinit(&rtpPusher_);
        rtpReady_ = false;
    }
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
    if (osdRenderer_) osdRenderer_->clear();
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
