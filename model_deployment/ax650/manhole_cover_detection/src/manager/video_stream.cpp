#include "video_stream.h"
#include "video_stream_manager.h"
#include "osd_renderer.h"
#include "ai_processor.h"
#include "../../utilities/sample_log.h"
#include "../../utilities/http_client.h"
#include "../../utilities/json.hpp"
#include "ax_sys_api.h"
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <chrono>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <queue>
#include <deque>
#include <condition_variable>
#include <atomic>
#include <cstdlib>

#include "inference_manager.h"

#define MAX_CONSECUTIVE_ERRORS 5  // 鍏佽鐨勬渶澶ц繛缁敊璇鏁?

static bool is_osd_disabled_by_env() {
    const char* v = std::getenv("AX_DISABLE_OSD");
    return v && v[0] == '1';
}

class AsyncAlarmSender {
public:
    struct AlarmTask {
        std::string url;
        std::string payload;
        int retriesLeft = 3;
        int timeoutSec = 5;
    };

    AsyncAlarmSender() : running_(true), worker_(&AsyncAlarmSender::run, this) {}

    ~AsyncAlarmSender() {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            running_ = false;
        }
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void enqueue(AlarmTask task) {
        // 鍛婅寰岀閫ｄ笉涓婃檪閫插叆鍐峰嵒锛岄伩鍏嶆寔绾屽爢绌嶄换鍕欏奖闊挎暣楂斿鏅傛€?
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
        if (now_ms < cooldown_until_ms_.load(std::memory_order_relaxed)) {
            return;
        }

        // 灏?localhost 鍛婅鏈嶅嫏鎺＄敤鏇寸煭 timeout锛岄伩鍏嶉暦鏅傞枔闃诲鍦ㄥけ鏁楄珛姹?
        if (task.url.find("127.0.0.1") != std::string::npos ||
            task.url.find("localhost") != std::string::npos) {
            task.timeoutSec = 1;
            task.retriesLeft = 0;
        }

        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (queue_.size() >= kMaxQueueSize) {
                // 闅婂垪婊挎檪涓熸鏈€鑸婁换鍕欙紝鍎厛淇濊瓑涓昏闋婚張璺?
                queue_.pop();
            }
            queue_.push(std::move(task));
        }
        cv_.notify_one();
    }

private:
    void run() {
        while (true) {
            AlarmTask task;
            {
                std::unique_lock<std::mutex> lk(mutex_);
                cv_.wait(lk, [&] { return !queue_.empty() || !running_; });
                if (!running_ && queue_.empty()) break;
                task = std::move(queue_.front());
                queue_.pop();
            }

            auto response = HttpClient::post(task.url, task.payload, task.timeoutSec);
            if (response.statusCode != 200) {
                // 寰岀涓嶅彲閬旀檪閬垮厤楂橀牷閲嶈│鑸囬珮闋绘棩瑾岋紝闄嶄綆 CPU/IO 鎶栧嫊灏嶅鏅傞張璺殑褰遍熆
                static int fail_log_count = 0;
                if (++fail_log_count % 120 == 0) {
                    ALOGW("[Alarm] 鍙戦€佸け璐? status=%d, error=%s",
                          response.statusCode, response.error.c_str());
                }

                const bool backend_unreachable = (response.statusCode < 0);
                if (backend_unreachable) {
                    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::steady_clock::now().time_since_epoch())
                                            .count();
                    cooldown_until_ms_.store(now_ms + 30000, std::memory_order_relaxed); // 30s 鐔旀柗
                    // 鐔旀柗鏅傛竻绌洪殜鍒楋紝閬垮厤绌嶅浠诲嫏寰岀簩绐佺櫦閫佸嚭
                    std::lock_guard<std::mutex> lk(mutex_);
                    std::queue<AlarmTask> empty;
                    queue_.swap(empty);
                }
                if (!backend_unreachable && task.retriesLeft-- > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    enqueue(std::move(task));
                }
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<AlarmTask> queue_;
    std::thread worker_;
    bool running_;
    std::atomic<long long> cooldown_until_ms_{0};
    static constexpr size_t kMaxQueueSize = 64;
};

static AsyncAlarmSender g_alarmSender;

// 绮惧害灏嶆瘮绲愭灉涓婂牨锛堢暟姝?POST /api/benchmark/result锛宻ource=edge锛?
class AsyncBenchmarkSender {
public:
    struct Task {
        std::string url;
        std::string payload;
        int timeoutSec = 5;
    };
    AsyncBenchmarkSender() : running_(true), worker_(&AsyncBenchmarkSender::run, this) {}
    ~AsyncBenchmarkSender() {
        { std::lock_guard<std::mutex> lk(mutex_); running_ = false; }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    void enqueue(Task task) {
        { std::lock_guard<std::mutex> lk(mutex_); queue_.push(std::move(task)); }
        cv_.notify_one();
    }
private:
    void run() {
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lk(mutex_);
                cv_.wait(lk, [&] { return !queue_.empty() || !running_; });
                if (!running_ && queue_.empty()) break;
                task = std::move(queue_.front());
                queue_.pop();
            }
            auto response = HttpClient::post(task.url, task.payload, task.timeoutSec);
            if (response.statusCode != 200) {
                ALOGW("[Benchmark] 涓婂牨澶辨晽: status=%d", response.statusCode);
            }
        }
    }
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Task> queue_;
    std::thread worker_;
    bool running_;
};
static AsyncBenchmarkSender g_benchmarkSender;

static std::string toLowerCodec(std::string codec) {
    std::transform(codec.begin(), codec.end(), codec.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return codec;
}

// 鍏ㄥ眬 frame_id 鐢ㄦ柤绮惧害灏嶆瘮涓婂牨锛堝緦绔寜鏅傞枔鎴冲皪榻婏紝姝よ檿鍍呴渶閬炲锛?
static std::atomic<uint64_t> g_benchmark_frame_id(0);

struct BenchmarkReportCfg {
    bool enabled = false;
    std::string backendUrl = "http://127.0.0.1:8001";
    int streamId = -1; // -1: all
};

static BenchmarkReportCfg loadBenchmarkReportCfg() {
    BenchmarkReportCfg cfg;

    // 1) env 闁嬮棞锛堝悜寰岀浉瀹癸級
    const char* backendUrlEnv = getenv("BACKEND_API_URL");
    if (backendUrlEnv && *backendUrlEnv) cfg.backendUrl = backendUrlEnv;

    const char* enableEnv = getenv("BENCHMARK_REPORT");
    if (enableEnv && (strcmp(enableEnv, "1") == 0 || strcmp(enableEnv, "true") == 0)) {
        cfg.enabled = true;
    }
    const char* sidEnv = getenv("BENCHMARK_REPORT_STREAM_ID");
    if (sidEnv && *sidEnv) cfg.streamId = atoi(sidEnv);

    // 2) /dev/shm 妾旀闁嬮棞锛堝彲鐢卞緦绔嫊鎱嬪鍏ワ紝涓嶉渶閲嶅暉閫茬▼锛?
    //    妾旀鏍煎紡锛?
    //    { "enabled": true, "backend_url": "http://x.x.x.x:8001", "stream_id": 0 }
    try {
        static std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now() - std::chrono::seconds(10);
        static BenchmarkReportCfg cached = cfg;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count() < 500) {
            return cached; // 閬垮厤姣忓箑璁€妾?
        }
        last = now;

        std::ifstream f("/dev/shm/benchmark_report.json");
        if (f.good()) {
            std::stringstream ss;
            ss << f.rdbuf();
            auto j = nlohmann::json::parse(ss.str(), nullptr, false);
            if (!j.is_discarded() && j.is_object()) {
                if (j.contains("backend_url") && j["backend_url"].is_string())
                    cfg.backendUrl = j["backend_url"].get<std::string>();
                if (j.contains("stream_id") && j["stream_id"].is_number_integer())
                    cfg.streamId = j["stream_id"].get<int>();
                if (j.contains("enabled") && j["enabled"].is_boolean())
                    cfg.enabled = j["enabled"].get<bool>();
            }
        }
        cached = cfg;
        return cached;
    } catch (...) {
        return cfg;
    }
}

static VideoStreamManager* g_stream_manager = nullptr;
static std::mutex g_stream_manager_mutex;

// 涓婂牨 benchmark / 鍛婅 / OSD 绶╁瓨 / updateAIResult锛圓IWorker 鍚屾鑸囨祦姘寸窔鍏辩敤锛?
void deliverWorkerInferenceResult(VideoStream* s, int streamId, bool processSuccess, AI_RESULT_T& stResult,
                                  double inferenceTimeMs) {
    if (!processSuccess) {
        stResult.nObjSize = 0;
        std::lock_guard<std::mutex> lock(g_stream_manager_mutex);
        if (g_stream_manager) g_stream_manager->updateAIResult(streamId, &stResult);
        return;
    }

    BenchmarkReportCfg bc = loadBenchmarkReportCfg();
    if (bc.enabled && (bc.streamId < 0 || bc.streamId == streamId)) {
        uint64_t frameId = g_benchmark_frame_id++;
        double timestampSec = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        nlohmann::json boxesArr = nlohmann::json::array();
        for (AX_U32 i = 0; i < stResult.nObjSize && i < MAX_DETECT_OBJ_NUM; i++) {
            const AI_OBJ_T& o = stResult.objects[i];
            nlohmann::json b;
            b["x"] = o.x;
            b["y"] = o.y;
            b["w"] = o.w;
            b["h"] = o.h;
            b["score"] = o.score;
            b["label"] = std::string(o.label);
            b["class_id"] = o.class_id;
            boxesArr.push_back(b);
        }
        nlohmann::json payload;
        payload["frame_id"] = static_cast<int64_t>(frameId);
        payload["timestamp"] = timestampSec;
        payload["source"] = "edge";
        payload["boxes"] = boxesArr;
        payload["fps"] = 0.0;
        payload["inference_time"] = inferenceTimeMs;
        AsyncBenchmarkSender::Task task;
        task.url = bc.backendUrl + "/api/benchmark/result";
        task.payload = payload.dump();
        task.timeoutSec = 5;
        g_benchmarkSender.enqueue(std::move(task));
    }

    std::shared_ptr<AIProcessor> firstProc = s->getAIProcessor();
    if (firstProc) {
        std::lock_guard<std::mutex> lock(s->stateMutex_);
        if ((s->cachedInputWidth_ == 0 || s->cachedInputHeight_ == 0)) {
            try {
                firstProc->getInputSize(&s->cachedInputWidth_, &s->cachedInputHeight_);
            } catch (...) {
                s->cachedInputWidth_ = 640;
                s->cachedInputHeight_ = 640;
            }
        }
    }

    std::lock_guard<std::mutex> lock(g_stream_manager_mutex);
    if (g_stream_manager) g_stream_manager->updateAIResult(streamId, &stResult);
}

// 姣忚矾 AI 鍥哄畾涓€鍊?worker 绶氱▼锛岄伩鍏嶆瘡骞€ std::thread().detach() 灏庤嚧闀锋檪闁撻亱琛屽緦绶氱▼/瑷樻喍楂旇硣婧愯€楃洝锛垀10 鍒嗛悩鏂锋祦锛?
struct AIFrameJob {
    std::vector<uint8_t> frameData;
    uint32_t w = 0, h = 0, stridePix = 0, sz = 0;
};

class VideoStream::AIWorker {
public:
    explicit AIWorker(int streamId) : streamId_(streamId) {
        thread_ = std::thread(&AIWorker::run, this);
    }
    ~AIWorker() {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            running_ = false;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
        while (!overlapFutures_.empty()) {
            InferenceManager::FrameResult fr = overlapFutures_.front().get();
            overlapFutures_.pop_front();
            VideoStream* s = nullptr;
            {
                std::lock_guard<std::mutex> lock(g_stream_manager_mutex);
                if (g_stream_manager) s = g_stream_manager->getStream(streamId_);
            }
            if (!s) continue;
            AI_RESULT_T st = fr.result;
            deliverWorkerInferenceResult(s, streamId_, fr.ok, st, 0.0);
        }
    }
    bool submit(std::vector<uint8_t> frameData, uint32_t w, uint32_t h, uint32_t stridePix, uint32_t sz) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (pending_) return false;
        pending_ = std::make_unique<AIFrameJob>();
        pending_->frameData = std::move(frameData);
        pending_->w = w;
        pending_->h = h;
        pending_->stridePix = stridePix;
        pending_->sz = sz;
        cv_.notify_one();
        return true;
    }

private:
    void run() {
        while (true) {
            std::unique_ptr<AIFrameJob> job;
            {
                std::unique_lock<std::mutex> lk(mutex_);
                cv_.wait(lk, [&] { return pending_ != nullptr || !running_; });
                if (!running_) break;
                job = std::move(pending_);
            }
            if (!job) continue;
            VideoStream* s = nullptr;
            {
                std::lock_guard<std::mutex> lock(g_stream_manager_mutex);
                if (g_stream_manager) s = g_stream_manager->getStream(streamId_);
            }
            if (!s || !s->isRunning()) continue;

            AI_RESULT_T stResult;
            memset(&stResult, 0, sizeof(stResult));
            stResult.nObjSize = 0;
            bool processSuccess = false;
            double inferenceTimeMs = 0.0;

            try {
                if (s->inferenceManager_ && s->inferenceManager_->pipelineOverlapActive()) {
                    if (overlapFutures_.size() >= 2) {
                        auto tDrain0 = std::chrono::steady_clock::now();
                        InferenceManager::FrameResult fr = overlapFutures_.front().get();
                        overlapFutures_.pop_front();
                        auto tDrain1 = std::chrono::steady_clock::now();
                        inferenceTimeMs =
                            std::chrono::duration<double, std::milli>(tDrain1 - tDrain0).count();
                        stResult = fr.result;
                        deliverWorkerInferenceResult(s, streamId_, fr.ok, stResult, inferenceTimeMs);
                    }
                    overlapFutures_.push_back(s->inferenceManager_->submitFrameAsync(
                        std::move(job->frameData), job->w, job->h, job->stridePix, job->sz));
                    continue;
                }

                AX_VIDEO_FRAME_T tFrame = {0};
                tFrame.u32Width = job->w;
                tFrame.u32Height = job->h;
                tFrame.u32FrameSize = job->sz;
                tFrame.u64PhyAddr[0] = 0;
                tFrame.u64VirAddr[0] = (AX_U64)job->frameData.data();
                tFrame.u32PicStride[0] = job->stridePix;
                tFrame.enImgFormat = AX_FORMAT_YUV420_SEMIPLANAR;

                auto t0 = std::chrono::steady_clock::now();
                if (s->inferenceManager_) {
                    processSuccess = s->inferenceManager_->run(&tFrame, &stResult);
                } else {
                    processSuccess = false;
                }
                auto t1 = std::chrono::steady_clock::now();
                inferenceTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

                deliverWorkerInferenceResult(s, streamId_, processSuccess, stResult, inferenceTimeMs);
            } catch (const std::exception& e) {
                ALOGE("AI inference exception (worker): %s", e.what());
                s->incrementErrorCount();
                if (s->getErrorCount() > MAX_CONSECUTIVE_ERRORS) {
                    ALOGW("Too many consecutive errors, disabling AI for stream %d", streamId_);
                    s->setAIEnabled(false);
                    std::lock_guard<std::mutex> lock(g_stream_manager_mutex);
                    if (g_stream_manager) g_stream_m…3414 tokens truncated…ks_.fetch_add(1, std::memory_order_relaxed);
}

void VideoStream::onCallbackExit() {
    if (activeCallbacks_.fetch_sub(1, std::memory_order_relaxed) == 1) {
        stopCv_.notify_all();
    }
}

// 鍔ㄦ€佹洿鏂版祦閰嶇疆
void VideoStream::updateConfig(const StreamConfig& newConfig) {
    StreamConfig oldConfig;
    bool needRestart = false;
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        oldConfig = config_;
        config_ = newConfig;
        resolveInputCodecConfig();
        consecutiveInputErrors_ = 0;
        if (inferenceManager_) {
            inferenceManager_->configure(aiProcessors_, config_.modelStages, config_.aiPipelineMode);
        }

        needRestart =
            oldConfig.inputSource != newConfig.inputSource ||
            oldConfig.inputCodec != newConfig.inputCodec ||
            oldConfig.isRTSPOutput != newConfig.isRTSPOutput ||
            oldConfig.rtspEndpoint != newConfig.rtspEndpoint ||
            oldConfig.isMediaMTXOutput != newConfig.isMediaMTXOutput ||
            oldConfig.mediamtxEndpoint != newConfig.mediamtxEndpoint ||
            oldConfig.enableAI != newConfig.enableAI ||
            oldConfig.outputWidth != newConfig.outputWidth ||
            oldConfig.outputHeight != newConfig.outputHeight ||
            oldConfig.vdecGroup != newConfig.vdecGroup ||
            oldConfig.ivpsGroup != newConfig.ivpsGroup ||
            oldConfig.modelPath != newConfig.modelPath;
    }

    if (needRestart && running_) {
        ALOGN("[VideoStream] Stream %d: needRestart=true (config changed), stopping then restarting...", config_.streamId);
        stop();
        if (!start()) {
            ALOGE("[VideoStream] Stream %d: restart failed, rollback to previous config", config_.streamId);
            {
                std::lock_guard<std::mutex> lock(stateMutex_);
                config_ = oldConfig;
            }
            if (!start()) {
                ALOGE("[VideoStream] Stream %d: rollback restart failed, stream remains stopped (no RTP until start succeeds)", config_.streamId);
            } else {
                ALOGN("[VideoStream] Stream %d: rollback start succeeded", config_.streamId);
            }
        } else {
            ALOGN("[VideoStream] Stream %d: restart succeeded", config_.streamId);
        }
    } else if (running_) {
        std::lock_guard<std::mutex> lock(pipelineMutex_);
        pipeline_.m_ivps_attr.n_ivps_fps = config_.fps;
    }
}

// 璁剧疆AI鍚敤鐘舵€?
void VideoStream::setAIEnabled(bool enable) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.enableAI = enable;
    if (!enable && !aiProcessors_.empty()) {
        aiProcessors_.clear();
    }
    if (inferenceManager_) {
        inferenceManager_->configure(aiProcessors_, config_.modelStages, config_.aiPipelineMode);
    }
}

// 娓呴櫎鍛戒护琛屾ā鍨嬫瑷橈紝鍏佽ū Web 閰嶇疆瑕嗚搵
void VideoStream::clearCommandLineModelFlag() {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.isCommandLineModel = false;
}

// 娓呴櫎 OSD 椤ず
void VideoStream::clearOSD() {
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (osdRenderer_) {
        osdRenderer_->clear();
    }
}

// 璁剧疆闃堝€?
void VideoStream::setThresholds(float conf, float nms) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    for (auto& p : aiProcessors_) {
        if (p) p->setThresholds(conf, nms);
    }
}

// 鏇存柊鐣跺墠妯″瀷璺緫锛堝垏鎻涙ā鍨嬪緦蹇呴爤瑾跨敤锛屽惁鍓?getModelPath() 浠嶇偤鑸婂€硷級
void VideoStream::setModelPath(const std::string& path) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.modelPath = path;
}

void VideoStream::setModelName(const std::string& name) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.modelName = name;
}

// 鏇存柊澶氭ā鍨嬮厤缃?
void VideoStream::setModelStages(const std::vector<ModelStageConfig>& stages, AIPipelineMode mode) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.modelStages = stages;
    config_.aiPipelineMode = mode;
    if (!stages.empty()) {
        config_.modelName = stages[0].modelName;
        config_.modelPath = stages[0].modelPath;
    }
    if (inferenceManager_) {
        inferenceManager_->configure(aiProcessors_, config_.modelStages, config_.aiPipelineMode);
    }
}

// 鑾峰彇妯″瀷璺緞锛堝妯″瀷鏅傝繑鍥炵涓€鍊嬶級
std::string VideoStream::getModelPath() const {
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (!config_.modelPath.empty()) return config_.modelPath;
    if (!aiProcessors_.empty() && aiProcessors_[0]) {
        return aiProcessors_[0]->getModelPath();
    }
    return "";
}

std::vector<std::string> VideoStream::getModelPaths() const {
    std::lock_guard<std::mutex> lock(stateMutex_);
    std::vector<std::string> paths;
    if (!config_.modelStages.empty()) {
        for (const auto& s : config_.modelStages)
            paths.push_back(s.modelPath);
    } else {
        std::string one = config_.modelPath.empty() && !aiProcessors_.empty() && aiProcessors_[0]
                          ? aiProcessors_[0]->getModelPath() : config_.modelPath;
        if (!one.empty()) paths.push_back(one);
    }
    return paths;
}

// 鍦?worker 绶氱▼涓煼琛屾帹鐞嗙殑闈滄厠鍑芥暩锛堣垏涓诲洖瑾垮叡浜殑閭忚集锛岄伩鍏嶅湪 IVPS 绶氱▼涓樆濉烇級
bool runAIInference(VideoStream* stream, AX_VIDEO_FRAME_T* tFrame, AI_RESULT_T* stResult) {
    if (!stream || !tFrame || !stResult) return false;
    InferenceManager* mgr = nullptr;
    {
        std::lock_guard<std::mutex> lock(stream->stateMutex_);
        if (stream->pipeline_.n_loog_exit) return false;
        mgr = stream->inferenceManager_.get();
    }
    if (!mgr) return false;
    return mgr->run(tFrame, stResult);
}

void VideoStream::aiInferenceCallback(pipeline_buffer_t* buf) {
    if (!buf) return;

    // 鍦ㄥ嚱鏁稿叆鍙ｅ氨瑷橀寗锛岀⒑瑾嶅洖瑾胯瑾跨敤锛堟瘡30娆¤閷勪竴娆★級
    static int entry_count[64] = {0};
    if (++entry_count[buf->pipeid] % 300 == 0) {
        ALOGN("[AI] aiInferenceCallback ENTRY for pipeid=%d, output_type=%d, frame_size=%dx%d", 
              buf->pipeid, buf->m_output_type, buf->n_width, buf->n_height);
    }

    VideoStream* stream = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_stream_instances_mutex);
        auto it = g_stream_instances.find(buf->pipeid);
        if (it != g_stream_instances.end()) {
            stream = it->second;
        } else {
            // [瑾胯│] 绔嬪嵆瑷橀寗鎵句笉鍒?stream 鐨勬儏娉侊紙姣?0娆¤閷勪竴娆★級
            static int not_found_count[64] = {0};
            if (++not_found_count[buf->pipeid] % 30 == 0) {
                ALOGW("[AI] aiInferenceCallback: stream not found in g_stream_instances for pipeid=%d (total instances: %zu)", 
                      buf->pipeid, g_stream_instances.size());
                // 鎵撳嵃鎵€鏈夊凡瑷诲唺鐨?pipeid
                for (const auto& pair : g_stream_instances) {
                    ALOGW("[AI]   Registered: pipeid=%d -> streamId=%d", 
                          pair.first, pair.second ? pair.second->config_.streamId : -1);
                }
            }
        }
    }
    CallbackGuard cbGuard(stream);
    if (!stream) {
        return;
    }
    
    // 瑷橀寗鍥炶琚鐢紙姣?0骞€瑷橀寗涓€娆★級
    static int callback_count[64] = {0};
    if (++callback_count[stream->config_.streamId] % 300 == 0) {
        ALOGN("[AI] aiInferenceCallback called for stream %d (pipeid=%d, frame_size=%d)", 
              stream->config_.streamId, buf->pipeid, buf->n_size);
    }

    // 妾㈡煡閫€鍑烘瑾岋紝鎻愬墠杩斿洖閬垮厤铏曠悊宸插仠姝㈢殑娴?
    if (stream->pipeline_.n_loog_exit) {
        return;
    }

    std::vector<std::shared_ptr<AIProcessor>> processors;
    AIPipelineMode pipelineMode = AIPipelineMode::Parallel;
    {
        std::lock_guard<std::mutex> lock(stream->stateMutex_);
        if (stream->pipeline_.n_loog_exit) return;
        processors = stream->aiProcessors_;
        pipelineMode = stream->config_.aiPipelineMode;
    }
    if (processors.empty()) {
        static int no_processor_count[64] = {0};
        if (++no_processor_count[stream->config_.streamId] % 100 == 0) {
            ALOGW("[AI] aiInferenceCallback: no AIProcessor for stream %d (pipeid=%d)", 
                  stream->config_.streamId, buf->pipeid);
        }
        return;
    }

    // [鏁堣兘/寤堕伈鍎寲] 闄嶄綆 AI 鎺ㄧ悊闋荤巼鍒颁富纰兼祦鐨?1/3锛堜緥濡?30fps 鈫?10fps锛?
    static int infer_stride_count[64] = {0};
    static int adaptive_stride[64] = {0};
    static int submit_fail_count[64] = {0};
    static int submit_ok_count[64] = {0};
    const int sid = stream->config_.streamId;
    const int idx = (sid >= 0 && sid < 64) ? sid : 0;
    if (adaptive_stride[idx] == 0) adaptive_stride[idx] = 6;
    if (++infer_stride_count[idx] % adaptive_stride[idx] != 0) {
        return;
    }

    // [闂滈嵉淇京] 澶氳矾澶氭ā鍨嬫檪 AI 鍥炶鏈冮樆濉?IVPS 绶氱▼锛屽皫鑷磋垏涓荤⒓娴佸叡鐢ㄧ殑 VDEC 鐒℃硶鍚戜富 IVPS 閫佸箑锛屼富纰兼祦 RTP 鍋滄銆?
    // 鍋氭硶锛氳瑁藉箑寰岀珛鍗宠繑鍥烇紝鍦?worker 绶氱▼涓煼琛屾帹鐞嗭紝浣?IVPS 绶氱▼鐩″揩 ReleaseChnFrame锛屼笉闃诲 VDEC銆?
    AX_BOOL bMapped = AX_FALSE;
    if (!buf->p_vir && buf->p_phy) {
        buf->p_vir = AX_SYS_Mmap(buf->p_phy, buf->n_size);
        bMapped = AX_TRUE;
    }
    MmapGuard mmapGuard{ bMapped ? buf->p_vir : nullptr, static_cast<size_t>(bMapped ? buf->n_size : 0) };
    if (!buf->p_vir) return;

    std::vector<uint8_t> frameData((uint8_t*)buf->p_vir, (uint8_t*)buf->p_vir + buf->n_size);
    const uint32_t w = buf->n_width, h = buf->n_height, stridePix = buf->n_stride, sz = buf->n_size;

    // 鎻愪氦绲︽湰娴佸皥鐢?worker 绶氱▼锛堝浐瀹氱窔绋嬶紝涓嶆瘡骞€鏂板缓锛夛紝浣囧垪婊垮墖涓熷箑
    if (!stream->submitAIFrame(std::move(frameData), w, h, stridePix, sz)) {
        submit_fail_count[idx]++;
        submit_ok_count[idx] = 0;
        // Worker 绻佸繖鏅傝嚜閬╂噳闄嶉牷锛屽劒鍏堜繚闅滀富纰兼祦/VENC 璺緫銆?
        if (submit_fail_count[idx] % 4 == 0 && adaptive_stride[idx] < 18) {
            adaptive_stride[idx]++;
        }
        if (submit_fail_count[idx] % 60 == 0) {
            ALOGW("[AI] stream %d worker busy, drop=%d, adapt_stride=%d",
                  sid, submit_fail_count[idx], adaptive_stride[idx]);
        }
        return;
    }
    submit_fail_count[idx] = 0;
    if (++submit_ok_count[idx] >= 120) {
        submit_ok_count[idx] = 0;
        if (adaptive_stride[idx] > 4) adaptive_stride[idx]--;
    }
}

bool VideoStream::submitAIFrame(std::vector<uint8_t> frameData, uint32_t w, uint32_t h, uint32_t stridePix, uint32_t sz) {
    if (!aiWorker_) return false;
    return aiWorker_->submit(std::move(frameData), w, h, stridePix, sz);
}

// 鍒涘缓澶勭悊娴佺▼锛坧ipeline锛?
bool VideoStream::createProcessingPipeline() {
    // 閲嶆柊瑷疆 pipeline 鐨勮几鍑洪鍨嬪拰鍥炶鍑芥暩锛堝弮鑰?ai_platform_RTP锛?
    // 閫欎簺瑷疆蹇呴爤鍦ㄦ瘡娆″壍寤?pipeline 鏅傞噸鏂拌ō缃?
    if (config_.isRTSPOutput) {
        // RTSP鎺ㄦ祦
        pipeline_.m_output_type = po_rtsp_h264;
        snprintf(pipeline_.m_venc_attr.end_point, sizeof(pipeline_.m_venc_attr.end_point), 
                "%s%d", config_.rtspEndpoint.c_str(), config_.streamId);
        pipeline_.m_venc_attr.n_venc_chn = config_.streamId;
    } else if (config_.isMediaMTXOutput) {
        // MediaMTX鎺ㄩ€?
        // 姣忓€嬩富纰兼祦浣跨敤涓嶅悓鐨?VENC channel锛坰treamId锛夛紝閬垮厤澶氬€嬩富纰兼祦琛濈獊
        pipeline_.m_output_type = po_mediamtx_h264;
        snprintf(pipeline_.m_venc_attr.end_point, sizeof(pipeline_.m_venc_attr.end_point), 
                "%s", config_.mediamtxEndpoint.c_str());
        pipeline_.m_venc_attr.n_venc_chn = config_.streamId;  // 浣跨敤 streamId 浣滅偤 VENC channel锛岄伩鍏嶈绐?
        ALOGN("[VideoStream] Stream %d MediaMTX endpoint set to: %s", 
              config_.streamId, config_.mediamtxEndpoint.c_str());
    } else if (config_.enableAI) {
        // AI鎺ㄧ悊杈撳嚭涓篘V12缂撳啿鍖?
        pipeline_.m_output_type = po_buff_nv12;
        pipeline_.output_func = aiInferenceCallback;  // 閲嶆柊瑷疆鍥炶鍑芥暩
        // AI 娴佷笉浣跨敤 VENC锛屾竻闄?VENC 閰嶇疆
        pipeline_.m_venc_attr.n_venc_chn = -1;  // 瑷疆鐐虹劇鏁堝€硷紝琛ㄧず涓嶄娇鐢?VENC
        if (!aiWorker_) aiWorker_ = std::make_unique<AIWorker>(config_.streamId);
        ALOGN("[VideoStream] Setting output_func for AI stream %d", config_.streamId);
    }
    
    // 瑷疆 pipeline ID锛堝繀闋堝湪 create_pipeline 涔嬪墠瑷疆锛?
    pipeline_.pipeid = config_.streamId;
    
    // 瑷疆杓稿叆椤炲瀷锛堟敮鎻?inputCodec=h264/h265/auto锛?
    pipeline_.m_input_type = preferredInputType_;
    
    // 鎸夊綋鍓嶉厤缃垵濮嬪寲pipeline
    configureIVPS();
    configureVDEC();
    // AI 娴佷娇鐢?po_buff_nv12锛屼笉闇€瑕?VENC
    // 鍙湁 RTSP 鍜?MediaMTX 杓稿嚭鎵嶉渶瑕?VENC
    if (config_.isRTSPOutput || config_.isMediaMTXOutput) {
        configureVENC();
    } else if (config_.enableAI) {
        // AI 娴侊細纰轰繚涓嶉厤缃?VENC锛堜娇鐢?po_buff_nv12锛?
        // 娓呴櫎浠讳綍鍙兘鐨?VENC 閰嶇疆
        pipeline_.m_venc_attr.n_venc_chn = -1;  // 瑷疆鐐虹劇鏁堝€硷紝琛ㄧず涓嶄娇鐢?VENC
    }
    
    // 鍟熺敤 pipeline锛堝繀闋堝湪 create_pipeline 涔嬪墠瑷疆锛?
    pipeline_.enable = 1;
    pipeline_.n_loog_exit = 0;  // 閲嶇疆閫€鍑烘瑾岋紙鍙冭€?ai_platform_RTP锛?
    
    // 鍚姩pipeline
    int ret = create_pipeline(&pipeline_);
    if (ret != 0) {
        ALOGE("Failed to create pipeline %d", config_.streamId);
        pipeline_.enable = 0;  // 鍓靛缓澶辨晽鏅傞噸缃?
        return false;
    }
    
    // 鍒ゆ柗鏄惁闇€瑕?OSD锛氬儏鍦ㄥ暉鐢?AI 鏅傛帥杓夛紝闂滈枆 AI 鏅傚彲娓│鏄惁鐐?OSD 灏庤嚧娈樺奖
    bool needOSD = config_.enableAI && !is_osd_disabled_by_env();
    if (needOSD && !osdRenderer_) {
        osdRenderer_ = make_unique<OSDRenderer>(&pipeline_);
    }
    
    // 鍒濆鍖朞SD
    // #region agent log
    debug_log("video_stream.cpp:562", "OSD initialization: starting", {
        {"stream_id", std::to_string(config_.streamId)},
        {"has_osd_renderer", osdRenderer_ ? "true" : "false"},
        {"ivps_group", std::to_string(config_.ivpsGroup)},
        {"n_osd_rgn", std::to_string(pipeline_.m_ivps_attr.n_osd_rgn)}
    });
    // #endregion
    if (osdRenderer_ && !osdRenderer_->init()) {
        // #region agent log
        debug_log("video_stream.cpp:563", "OSD initialization: failed", {
            {"stream_id", std::to_string(config_.streamId)}
        });
        // #endregion
        ALOGE("OSD initialization failed for stream %d", config_.streamId);
        destroyProcessingPipeline();
        return false;
    }
    // #region agent log
    if (osdRenderer_) {
        debug_log("video_stream.cpp:563", "OSD initialization: success", {
            {"stream_id", std::to_string(config_.streamId)},
            {"ivps_group", std::to_string(config_.ivpsGroup)}
        });
        osdRenderer_->clear();
    }
    // #endregion
    
    ALOGN("[VideoStream] Pipeline %d created successfully, output_type=%d, output_func=%p", 
          config_.streamId, pipeline_.m_output_type, pipeline_.output_func);
    
    return true;
}

void VideoStream::destroyProcessingPipeline() {
    aiWorker_.reset();
    if (pipeline_.enable) {
        destory_pipeline(&pipeline_);
        pipeline_.enable = 0;
    }
}

void VideoStream::configureIVPS() {
    auto& ivps = pipeline_.m_ivps_attr;
    ivps.n_ivps_grp = config_.ivpsGroup;
    ivps.n_ivps_width = config_.outputWidth;
    ivps.n_ivps_height = config_.outputHeight;
    ivps.n_ivps_fps = config_.fps;
    // 璁?IVPS 鍦ㄩ渶瑕佽几鍑虹郸 RTSP/MediaMTX锛堜富纰兼祦锛夋檪涔熻兘鍟熷嫊銆?
    // Raw 涓荤⒓娴佺殑 enableAI=false锛堜笉鐣?OSD锛変絾浠嶉渶瑕?IVPS 鐢㈢敓鍙法纰艰几鍑恒€?
    ivps.n_fifo_count =
        (config_.enableAI || config_.isRTSPOutput || config_.isMediaMTXOutput) ? 1 : 0;
    bool needOSD = config_.enableAI && !is_osd_disabled_by_env();  // 鍙敱 AX_DISABLE_OSD=1 寮峰埗闂滈枆 OSD
    // 鍙娇鐢?1 鍊?OSD 鍗€鍩熶甫鍙洿鏂拌┎鍗€鍩燂紝閬垮厤澶氬崁鍩熸湭鍚屾鏇存柊灏庤嚧 IVPS 鍚堟垚鑺卞睆/鑹插
    ivps.n_osd_rgn = needOSD ? 1 : 0;
    
    // 瑷橀寗 IVPS 閰嶇疆
    ALOGN("[VideoStream] Stream %d IVPS config: grp=%d, size=%dx%d, fps=%d, n_fifo_count=%d, n_osd_rgn=%d, enableAI=%d", 
          config_.streamId, ivps.n_ivps_grp, ivps.n_ivps_width, ivps.n_ivps_height, 
          ivps.n_ivps_fps, ivps.n_fifo_count, ivps.n_osd_rgn, config_.enableAI ? 1 : 0);
}

void VideoStream::configureVDEC() {
    pipeline_.m_vdec_attr.n_vdec_grp = config_.vdecGroup;
}

void VideoStream::resolveInputCodecConfig() {
    std::string codec = toLowerCodec(config_.inputCodec);
    if (codec.empty() || codec == "auto") {
        preferredInputType_ = pi_vdec_h264;
        autoCodecFallbackEnabled_ = true;
        autoCodecFallbackSwitched_ = false;
        return;
    }

    autoCodecFallbackEnabled_ = false;
    autoCodecFallbackSwitched_ = false;
    if (codec == "h265" || codec == "hevc") {
        // 鐩墠 pipeline_input_e 灏氭湭鎻愪緵 H265 灏嶆噳杓稿叆椤炲瀷锛屽厛瀹夊叏闄嶇礆鐐?H264 閬垮厤绶ㄨ澶辨晽銆?
        ALOGW("[VideoStream] Stream %d input codec '%s' requested but H265 VDEC input type is unavailable, fallback to H264",
              config_.streamId, codec.c_str());
        preferredInputType_ = pi_vdec_h264;
    } else {
        preferredInputType_ = pi_vdec_h264;
    }
}

bool VideoStream::tryAutoCodecFallbackLocked() {
    if (preferredInputType_ != pi_vdec_h264) {
        return false;
    }

    // 鐩墠涓嶆敮鎻?H265 VDEC input type锛屼繚鐣欑媭鎱嬩絾涓嶅槜瑭﹂噸寤?pipeline銆?
    ALOGW("[VideoStream] Stream %d input codec auto-fallback skipped: H265 VDEC input type is unavailable",
          config_.streamId);
    autoCodecFallbackSwitched_ = true;
    return false;
}

void VideoStream::configureVENC() {
    auto& venc = pipeline_.m_venc_attr;
    if (config_.isRTSPOutput) {
        snprintf(venc.end_point, sizeof(venc.end_point), 
                "%s%d", config_.rtspEndpoint.c_str(), config_.streamId);
        venc.n_venc_chn = config_.streamId;
    } else if (config_.isMediaMTXOutput) {
        snprintf(venc.end_point, sizeof(venc.end_point), 
                "%s", config_.mediamtxEndpoint.c_str());
        venc.n_venc_chn = config_.streamId;
        ALOGN("[VideoStream] Stream %d VENC configured: end_point=%s, venc_chn=%d", 
              config_.streamId, venc.end_point, venc.n_venc_chn);
    }
}





