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

#define MAX_CONSECUTIVE_ERRORS 5  // 閸忎浇顔忛惃鍕付婢堆嗙箾缂侇參鏁婄拠顖涱偧閺?

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
        // 閸涘﹨顒熷宀€顏柅锝勭瑝娑撳﹥妾柅鎻掑弳閸愬嘲宓掗敍宀勪缉閸忓秵瀵旂痪灞界垻缁屽秳鎹㈤崟娆忓闂婃寧鏆ｆ鏂款嚊閺呭倹鈧?
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
        if (now_ms < cooldown_until_ms_.load(std::memory_order_relaxed)) {
            return;
        }

        // 鐏?localhost 閸涘﹨顒熼張宥呭珡閹猴紕鏁ら弴瀵哥叚 timeout閿涘矂浼╅崗宥夋殾閺呭倿鏋旈梼璇差敚閸︺劌銇戦弫妤勭彌濮?
        if (task.url.find("127.0.0.1") != std::string::npos ||
            task.url.find("localhost") != std::string::npos) {
            task.timeoutSec = 1;
            task.retriesLeft = 0;
        }

        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (queue_.size() >= kMaxQueueSize) {
                // 闂呭﹤鍨鎸庢娑撶喐顥夐張鈧懜濠佹崲閸曟瑱绱濋崕顏勫帥娣囨繆鐡戞稉鏄忣浕闂嬪寮电捄?
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
                // 瀵板瞼顏稉宥呭讲闁梹妾柆鍨帳妤傛﹢鐗烽柌宥堚攤閼稿洭鐝棆缁樻）鐟惧矉绱濋梽宥勭秵 CPU/IO 閹舵牕瀚婄亸宥咁嚊閺呭倿寮电捄顖滄畱瑜伴亶鐔?                static int fail_log_count = 0;
                if (++fail_log_count % 120 == 0) {
                    ALOGW("[Alarm] 閸欐垿鈧礁銇戠拹? status=%d, error=%s",
                          response.statusCode, response.error.c_str());
                }

                const bool backend_unreachable = (response.statusCode < 0);
                if (backend_unreachable) {
                    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::steady_clock::now().time_since_epoch())
                                            .count();
                    cooldown_until_ms_.store(now_ms + 30000, std::memory_order_relaxed); // 30s 閻旀梹鏌?                    // 閻旀梹鏌楅弲鍌涚缁屾椽娈滈崚妤嬬礉闁灝鍘ょ粚宥咁棙娴犺瀚忓宀€绨╃粣浣烘闁礁鍤?                    std::lock_guard<std::mutex> lk(mutex_);
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

// 缁儳瀹崇亸宥嗙槷缁叉劖鐏夋稉濠傜墾閿涘牏鏆熷?POST /api/benchmark/result閿涘ource=edge閿?
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
                ALOGW("[Benchmark] 娑撳﹤鐗ㄦ径杈ㄦ櫧: status=%d", response.statusCode);
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

// 閸忋劌鐪?frame_id 閻劍鏌ょ划鎯у鐏忓秵鐦稉濠傜墾閿涘牆绶︾粩顖涘瘻閺呭倿鏋旈幋鍐茬毆姒诲绱濆銈堟閸嶅懘娓堕柆鐐差杻閿?
static std::atomic<uint64_t> g_benchmark_frame_id(0);

struct BenchmarkReportCfg {
    bool enabled = false;
    std::string backendUrl = "http://127.0.0.1:8001";
    int streamId = -1; // -1: all
};

static BenchmarkReportCfg loadBenchmarkReportCfg() {
    BenchmarkReportCfg cfg;

    // 1) env 闂佸妫為敍鍫濇倻瀵板瞼娴夌€圭櫢绱?    const char* backendUrlEnv = getenv("BACKEND_API_URL");
    if (backendUrlEnv && *backendUrlEnv) cfg.backendUrl = backendUrlEnv;

    const char* enableEnv = getenv("BENCHMARK_REPORT");
    if (enableEnv && (strcmp(enableEnv, "1") == 0 || strcmp(enableEnv, "true") == 0)) {
        cfg.enabled = true;
    }
    const char* sidEnv = getenv("BENCHMARK_REPORT_STREAM_ID");
    if (sidEnv && *sidEnv) cfg.streamId = atoi(sidEnv);

    // 2) /dev/shm 濡炬梹顢嶉梺瀣閿涘牆褰查悽鍗炵乏缁旑垰瀚婇幈瀣嚑閸忋儻绱濇稉宥夋付闁插秴鏆夐柅鑼柤閿?
    //    濡炬梹顢嶉弽鐓庣础閿?
    //    { "enabled": true, "backend_url": "http://x.x.x.x:8001", "stream_id": 0 }
    try {
        static std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now() - std::chrono::seconds(10);
        static BenchmarkReportCfg cached = cfg;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count() < 500) {
            return cached; // 闁灝鍘ゅВ蹇撶畱鐠佲偓濡?
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

// 娑撳﹤鐗?benchmark / 閸涘﹨顒?/ OSD 缁垛晛鐡?/ updateAIResult閿涘湏IWorker 閸氬本顒為懜鍥ㄧウ濮樺绐旈崗杈╂暏閿?
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

// 濮ｅ繗鐭?AI 閸ュ搫鐣炬稉鈧崐?worker 缁舵氨鈻奸敍宀勪缉閸忓秵鐦￠獮鈧?std::thread().detach() 鐏忓氦鍤ч梹閿嬫闂佹捇浜辩悰灞界乏缁舵氨鈻?鐟锋ɑ鍠嶆鏃囩。濠ф劘鈧娲濋敍鍨€10 閸掑棝鎮╅弬閿嬬ウ閿?
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
                    if (g_stream_manager) g_stream_manager->updateAIResult(streamId_, &stResult);
                }
            }
        }
    }
};

void VideoStream::onCallbackExit() {
    if (activeCallbacks_.fetch_sub(1, std::memory_order_relaxed) == 1) {
        stopCv_.notify_all();
    }
}

// 閸斻劍鈧焦娲块弬鐗堢ウ闁板秶鐤?void VideoStream::updateConfig(const StreamConfig& newConfig) {
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

// 鐠佸墽鐤咥I閸氼垳鏁ら悩鑸碘偓?
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

// 濞撳懘娅庨崨鎴掓姢鐞涘本膩閸ㄥ顬跨懛姗堢礉閸忎浇奴 Web 闁板秶鐤嗙憰鍡氭惖
void VideoStream::clearCommandLineModelFlag() {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.isCommandLineModel = false;
}

// 濞撳懘娅?OSD 妞ゎ垳銇?void VideoStream::clearOSD() {
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (osdRenderer_) {
        osdRenderer_->clear();
    }
}

// 鐠佸墽鐤嗛梼鍫濃偓?
void VideoStream::setThresholds(float conf, float nms) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    for (auto& p : aiProcessors_) {
        if (p) p->setThresholds(conf, nms);
    }
}

// 閺囧瓨鏌婇悾璺哄濡€崇€风捄顖氱帆閿涘牆鍨忛幓娑櫮侀崹瀣乏韫囧懘鐖ょ懢璺ㄦ暏閿涘苯鎯侀崜?getModelPath() 娴犲秶鍋ら懜濠傗偓纭风礆
void VideoStream::setModelPath(const std::string& path) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.modelPath = path;
}

void VideoStream::setModelName(const std::string& name) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.modelName = name;
}

// 閺囧瓨鏌婃径姘侀崹瀣帳缂?
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

// 閼惧嘲褰囧Ο鈥崇€风捄顖氱窞閿涘牆顦垮Ο鈥崇€烽弲鍌濈箲閸ョ偟顑囨稉鈧崐瀣剁礆
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

// 閸?worker 缁舵氨鈻兼稉顓炵吋鐞涘本甯归悶鍡欐畱闂堟粍鍘犻崙鑺ユ毄閿涘牐鍨忔稉璇叉礀鐟惧灝鍙℃禍顐ゆ畱闁繗闆嗛敍宀勪缉閸忓秴婀?IVPS 缁舵氨鈻兼稉顓㈡▎婵夌儑绱?bool runAIInference(VideoStream* stream, AX_VIDEO_FRAME_T* tFrame, AI_RESULT_T* stResult) {
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

    // 閸︺劌鍤遍弫绋垮弳閸欙絽姘ㄧ懛姗€瀵楅敍宀€鈷戠懢宥呮礀鐟捐儻顫︾懢璺ㄦ暏閿涘牊鐦?0濞喡ゎ灳闁峰嫪绔村▎鈽呯礆
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
            // [鐟捐儻鈹俔 缁斿宓嗙懛姗€瀵楅幍鍙ョ瑝閸?stream 閻ㄥ嫭鍎忓▔渚婄礄濮?0濞喡ゎ灳闁峰嫪绔村▎鈽呯礆
            static int not_found_count[64] = {0};
            if (++not_found_count[buf->pipeid] % 30 == 0) {
                ALOGW("[AI] aiInferenceCallback: stream not found in g_stream_instances for pipeid=%d (total instances: %zu)", 
                      buf->pipeid, g_stream_instances.size());
                // 閹垫挸宓冮幍鈧張澶婂嚒鐟疯鍞洪惃?pipeid
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
    
    // 鐟锋﹢瀵楅崶鐐额€炵悮顐ヮ€為悽顭掔礄濮?0楠炩偓鐟锋﹢瀵楁稉鈧▎鈽呯礆
    static int callback_count[64] = {0};
    if (++callback_count[stream->config_.streamId] % 300 == 0) {
        ALOGN("[AI] aiInferenceCallback called for stream %d (pipeid=%d, frame_size=%d)", 
              stream->config_.streamId, buf->pipeid, buf->n_size);
    }

    // 濡俱垺鐓￠柅鈧崙鐑橆灴鐟惧矉绱濋幓鎰鏉╂柨娲栭柆鍨帳閾忔洜鎮婂鎻掍粻濮濄垻娈戝ù?
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

    // [閺佸牐鍏?瀵ゅ爼浼堥崕顏勫] 闂勫秳缍?AI 閹恒劎鎮婇棆鑽ゅ芳閸掗瀵岀喊鍏肩ウ閻?1/3閿涘牅绶ユ俊?30fps 閳?10fps閿?
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

    // [闂傛粓宓夋穱顔间含] 婢舵俺鐭炬径姘侀崹瀣 AI 閸ョ偠顎為張鍐▎婵?IVPS 缁舵氨鈻奸敍灞界毇閼风鍨忔稉鑽も挀濞翠礁鍙￠悽銊ф畱 VDEC 閻掆剝纭堕崥鎴滃瘜 IVPS 闁礁绠戦敍灞煎瘜绾板吋绁?RTP 閸嬫粍顒涢妴?
    // 閸嬫碍纭堕敍姘愁槵鐟佽棄绠戝宀€鐝涢崡瀹犵箲閸ョ儑绱濋崷?worker 缁舵氨鈻兼稉顓炵吋鐞涘本甯归悶鍡礉娴?IVPS 缁舵氨鈻奸惄鈥虫彥 ReleaseChnFrame閿涘奔绗夐梼璇差敚 VDEC閵?
    AX_BOOL bMapped = AX_FALSE;
    if (!buf->p_vir && buf->p_phy) {
        buf->p_vir = AX_SYS_Mmap(buf->p_phy, buf->n_size);
        bMapped = AX_TRUE;
    }
    MmapGuard mmapGuard{ bMapped ? buf->p_vir : nullptr, static_cast<size_t>(bMapped ? buf->n_size : 0) };
    if (!buf->p_vir) return;

    std::vector<uint8_t> frameData((uint8_t*)buf->p_vir, (uint8_t*)buf->p_vir + buf->n_size);
    const uint32_t w = buf->n_width, h = buf->n_height, stridePix = buf->n_stride, sz = buf->n_size;

    // 閹绘劒姘︾徊锔芥拱濞翠礁鐨ラ悽?worker 缁舵氨鈻奸敍鍫濇祼鐎规氨绐旂粙瀣剁礉娑撳秵鐦￠獮鈧弬鏉跨紦閿涘绱濇担鍥у灙濠婂灝澧栨稉鐔风畱
    if (!stream->submitAIFrame(std::move(frameData), w, h, stridePix, sz)) {
        submit_fail_count[idx]++;
        submit_ok_count[idx] = 0;
        // Worker 缁讳礁绻栭弲鍌濆殰闁晜鍣抽梽宥夌壏閿涘苯鍔掗崗鍫滅箽闂呮粈瀵岀喊鍏肩ウ/VENC 鐠侯垰绶妴?
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

// 閸掓稑缂撴径鍕倞濞翠胶鈻奸敍鍧peline閿?
bool VideoStream::createProcessingPipeline() {
    // 闁插秵鏌婄懛顓犵枂 pipeline 閻ㄥ嫯鍑犻崙娲敚閸ㄥ鎷伴崶鐐额€為崙鑺ユ毄閿涘牆寮懓?ai_platform_RTP閿?
    // 闁瑤绨虹懛顓犵枂韫囧懘鐖ら崷銊︾槨濞嗏€冲瀵?pipeline 閺呭倿鍣搁弬鎷屌嶇純?
    if (config_.isRTSPOutput) {
        // RTSP閹恒劍绁?        pipeline_.m_output_type = po_rtsp_h264;
        snprintf(pipeline_.m_venc_attr.end_point, sizeof(pipeline_.m_venc_attr.end_point), 
                "%s%d", config_.rtspEndpoint.c_str(), config_.streamId);
        pipeline_.m_venc_attr.n_venc_chn = config_.streamId;
    } else if (config_.isMediaMTXOutput) {
        // MediaMTX閹恒劑鈧?
        // 濮ｅ繐鈧瀵岀喊鍏肩ウ娴ｈ法鏁ゆ稉宥呮倱閻?VENC channel閿涘澃treamId閿涘绱濋柆鍨帳婢舵艾鈧瀵岀喊鍏肩ウ鐞涙繄鐛?        pipeline_.m_output_type = po_mediamtx_h264;
        snprintf(pipeline_.m_venc_attr.end_point, sizeof(pipeline_.m_venc_attr.end_point), 
                "%s", config_.mediamtxEndpoint.c_str());
        pipeline_.m_venc_attr.n_venc_chn = config_.streamId;  // 娴ｈ法鏁?streamId 娴ｆ粎鍋?VENC channel閿涘矂浼╅崗宥堫敘缁?
        ALOGN("[VideoStream] Stream %d MediaMTX endpoint set to: %s", 
              config_.streamId, config_.mediamtxEndpoint.c_str());
    } else if (config_.enableAI) {
        // AI閹恒劎鎮婃潏鎾冲毉娑撶瘶V12缂傛挸鍟块崠?
        pipeline_.m_output_type = po_buff_nv12;
        pipeline_.output_func = aiInferenceCallback;  // 闁插秵鏌婄懛顓犵枂閸ョ偠顎為崙鑺ユ毄
        // AI 濞翠椒绗夋担璺ㄦ暏 VENC閿涘本绔婚梽?VENC 闁板秶鐤?        pipeline_.m_venc_attr.n_venc_chn = -1;  // 鐟奉厾鐤嗛悙铏瑰妵閺佸牆鈧》绱濈悰銊с仛娑撳秳濞囬悽?VENC
        if (!aiWorker_) aiWorker_ = std::make_unique<AIWorker>(config_.streamId);
        ALOGN("[VideoStream] Setting output_func for AI stream %d", config_.streamId);
    }
    
    // 鐟奉厾鐤?pipeline ID閿涘牆绻€闂嬪牆婀?create_pipeline 娑斿澧犵懛顓犵枂閿?
    pipeline_.pipeid = config_.streamId;
    
    // 鐟奉厾鐤嗘潛绋垮弳妞ょ偛鐎烽敍鍫熸暜閹?inputCodec=h264/h265/auto閿?
    pipeline_.m_input_type = preferredInputType_;
    
    // 閹稿缍嬮崜宥夊帳缂冾喖鍨垫慨瀣pipeline
    configureIVPS();
    configureVDEC();
    // AI 濞翠椒濞囬悽?po_buff_nv12閿涘奔绗夐棁鈧憰?VENC
    // 閸欘亝婀?RTSP 閸?MediaMTX 鏉撶鍤幍宥夋付鐟?VENC
    if (config_.isRTSPOutput || config_.isMediaMTXOutput) {
        configureVENC();
    } else if (config_.enableAI) {
        // AI 濞翠緤绱扮喊杞扮箽娑撳秹鍘ょ純?VENC閿涘牅濞囬悽?po_buff_nv12閿?
        // 濞撳懘娅庢禒璁崇秿閸欘垵鍏橀惃?VENC 闁板秶鐤?        pipeline_.m_venc_attr.n_venc_chn = -1;  // 鐟奉厾鐤嗛悙铏瑰妵閺佸牆鈧》绱濈悰銊с仛娑撳秳濞囬悽?VENC
    }
    
    // 閸熺喓鏁?pipeline閿涘牆绻€闂嬪牆婀?create_pipeline 娑斿澧犵懛顓犵枂閿?
    pipeline_.enable = 1;
    pipeline_.n_loog_exit = 0;  // 闁插秶鐤嗛柅鈧崙鐑橆灴鐟惧矉绱欓崣鍐偓?ai_platform_RTP閿?
    
    // 閸氼垰濮﹑ipeline
    int ret = create_pipeline(&pipeline_);
    if (ret != 0) {
        ALOGE("Failed to create pipeline %d", config_.streamId);
        pipeline_.enable = 0;  // 閸撻潧缂撴径杈ㄦ櫧閺呭倿鍣哥純?
        return false;
    }
    
    // 閸掋倖鏌楅弰顖氭儊闂団偓鐟?OSD閿涙艾鍎忛崷銊ユ殙閻?AI 閺呭倹甯ユ潛澶涚礉闂傛粓鏋?AI 閺呭倸褰插〒顒冣攤閺勵垰鎯侀悙?OSD 鐏忓氦鍤у▓妯哄
    bool needOSD = config_.enableAI && !is_osd_disabled_by_env();
    if (needOSD && !osdRenderer_) {
        osdRenderer_ = make_unique<OSDRenderer>(&pipeline_);
    }
    
    // 閸掓繂顫愰崠鏈濻D
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
    // 鐠?IVPS 閸︺劑娓剁憰浣藉嚑閸戣櫣閮?RTSP/MediaMTX閿涘牅瀵岀喊鍏肩ウ閿涘妾稊鐔诲厴閸熺喎瀚婇妴?
    // Raw 娑撹崵鈷撳ù浣烘畱 enableAI=false閿涘牅绗夐悾?OSD閿涘绲炬禒宥夋付鐟?IVPS 閻垻鏁撻崣顖滄硶绾拌壈鍑犻崙鎭掆偓?
    ivps.n_fifo_count =
        (config_.enableAI || config_.isRTSPOutput || config_.isMediaMTXOutput) ? 1 : 0;
    bool needOSD = config_.enableAI && !is_osd_disabled_by_env();  // 閸欘垳鏁?AX_DISABLE_OSD=1 瀵嘲鍩楅梻婊堟瀱 OSD
    // 閸欘亙濞囬悽?1 閸?OSD 閸椻偓閸╃喍鐢崣顏呮纯閺傛媽鈹庨崡鈧崺鐕傜礉闁灝鍘ゆ径姘磥閸╃喐婀崥灞绢劄閺囧瓨鏌婄亸搴ゅ毀 IVPS 閸氬牊鍨氶懞鍗炵潌/閼规彃顢?    ivps.n_osd_rgn = needOSD ? 1 : 0;
    
    // 鐟锋﹢瀵?IVPS 闁板秶鐤?    ALOGN("[VideoStream] Stream %d IVPS config: grp=%d, size=%dx%d, fps=%d, n_fifo_count=%d, n_osd_rgn=%d, enableAI=%d", 
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
        // 閻╊喖澧?pipeline_input_e 鐏忔碍婀幓鎰返 H265 鐏忓秵鍣虫潛绋垮弳妞ょ偛鐎烽敍灞藉帥鐎瑰鍙忛梽宥囩閻?H264 闁灝鍘ょ欢銊劏婢惰鲸鏅介妴?
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

    // 閻╊喖澧犳稉宥嗘暜閹?H265 VDEC input type閿涘奔绻氶悾娆戝閹卞绲炬稉宥呮鐟箓鍣稿?pipeline閵?
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





