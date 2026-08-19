#include "video_stream.h"
#include "video_stream_manager.h"
#include "osd_renderer.h"
#include "ai_processor.h"
#include "alarm_filter.h"
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

#define MAX_CONSECUTIVE_ERRORS 5  // 允许的最大连续错误次数

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
        // 告警後端連不上時進入冷卻，避免持續堆積任務影響整體實時性
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
        if (now_ms < cooldown_until_ms_.load(std::memory_order_relaxed)) {
            return;
        }

        // 對 localhost 告警服務採用更短 timeout，避免長時間阻塞在失敗請求
        if (task.url.find("127.0.0.1") != std::string::npos ||
            task.url.find("localhost") != std::string::npos) {
            task.timeoutSec = 1;
            task.retriesLeft = 0;
        }

        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (queue_.size() >= kMaxQueueSize) {
                // 隊列滿時丟棄最舊任務，優先保證主視頻鏈路
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
                // 後端不可達時避免高頻重試與高頻日誌，降低 CPU/IO 抖動對實時鏈路的影響
                static int fail_log_count = 0;
                if (++fail_log_count % 120 == 0) {
                    ALOGW("[Alarm] 发送失败: status=%d, error=%s",
                          response.statusCode, response.error.c_str());
                }

                const bool backend_unreachable = (response.statusCode < 0);
                if (backend_unreachable) {
                    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::steady_clock::now().time_since_epoch())
                                            .count();
                    cooldown_until_ms_.store(now_ms + 30000, std::memory_order_relaxed); // 30s 熔斷
                    // 熔斷時清空隊列，避免積壓任務後續突發送出
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

// 精度對比結果上報（異步 POST /api/benchmark/result，source=edge）
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
                ALOGW("[Benchmark] 上報失敗: status=%d", response.statusCode);
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

// 全局 frame_id 用於精度對比上報（後端按時間戳對齊，此處僅需遞增）
static std::atomic<uint64_t> g_benchmark_frame_id(0);

struct BenchmarkReportCfg {
    bool enabled = false;
    std::string backendUrl = "http://127.0.0.1:8001";
    int streamId = -1; // -1: all
};

static BenchmarkReportCfg loadBenchmarkReportCfg() {
    BenchmarkReportCfg cfg;

    // 1) env 開關（向後相容）
    const char* backendUrlEnv = getenv("BACKEND_API_URL");
    if (backendUrlEnv && *backendUrlEnv) cfg.backendUrl = backendUrlEnv;

    const char* enableEnv = getenv("BENCHMARK_REPORT");
    if (enableEnv && (strcmp(enableEnv, "1") == 0 || strcmp(enableEnv, "true") == 0)) {
        cfg.enabled = true;
    }
    const char* sidEnv = getenv("BENCHMARK_REPORT_STREAM_ID");
    if (sidEnv && *sidEnv) cfg.streamId = atoi(sidEnv);

    // 2) /dev/shm 檔案開關（可由後端動態寫入，不需重啟進程）
    //    檔案格式：
    //    { "enabled": true, "backend_url": "http://x.x.x.x:8001", "stream_id": 0 }
    try {
        static std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now() - std::chrono::seconds(10);
        static BenchmarkReportCfg cached = cfg;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count() < 500) {
            return cached; // 避免每幀讀檔
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

// 上報 benchmark / 告警 / OSD 緩存 / updateAIResult（AIWorker 同步與流水線共用）
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

    if (stResult.nObjSize > 0) {
        AI_RESULT_T resultCopy;
        resultCopy.nObjSize = stResult.nObjSize;
        for (unsigned int i = 0; i < stResult.nObjSize && i < MAX_DETECT_OBJ_NUM; i++) {
            resultCopy.objects[i] = stResult.objects[i];
        }
        std::string modelName = s->getModelName();
        std::string modelPath = s->getModelPath();
        std::string effectiveModelName = modelName;
        if (effectiveModelName.empty() && !modelPath.empty()) {
            size_t lastSlash = modelPath.find_last_of("/\\");
            std::string fileName = (lastSlash != std::string::npos) ? modelPath.substr(lastSlash + 1) : modelPath;
            size_t lastDot = fileName.find_last_of(".");
            if (lastDot != std::string::npos) fileName = fileName.substr(0, lastDot);
            effectiveModelName = fileName;
        }
        std::vector<AI_OBJ_T> alarmObjects;
        std::string alarmType, modelType, severity;
        bool shouldReport = AlarmFilter::shouldReportAlarm(effectiveModelName, resultCopy, alarmType, modelType,
                                                           severity, alarmObjects);
        if (shouldReport && !alarmObjects.empty()) {
            const char* backendUrl = getenv("BACKEND_API_URL");
            const char* deviceIdStr = getenv("DEVICE_ID");
            if (!backendUrl) backendUrl = "http://127.0.0.1:8001";
            int deviceId = deviceIdStr ? std::atoi(deviceIdStr) : 1;
            nlohmann::json alarmJson;
            alarmJson["device_id"] = deviceId;
            alarmJson["model_type"] = modelType;
            alarmJson["alarm_type"] = alarmType;
            alarmJson["severity"] = severity;
            nlohmann::json detections = nlohmann::json::array();
            for (const auto& obj : alarmObjects) {
                nlohmann::json det;
                det["x"] = obj.x;
                det["y"] = obj.y;
                det["w"] = obj.w;
                det["h"] = obj.h;
                det["score"] = obj.score;
                det["label"] = obj.label;
                det["class_id"] = obj.class_id;
                if (obj.track_id > 0) det["track_id"] = static_cast<int64_t>(obj.track_id);
                if (obj.nKeypoints > 0) {
                    nlohmann::json keypoints = nlohmann::json::array();
                    for (unsigned int k = 0; k < obj.nKeypoints && k < MAX_KEYPOINTS; k++) {
                        nlohmann::json kp;
                        kp["x"] = obj.keypoints[k].x;
                        kp["y"] = obj.keypoints[k].y;
                        kp["conf"] = obj.keypoints[k].conf;
                        keypoints.push_back(kp);
                    }
                    det["keypoints"] = keypoints;
                    det["n_keypoints"] = obj.nKeypoints;
                }
                detections.push_back(det);
            }
            alarmJson["detections"] = detections;
            alarmJson["camera_id"] = streamId;
            AsyncAlarmSender::AlarmTask task;
            task.url = std::string(backendUrl) + "/api/alarms";
            task.payload = alarmJson.dump();
            task.timeoutSec = 5;
            g_alarmSender.enqueue(std::move(task));
        }
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

// 每路 AI 固定一個 worker 線程，避免每幀 std::thread().detach() 導致長時間運行後線程/記憶體資源耗盡（~10 分鐘斷流）
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
                    if (g_stream_manager) g_stream_manager->notifyAIError(streamId_, e.what());
                } else {
                    std::shared_ptr<AIProcessor> first = s->getAIProcessor();
                    if (first) {
                        first->unloadModel();
                        if (!first->loadModel(s->getModelPath(), s->getModelName())) {
                            ALOGE("Failed to reload model after exception");
                            s->setAIEnabled(false);
                        }
                    }
                }
                stResult.nObjSize = 0;
                std::lock_guard<std::mutex> lock(g_stream_manager_mutex);
                if (g_stream_manager) g_stream_manager->updateAIResult(streamId_, &stResult);
            }
        }
    }

    int streamId_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool running_ = true;
    std::unique_ptr<AIFrameJob> pending_;
    std::deque<std::shared_future<InferenceManager::FrameResult>> overlapFutures_;
};

// RAII: ensure callback count is decremented
struct CallbackGuard {
    VideoStream* s;
    explicit CallbackGuard(VideoStream* stream) : s(stream) {
        if (s) s->onCallbackEnter();
    }
    ~CallbackGuard() {
        if (s) s->onCallbackExit();
    }
};

// RAII: ensure mmap is always unmapped
struct MmapGuard {
    void* addr = nullptr;
    size_t size = 0;
    ~MmapGuard() {
        if (addr && size > 0) {
            AX_SYS_Munmap(addr, size);
        }
    }
};

static std::map<int, VideoStream*> g_stream_instances;
static std::mutex g_stream_instances_mutex;

void VideoStream::setGlobalStreamManager(VideoStreamManager* manager) {
    std::lock_guard<std::mutex> lock(g_stream_manager_mutex);
    g_stream_manager = manager;
}

// #region agent log
// Debug logging helper function
static void debug_log(const std::string& location, const std::string& message, const std::map<std::string, std::string>& data = {}) {
    std::ofstream log_file("c:\\Users\\ly0248\\Desktop\\20251231\\.cursor\\debug.log", std::ios::app);
    if (!log_file.is_open()) return;
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();
    
    log_file << "{\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()
             << ",\"location\":\"" << location << "\",\"message\":\"" << message << "\"";
    
    for (const auto& pair : data) {
        log_file << ",\"" << pair.first << "\":\"" << pair.second << "\"";
    }
    
    log_file << "}\n";
    log_file.close();
}
// #endregion

VideoStream::VideoStream(const StreamConfig& config) 
    : config_(config), 
      pipeline_({0}),
      // 參考 ax650_o：OSD Renderer 使用 pipeline 創建的 OSD Region
      // 先創建 OSDRenderer，但傳入 pipeline 指針（在 createProcessingPipeline 之後初始化）
      osdRenderer_(nullptr), 
      activeCallbacks_(0) {
  
    // 初始化pipeline结构体
    pipeline_.pipeid = config_.streamId;
    pipeline_.enable = 0;
    pipeline_.n_loog_exit = 0;
    
    resolveInputCodecConfig();
    pipeline_.m_input_type = preferredInputType_;
    
    // 配置输出类型
    if (config_.isRTSPOutput) {
        // RTSP推流
        pipeline_.m_output_type = po_rtsp_h264;
        snprintf(pipeline_.m_venc_attr.end_point, sizeof(pipeline_.m_venc_attr.end_point), 
                "%s%d", config_.rtspEndpoint.c_str(), config_.streamId);
        pipeline_.m_venc_attr.n_venc_chn = config_.streamId;
    } else if (config_.isMediaMTXOutput) {
        // MediaMTX推送
        // 每個主碼流使用不同的 VENC channel（streamId），避免多個主碼流衝突
        // 650_o 可能只支持單個主碼流，所以使用 channel 0
        // 但我們支持多個主碼流，所以需要使用不同的 channel
        pipeline_.m_output_type = po_mediamtx_h264;
        snprintf(pipeline_.m_venc_attr.end_point, sizeof(pipeline_.m_venc_attr.end_point), 
                "%s", config_.mediamtxEndpoint.c_str());
        pipeline_.m_venc_attr.n_venc_chn = config_.streamId;
    } else if (config_.isFileOutput) {
        pipeline_.m_output_type = po_venc_h264;
        pipeline_.m_venc_attr.n_venc_chn = config_.streamId;
        snprintf(pipeline_.m_venc_attr.output_file, sizeof(pipeline_.m_venc_attr.output_file),
                 "%s", config_.outputFilePath.c_str());
    } else if (config_.enableAI) {
        // AI推理输出为NV12缓冲区
        pipeline_.m_output_type = po_buff_nv12;
        pipeline_.output_func = aiInferenceCallback;
        // AI 流不使用 VENC，清除 VENC 配置
        pipeline_.m_venc_attr.n_venc_chn = -1;  // 設置為無效值，表示不使用 VENC
    }
    
    // 配置解码器属性
    pipeline_.m_vdec_attr.n_vdec_grp = config_.vdecGroup;
    
    // 配置IVPS属性
    pipeline_.m_ivps_attr.n_ivps_grp = config_.ivpsGroup;
    pipeline_.m_ivps_attr.n_ivps_width = config_.outputWidth;
    pipeline_.m_ivps_attr.n_ivps_height = config_.outputHeight;
    pipeline_.m_ivps_attr.n_ivps_fps = config_.fps;
    pipeline_.m_ivps_attr.n_fifo_count = config_.enableAI ? 1 : 0;
    // OSD 僅在啟用 AI 時掛載，方便隔離測試：關閉 AI 時無 OSD，若殘影消失則問題在 OSD
    bool needOSD = config_.enableAI && !is_osd_disabled_by_env();
    pipeline_.m_ivps_attr.n_osd_rgn = needOSD ? 1 : 0;
    
    // 注册实例指针
    {
        std::lock_guard<std::mutex> lock(g_stream_instances_mutex);
        g_stream_instances[config_.streamId] = this;
    }

    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        inferenceManager_ = std::make_unique<InferenceManager>();
        inferenceManager_->configure(aiProcessors_, config_.modelStages, config_.aiPipelineMode);
    }
}

VideoStream::~VideoStream() {
    // 1. 先从全局映射移除，防止新的回调进入
    {
        std::lock_guard<std::mutex> lock(g_stream_instances_mutex);
        // 注意：VideoStream 可能經歷 move，moved-from 物件的 destructor 不應該把
        // moved-to 物件的映射刪掉；因此只在 key 對應的指標仍是自己時才 erase。
        auto it = g_stream_instances.find(config_.streamId);
        if (it != g_stream_instances.end() && it->second == this) {
            g_stream_instances.erase(it);
        }
    }

    // 2. 停止流并等待现有回调结束
    stop();
}

// 移動構造函數
VideoStream::VideoStream(VideoStream&& other) noexcept
    : config_(std::move(other.config_)),
      pipeline_(other.pipeline_),
      running_(other.running_),
      aiProcessors_(std::move(other.aiProcessors_)),
      osdRenderer_(std::move(other.osdRenderer_)),
      outputCallback_(std::move(other.outputCallback_)),
      aiWorker_(std::move(other.aiWorker_)),
      inferenceManager_(std::move(other.inferenceManager_)) {
    other.running_ = false;
    memset(&other.pipeline_, 0, sizeof(other.pipeline_));

    // move 後更新全域映射，避免 aiInferenceCallback 找不到對應 stream
    {
        std::lock_guard<std::mutex> lock(g_stream_instances_mutex);
        g_stream_instances[config_.streamId] = this;
    }
    // 防止 moved-from 物件 destructor 誤刪映射
    other.config_.streamId = -1;
}

// 移動賦值運算符
VideoStream& VideoStream::operator=(VideoStream&& other) noexcept {
    if (this != &other) {
        stop();
        config_ = std::move(other.config_);
        pipeline_ = other.pipeline_;
        running_ = other.running_;
        aiProcessors_ = std::move(other.aiProcessors_);
        osdRenderer_ = std::move(other.osdRenderer_);
        outputCallback_ = std::move(other.outputCallback_);
        aiWorker_ = std::move(other.aiWorker_);
        inferenceManager_ = std::move(other.inferenceManager_);

        other.running_ = false;
        memset(&other.pipeline_, 0, sizeof(other.pipeline_));

        // move 後更新全域映射，避免 aiInferenceCallback 找不到對應 stream
        {
            std::lock_guard<std::mutex> lock(g_stream_instances_mutex);
            g_stream_instances[config_.streamId] = this;
        }
        // 防止 moved-from 物件 destructor 誤刪映射
        other.config_.streamId = -1;
    }
    return *this;
}

// 启动视频流
bool VideoStream::start() {
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (running_) {
        ALOGN("Stream %d already running", config_.streamId);
        return true;
    }
    
    ALOGN("Creating processing pipeline for stream %d...", config_.streamId);
    consecutiveInputErrors_ = 0;
    if (createProcessingPipeline()) {
        running_ = true;
        ALOGI("Stream %d started (IVPS group %d, VDEC group %d)", 
              config_.streamId, config_.ivpsGroup, config_.vdecGroup);
        return true;
    }
    ALOGE("Failed to create processing pipeline for stream %d", config_.streamId);
    return false;
}

// 停止视频流
void VideoStream::stop() {
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (!running_) return;

    ALOGN("Stopping stream %d...", config_.streamId);
    
    // 1. Signal pipeline logic to exit
    pipeline_.n_loog_exit = 1;

    // 2. Wait for active AI callbacks to finish
    {
        std::unique_lock<std::mutex> stopLock(stopMutex_);
        if (activeCallbacks_.load(std::memory_order_relaxed) > 0) {
            ALOGW("Stream %d waiting for %d callbacks...", config_.streamId, activeCallbacks_.load());
            // 等待直到计数归零，或超时 2s 防止死锁
            stopCv_.wait_for(stopLock, std::chrono::seconds(2), 
                [this] { return activeCallbacks_.load(std::memory_order_relaxed) == 0; });
        }
    }

    // 3. Destory pipeline resources
    destroyProcessingPipeline();
    
    if (osdRenderer_) {
        osdRenderer_->clear();
    }
    
    running_ = false;
    ALOGI("Stream %d stopped", config_.streamId);
}

// 查询流是否在运行
bool VideoStream::isRunning() const {
    std::lock_guard<std::mutex> lock(stateMutex_);
    return running_;
}

// 设置AI处理器（锁内仅swap，锁外析构）；單模型向後兼容
void VideoStream::setAIProcessor(std::unique_ptr<AIProcessor> processor) {
    std::vector<std::shared_ptr<AIProcessor>> next;
    if (processor) {
        next.push_back(std::shared_ptr<AIProcessor>(processor.release()));
    }
    std::lock_guard<std::mutex> lock(stateMutex_);
    aiProcessors_.swap(next);
    if (!inferenceManager_) inferenceManager_ = std::make_unique<InferenceManager>();
    inferenceManager_->configure(aiProcessors_, config_.modelStages, config_.aiPipelineMode);
}

// 設置多個 AI 處理器（並行多模型或串行階段）
void VideoStream::setAIProcessors(std::vector<std::unique_ptr<AIProcessor>> processors) {
    std::vector<std::shared_ptr<AIProcessor>> next;
    for (auto& p : processors) {
        if (p) next.push_back(std::shared_ptr<AIProcessor>(p.release()));
    }
    std::lock_guard<std::mutex> lock(stateMutex_);
    aiProcessors_.swap(next);
    if (!inferenceManager_) inferenceManager_ = std::make_unique<InferenceManager>();
    inferenceManager_->configure(aiProcessors_, config_.modelStages, config_.aiPipelineMode);
}

// 外部输入帧处理（如推理/转码等）
void VideoStream::processFrame(pipeline_buffer_t* buffer) {
    std::lock_guard<std::mutex> lock(pipelineMutex_);
    if (!isRunning() || !buffer) return;
    // 若 pipeline 已標記退出，不再送幀，避免 user_input 內 pipe 已從 pipeline_handle 移除導致異常
    if (pipeline_.n_loog_exit) return;

    // 商業穩定優先：禁止在進入 VDEC 前做編碼幀節流丟棄。
    // 對 H264/H265 這會破壞參考鏈，直接導致花屏與週期性卡頓。

    static int frame_log_count[64] = {0};
    const int sid_log = (config_.streamId >= 0 && config_.streamId < 64) ? config_.streamId : 0;
    if (++frame_log_count[sid_log] % 300 == 0) {
        ALOGN("[VideoStream] Processing frame for stream %d (pipeid=%d, inputSource=%s, isMediaMTXOutput=%d, venc_chn=%d)", 
              config_.streamId, pipeline_.pipeid, config_.inputSource.c_str(), 
              config_.isMediaMTXOutput ? 1 : 0, pipeline_.m_venc_attr.n_venc_chn);
    }

    // 将帧送入pipeline
    int ret = user_input(&pipeline_, 1, buffer);
    if (ret == 0) {
        consecutiveInputErrors_ = 0;
        return;
    }

    ++consecutiveInputErrors_;
    if (autoCodecFallbackEnabled_ && !autoCodecFallbackSwitched_ && consecutiveInputErrors_ >= 6) {
        if (tryAutoCodecFallbackLocked()) {
            consecutiveInputErrors_ = 0;
            return;
        }
    }
}

void VideoStream::onCallbackEnter() {
    activeCallbacks_.fetch_add(1, std::memory_order_relaxed);
}

void VideoStream::onCallbackExit() {
    if (activeCallbacks_.fetch_sub(1, std::memory_order_relaxed) == 1) {
        stopCv_.notify_all();
    }
}

// 动态更新流配置
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

// 设置AI启用状态
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

// 清除命令行模型標記，允許 Web 配置覆蓋
void VideoStream::clearCommandLineModelFlag() {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.isCommandLineModel = false;
}

// 清除 OSD 顯示
void VideoStream::clearOSD() {
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (osdRenderer_) {
        osdRenderer_->clear();
    }
}

// 设置阈值
void VideoStream::setThresholds(float conf, float nms) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    for (auto& p : aiProcessors_) {
        if (p) p->setThresholds(conf, nms);
    }
}

// 更新當前模型路徑（切換模型後必須調用，否則 getModelPath() 仍為舊值）
void VideoStream::setModelPath(const std::string& path) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.modelPath = path;
}

void VideoStream::setModelName(const std::string& name) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    config_.modelName = name;
}

// 更新多模型配置
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

// 获取模型路径（多模型時返回第一個）
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

// 在 worker 線程中執行推理的靜態函數（與主回調共享的邏輯，避免在 IVPS 線程中阻塞）
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

    // 在函數入口就記錄，確認回調被調用（每30次記錄一次）
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
            // [調試] 立即記錄找不到 stream 的情況（每30次記錄一次）
            static int not_found_count[64] = {0};
            if (++not_found_count[buf->pipeid] % 30 == 0) {
                ALOGW("[AI] aiInferenceCallback: stream not found in g_stream_instances for pipeid=%d (total instances: %zu)", 
                      buf->pipeid, g_stream_instances.size());
                // 打印所有已註冊的 pipeid
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
    
    // 記錄回調被調用（每30幀記錄一次）
    static int callback_count[64] = {0};
    if (++callback_count[stream->config_.streamId] % 300 == 0) {
        ALOGN("[AI] aiInferenceCallback called for stream %d (pipeid=%d, frame_size=%d)", 
              stream->config_.streamId, buf->pipeid, buf->n_size);
    }

    // 檢查退出標誌，提前返回避免處理已停止的流
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

    // [效能/延遲優化] 降低 AI 推理頻率到主碼流的 1/3（例如 30fps → 10fps）
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

    // [關鍵修復] 多路多模型時 AI 回調會阻塞 IVPS 線程，導致與主碼流共用的 VDEC 無法向主 IVPS 送幀，主碼流 RTP 停止。
    // 做法：複製幀後立即返回，在 worker 線程中執行推理，使 IVPS 線程盡快 ReleaseChnFrame，不阻塞 VDEC。
    AX_BOOL bMapped = AX_FALSE;
    if (!buf->p_vir && buf->p_phy) {
        buf->p_vir = AX_SYS_Mmap(buf->p_phy, buf->n_size);
        bMapped = AX_TRUE;
    }
    MmapGuard mmapGuard{ bMapped ? buf->p_vir : nullptr, static_cast<size_t>(bMapped ? buf->n_size : 0) };
    if (!buf->p_vir) return;

    std::vector<uint8_t> frameData((uint8_t*)buf->p_vir, (uint8_t*)buf->p_vir + buf->n_size);
    const uint32_t w = buf->n_width, h = buf->n_height, stridePix = buf->n_stride, sz = buf->n_size;

    // 提交給本流專用 worker 線程（固定線程，不每幀新建），佇列滿則丟幀
    if (!stream->submitAIFrame(std::move(frameData), w, h, stridePix, sz)) {
        submit_fail_count[idx]++;
        submit_ok_count[idx] = 0;
        // Worker 繁忙時自適應降頻，優先保障主碼流/VENC 路徑。
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

// 创建处理流程（pipeline）
bool VideoStream::createProcessingPipeline() {
    // 重新設置 pipeline 的輸出類型和回調函數（參考 ai_platform_RTP）
    // 這些設置必須在每次創建 pipeline 時重新設置
    if (config_.isRTSPOutput) {
        // RTSP推流
        pipeline_.m_output_type = po_rtsp_h264;
        snprintf(pipeline_.m_venc_attr.end_point, sizeof(pipeline_.m_venc_attr.end_point), 
                "%s%d", config_.rtspEndpoint.c_str(), config_.streamId);
        pipeline_.m_venc_attr.n_venc_chn = config_.streamId;
    } else if (config_.isMediaMTXOutput) {
        // MediaMTX推送
        // 每個主碼流使用不同的 VENC channel（streamId），避免多個主碼流衝突
        pipeline_.m_output_type = po_mediamtx_h264;
        snprintf(pipeline_.m_venc_attr.end_point, sizeof(pipeline_.m_venc_attr.end_point), 
                "%s", config_.mediamtxEndpoint.c_str());
        pipeline_.m_venc_attr.n_venc_chn = config_.streamId;  // 使用 streamId 作為 VENC channel，避免衝突
        ALOGN("[VideoStream] Stream %d MediaMTX endpoint set to: %s", 
              config_.streamId, config_.mediamtxEndpoint.c_str());
    } else if (config_.isFileOutput) {
        pipeline_.m_output_type = po_venc_h264;
        pipeline_.m_venc_attr.n_venc_chn = config_.streamId;
        snprintf(pipeline_.m_venc_attr.output_file, sizeof(pipeline_.m_venc_attr.output_file),
                 "%s", config_.outputFilePath.c_str());
        ALOGN("[VideoStream] Stream %d offline H.264 output: %s",
              config_.streamId, config_.outputFilePath.c_str());
    } else if (config_.enableAI) {
        // AI推理输出为NV12缓冲区
        pipeline_.m_output_type = po_buff_nv12;
        pipeline_.output_func = aiInferenceCallback;  // 重新設置回調函數
        // AI 流不使用 VENC，清除 VENC 配置
        pipeline_.m_venc_attr.n_venc_chn = -1;  // 設置為無效值，表示不使用 VENC
        if (!aiWorker_) aiWorker_ = std::make_unique<AIWorker>(config_.streamId);
        ALOGN("[VideoStream] Setting output_func for AI stream %d", config_.streamId);
    }
    
    // 設置 pipeline ID（必須在 create_pipeline 之前設置）
    pipeline_.pipeid = config_.streamId;
    
    // 設置輸入類型（支援 inputCodec=h264/h265/auto）
    pipeline_.m_input_type = preferredInputType_;
    
    // 按当前配置初始化pipeline
    configureIVPS();
    configureVDEC();
    // AI 流使用 po_buff_nv12，不需要 VENC
    // 只有 RTSP 和 MediaMTX 輸出才需要 VENC
    if (config_.isRTSPOutput || config_.isMediaMTXOutput || config_.isFileOutput) {
        configureVENC();
    } else if (config_.enableAI) {
        // AI 流：確保不配置 VENC（使用 po_buff_nv12）
        // 清除任何可能的 VENC 配置
        pipeline_.m_venc_attr.n_venc_chn = -1;  // 設置為無效值，表示不使用 VENC
    }
    
    // 啟用 pipeline（必須在 create_pipeline 之前設置）
    pipeline_.enable = 1;
    pipeline_.n_loog_exit = 0;  // 重置退出標誌（參考 ai_platform_RTP）
    
    // 启动pipeline
    int ret = create_pipeline(&pipeline_);
    if (ret != 0) {
        ALOGE("Failed to create pipeline %d", config_.streamId);
        pipeline_.enable = 0;  // 創建失敗時重置
        return false;
    }
    
    // 判斷是否需要 OSD：僅在啟用 AI 時掛載，關閉 AI 時可測試是否為 OSD 導致殘影
    bool needOSD = config_.enableAI && !is_osd_disabled_by_env();
    if (needOSD && !osdRenderer_) {
        osdRenderer_ = make_unique<OSDRenderer>(&pipeline_);
    }
    
    // 初始化OSD
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
    // 讓 IVPS 在需要輸出給 RTSP/MediaMTX（主碼流）時也能啟動。
    // Raw 主碼流的 enableAI=false（不畫 OSD）但仍需要 IVPS 產生可編碼輸出。
    ivps.n_fifo_count =
        (config_.enableAI || config_.isRTSPOutput || config_.isMediaMTXOutput) ? 1 : 0;
    bool needOSD = config_.enableAI && !is_osd_disabled_by_env();  // 可由 AX_DISABLE_OSD=1 強制關閉 OSD
    // 只使用 1 個 OSD 區域並只更新該區域，避免多區域未同步更新導致 IVPS 合成花屏/色塊
    ivps.n_osd_rgn = needOSD ? 1 : 0;
    
    // 記錄 IVPS 配置
    ALOGN("[VideoStream] Stream %d IVPS config: grp=%d, size=%dx%d, fps=%d, n_fifo_count=%d, n_osd_rgn=%d, enableAI=%d", 
          config_.streamId, ivps.n_ivps_grp, ivps.n_ivps_width, ivps.n_ivps_height, 
          ivps.n_ivps_fps, ivps.n_fifo_count, ivps.n_osd_rgn, config_.enableAI ? 1 : 0);
}

void VideoStream::configureVDEC() {
    pipeline_.m_vdec_attr.n_vdec_grp = config_.vdecGroup;
}

void VideoStream::resolveInputCodecConfig() {
    std::string codec = toLowerCodec(config_.inputCodec);
    autoCodecFallbackEnabled_ = false;
    autoCodecFallbackSwitched_ = false;
    if (!codec.empty() && codec != "h264" && codec != "auto") {
        // 目前 pipeline_input_e 尚未提供 H265 對應輸入類型，先安全降級為 H264 避免編譯失敗。
        ALOGW("[VideoStream] Stream %d input codec '%s' is unsupported; only H264 is supported",
              config_.streamId, codec.c_str());
    }
    preferredInputType_ = pi_vdec_h264;
}

bool VideoStream::tryAutoCodecFallbackLocked() {
    if (preferredInputType_ != pi_vdec_h264) {
        return false;
    }

    // 目前不支援 H265 VDEC input type，保留狀態但不嘗試重建 pipeline。
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



