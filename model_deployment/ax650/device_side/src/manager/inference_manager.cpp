#include "inference_manager.h"

#include "ai_processor.h"
#include "../../utilities/sample_log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <future>
#include <utility>
#include <vector>

namespace {

/** 串行 ROI 子推理上限：平衡延遲與召回 */
constexpr size_t kMaxCascadeRois = 16;
/** 與已選 ROI 重疊超過此 IoU 的較低分框不再送二階，降低 NPU 重複計算 */
constexpr float kSerialRoiDedupIou = 0.65f;

static float iouNorm(const AI_OBJ_T& a, const AI_OBJ_T& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.w, b.x + b.w);
    const float y2 = std::min(a.y + a.h, b.y + b.h);
    const float iw = std::max(0.f, x2 - x1);
    const float ih = std::max(0.f, y2 - y1);
    const float inter = iw * ih;
    const float u = a.w * a.h + b.w * b.h - inter;
    return u > 1e-6f ? inter / u : 0.f;
}

/**
 * 依 score 由高到低選 ROI；可選 IoU 去重（商業場景：優先算高置信、少算重疊裁切）。
 */
static void selectRoiIndicesForCascade(const AI_RESULT_T& src, size_t maxRois, float dedupIou,
                                       std::vector<AX_U32>& outIndices) {
    outIndices.clear();
    std::vector<AX_U32> valid;
    valid.reserve(src.nObjSize);
    for (AX_U32 i = 0; i < src.nObjSize; i++) {
        const AI_OBJ_T& o = src.objects[i];
        if (o.w > 0.f && o.h > 0.f) valid.push_back(i);
    }
    if (valid.empty()) return;

    std::stable_sort(valid.begin(), valid.end(), [&](AX_U32 ia, AX_U32 ib) {
        float sa = src.objects[ia].score;
        float sb = src.objects[ib].score;
        if (!std::isfinite(sa)) sa = 0.f;
        if (!std::isfinite(sb)) sb = 0.f;
        if (sa != sb) return sa > sb;
        return ia < ib;
    });

    if (dedupIou >= 1.f) {
        for (size_t k = 0; k < valid.size() && outIndices.size() < maxRois; k++) outIndices.push_back(valid[k]);
        return;
    }

    for (AX_U32 idx : valid) {
        if (outIndices.size() >= maxRois) break;
        bool ok = true;
        for (AX_U32 kept : outIndices) {
            if (iouNorm(src.objects[idx], src.objects[kept]) > dedupIou) {
                ok = false;
                break;
            }
        }
        if (ok) outIndices.push_back(idx);
    }
}

/** 串行管線輸出：全域依 score 填滿緩衝，避免前序階段佔滿名額導致高置信後階結果被截斷 */
static void mergeSerialPrioritized(AI_RESULT_T* out, const std::vector<AI_RESULT_T>& results, bool prefixLabel) {
    struct Ref {
        float score;
        size_t r;
        AX_U32 i;
    };
    std::vector<Ref> refs;
    refs.reserve(MAX_DETECT_OBJ_NUM * 2);
    for (size_t r = 0; r < results.size(); r++) {
        for (AX_U32 i = 0; i < results[r].nObjSize; i++) {
            float s = results[r].objects[i].score;
            if (!std::isfinite(s)) s = 0.f;
            refs.push_back({s, r, i});
        }
    }
    std::stable_sort(refs.begin(), refs.end(), [](const Ref& a, const Ref& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.r != b.r) return a.r < b.r;
        return a.i < b.i;
    });

    memset(out, 0, sizeof(AI_RESULT_T));
    out->nObjSize = 0;
    for (const Ref& ref : refs) {
        if (out->nObjSize >= MAX_DETECT_OBJ_NUM) break;
        out->objects[out->nObjSize] = results[ref.r].objects[ref.i];
        if (prefixLabel && results.size() > 1) {
            char prefixed[48];
            snprintf(prefixed, sizeof(prefixed), "[%zu]%.31s", ref.r, results[ref.r].objects[ref.i].label);
            strncpy(out->objects[out->nObjSize].label, prefixed, sizeof(out->objects[out->nObjSize].label) - 1);
            out->objects[out->nObjSize].label[sizeof(out->objects[out->nObjSize].label) - 1] = '\0';
        }
        out->nObjSize++;
    }
}

}  // namespace

namespace inference_detail {

void mergeAIResults(AI_RESULT_T* out, const std::vector<AI_RESULT_T>& results, bool prefixLabel) {
    memset(out, 0, sizeof(AI_RESULT_T));
    out->nObjSize = 0;
    for (size_t r = 0; r < results.size() && out->nObjSize < MAX_DETECT_OBJ_NUM; r++) {
        for (AX_U32 i = 0; i < results[r].nObjSize && out->nObjSize < MAX_DETECT_OBJ_NUM; i++) {
            out->objects[out->nObjSize] = results[r].objects[i];
            if (prefixLabel && results.size() > 1) {
                char prefixed[48];
                snprintf(prefixed, sizeof(prefixed), "[%zu]%.31s", r, results[r].objects[i].label);
                strncpy(out->objects[out->nObjSize].label, prefixed, sizeof(out->objects[out->nObjSize].label) - 1);
                out->objects[out->nObjSize].label[sizeof(out->objects[out->nObjSize].label) - 1] = '\0';
            }
            out->nObjSize++;
        }
    }
}

void transformResultToFullFrame(AI_RESULT_T* result, float roiX, float roiY, float roiW, float roiH) {
    for (AX_U32 i = 0; i < result->nObjSize; i++) {
        AI_OBJ_T& o = result->objects[i];
        o.x = roiX + o.x * roiW;
        o.y = roiY + o.y * roiH;
        o.w *= roiW;
        o.h *= roiH;
    }
}

bool cropFrameToROI(const AX_VIDEO_FRAME_T* src, float nx, float ny, float nw, float nh,
                    AX_VIDEO_FRAME_T* cropFrame, std::vector<char>* buf) {
    if (!src || !cropFrame || !buf || src->enImgFormat != AX_FORMAT_YUV420_SEMIPLANAR) return false;
    AX_U32 w = src->u32Width, h = src->u32Height;
    AX_U32 stride = src->u32PicStride[0] ? src->u32PicStride[0] : w;
    int sx = (int)(nx * w), sy = (int)(ny * h);
    int sw = (int)(nw * w), sh = (int)(nh * h);
    if (sw <= 0 || sh <= 0) return false;
    sx = std::max(0, std::min(sx, (int)w - 1));
    sy = std::max(0, std::min(sy, (int)h - 1));
    if (sx + sw > (int)w) sw = w - sx;
    if (sy + sh > (int)h) sh = h - sy;
    if (sw <= 0 || sh <= 0) return false;
    // NV12 / OpenCV COLOR_YUV2BGR_NV12：luma 寬高須為偶數，且 rows=h*3/2 須滿足內部對齊（否則
    // sz.width%2==0 && sz.height%3==0 斷言失敗）。串行 ROI 裁切由浮點框換算易出現奇數寬高。
    sx = (sx / 2) * 2;
    sy = (sy / 2) * 2;
    sw = (sw / 2) * 2;
    sh = (sh / 2) * 2;
    if (sw < 2 || sh < 2) return false;
    if (sx + sw > (int)w) sw = ((int)w - sx) / 2 * 2;
    if (sy + sh > (int)h) sh = ((int)h - sy) / 2 * 2;
    if (sw < 2 || sh < 2) return false;
    size_t cropYSize = (size_t)sw * sh;
    size_t cropUVSize = (size_t)sw * (sh / 2);
    buf->resize(cropYSize + cropUVSize);
    const unsigned char* srcY = (const unsigned char*)src->u64VirAddr[0];
    const unsigned char* srcUV = srcY + (size_t)stride * h;
    unsigned char* dstY = (unsigned char*)buf->data();
    unsigned char* dstUV = dstY + cropYSize;
    for (int row = 0; row < sh; row++)
        memcpy(dstY + row * sw, srcY + (sy + row) * stride + sx, sw);
    for (int row = 0; row < sh / 2; row++)
        memcpy(dstUV + row * sw, srcUV + ((sy / 2) + row) * stride + sx, sw);
    memset(cropFrame, 0, sizeof(AX_VIDEO_FRAME_T));
    cropFrame->u32Width = (AX_U32)sw;
    cropFrame->u32Height = (AX_U32)sh;
    cropFrame->u32FrameSize = (AX_U32)buf->size();
    cropFrame->u64VirAddr[0] = (AX_U64)buf->data();
    cropFrame->u32PicStride[0] = (AX_U32)sw;
    cropFrame->enImgFormat = AX_FORMAT_YUV420_SEMIPLANAR;
    return true;
}

}  // namespace inference_detail

namespace inference_mgr_detail {

constexpr size_t kPipelineQueueDepth = 2;

AX_VIDEO_FRAME_T makeFrameFromNv12(std::vector<uint8_t>& nv12, uint32_t w, uint32_t h, uint32_t stridePix, uint32_t sz) {
    AX_VIDEO_FRAME_T tFrame = {0};
    tFrame.u32Width = w;
    tFrame.u32Height = h;
    tFrame.u32FrameSize = sz;
    tFrame.u64PhyAddr[0] = 0;
    tFrame.u64VirAddr[0] = (AX_U64)nv12.data();
    tFrame.u32PicStride[0] = stridePix;
    tFrame.enImgFormat = AX_FORMAT_YUV420_SEMIPLANAR;
    return tFrame;
}

template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t cap) : cap_(cap) {}

    void setClosed(bool c) {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = c;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    void wakeAll() {
        std::lock_guard<std::mutex> lk(mu_);
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    void push(T item) {
        std::unique_lock<std::mutex> lk(mu_);
        not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
        if (closed_) return;
        q_.push_back(std::move(item));
        not_empty_.notify_one();
    }

    bool pop(T& out, std::atomic<bool>& running) {
        std::unique_lock<std::mutex> lk(mu_);
        not_empty_.wait(lk, [&] { return !q_.empty() || !running.load(std::memory_order_relaxed) || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return true;
    }

private:
    std::deque<T> q_;
    std::mutex mu_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    size_t cap_ = 2;
    bool closed_ = false;
};

struct PipelineJob {
    std::vector<uint8_t> nv12;
    uint32_t w = 0, h = 0, stridePix = 0, sz = 0;
    AI_RESULT_T r0{};
    std::shared_ptr<std::promise<InferenceManager::FrameResult>> done;
};

/**
 * 雙階段流水線：Stage0 與 Stage1 以 Queue 解耦，使兩幀在時間上重疊（幀 N 的 Stage1 與幀 N+1 的 Stage0）。
 * 同一階段內仍為單執行緒呼叫引擎，避免同一 IAIModel 實例並行推理。
 */
class PipelineOverlapRuntime {
public:
    PipelineOverlapRuntime(std::shared_ptr<BaseInferenceEngine> e0, std::shared_ptr<BaseInferenceEngine> e1)
        : e0_(std::move(e0)), e1_(std::move(e1)) {
        running_ = true;
        t0_ = std::thread(&PipelineOverlapRuntime::stage0Loop, this);
        t1_ = std::thread(&PipelineOverlapRuntime::stage1Loop, this);
    }

    ~PipelineOverlapRuntime() {
        running_ = false;
        q0_.wakeAll();
        q1_.wakeAll();
        if (t0_.joinable()) t0_.join();
        if (t1_.joinable()) t1_.join();
        q0_.setClosed(true);
        q1_.setClosed(true);
    }

    std::shared_future<InferenceManager::FrameResult> submitAsync(std::vector<uint8_t> frameData, uint32_t w,
                                                                   uint32_t h, uint32_t stridePix, uint32_t sz) {
        auto prom = std::make_shared<std::promise<InferenceManager::FrameResult>>();
        std::shared_future<InferenceManager::FrameResult> fut = prom->get_future().share();
        auto job = std::make_shared<PipelineJob>();
        job->nv12 = std::move(frameData);
        job->w = w;
        job->h = h;
        job->stridePix = stridePix;
        job->sz = sz;
        job->done = std::move(prom);
        q0_.push(std::move(job));
        return fut;
    }

private:
    void stage0Loop() {
        while (running_) {
            std::shared_ptr<PipelineJob> job;
            if (!q0_.pop(job, running_)) break;
            if (!job) break;
            if (!job->done) continue;

            if (!e0_) {
                InferenceManager::FrameResult frFail;
                frFail.ok = false;
                job->done->set_value(frFail);
                continue;
            }
            memset(&job->r0, 0, sizeof(job->r0));
            AX_VIDEO_FRAME_T tFrame = makeFrameFromNv12(job->nv12, job->w, job->h, job->stridePix, job->sz);
            if (!e0_->run(&tFrame, &job->r0)) {
                InferenceManager::FrameResult frFail;
                frFail.ok = false;
                job->done->set_value(frFail);
                continue;
            }
            q1_.push(std::move(job));
        }
    }

    void stage1Loop() {
        while (running_) {
            std::shared_ptr<PipelineJob> job;
            if (!q1_.pop(job, running_)) break;
            if (!job) break;
            if (!job->done) continue;

            InferenceManager::FrameResult fr;
            if (!e1_) {
                fr.ok = false;
                job->done->set_value(fr);
                continue;
            }

            AX_VIDEO_FRAME_T fullFrame = makeFrameFromNv12(job->nv12, job->w, job->h, job->stridePix, job->sz);
            AI_RESULT_T currentROIs = job->r0;

            std::vector<AI_RESULT_T> allResults;
            allResults.push_back(currentROIs);

            AI_RESULT_T nextStageMerged;
            memset(&nextStageMerged, 0, sizeof(nextStageMerged));
            thread_local std::vector<char> cropBuf;
            cropBuf.clear();
            thread_local std::vector<AX_U32> roiOrder;
            selectRoiIndicesForCascade(currentROIs, kMaxCascadeRois, kSerialRoiDedupIou, roiOrder);
            AX_VIDEO_FRAME_T cropFrame;
            for (AX_U32 b : roiOrder) {
                AI_OBJ_T& box = currentROIs.objects[b];
                if (!inference_detail::cropFrameToROI(&fullFrame, box.x, box.y, box.w, box.h, &cropFrame, &cropBuf))
                    continue;
                AI_RESULT_T stageResult;
                memset(&stageResult, 0, sizeof(stageResult));
                if (e1_->run(&cropFrame, &stageResult)) {
                    inference_detail::transformResultToFullFrame(&stageResult, box.x, box.y, box.w, box.h);
                    allResults.push_back(stageResult);
                    for (AX_U32 i = 0; i < stageResult.nObjSize && nextStageMerged.nObjSize < MAX_DETECT_OBJ_NUM; i++)
                        nextStageMerged.objects[nextStageMerged.nObjSize++] = stageResult.objects[i];
                }
            }
            (void)nextStageMerged;

            AI_RESULT_T merged{};
            mergeSerialPrioritized(&merged, allResults, true);
            fr.ok = true;
            fr.result = merged;
            job->done->set_value(fr);
        }
    }

    std::shared_ptr<BaseInferenceEngine> e0_;
    std::shared_ptr<BaseInferenceEngine> e1_;
    BlockingQueue<std::shared_ptr<PipelineJob>> q0_{kPipelineQueueDepth};
    BlockingQueue<std::shared_ptr<PipelineJob>> q1_{kPipelineQueueDepth};
    std::atomic<bool> running_{false};
    std::thread t0_, t1_;
};

}  // namespace inference_mgr_detail

InferenceManager::InferenceManager() = default;

InferenceManager::~InferenceManager() {
    std::lock_guard<std::mutex> lk(mutex_);
    pipeline_.reset();
}

void InferenceManager::stopPipelineThreads() {
    pipeline_.reset();
}

void InferenceManager::configure(const std::vector<std::shared_ptr<AIProcessor>>& processors,
                                 const std::vector<ModelStageConfig>& stages, AIPipelineMode mode) {
    std::lock_guard<std::mutex> lk(mutex_);
    stopPipelineThreads();
    processors_ = processors;
    stages_ = stages;
    mode_ = mode;
    engines_.clear();
    engines_.reserve(processors_.size());
    for (const auto& p : processors_) {
        if (p)
            engines_.push_back(std::make_shared<ProcessorInferenceEngine>(p));
        else
            engines_.push_back(nullptr);
    }
    // 幀級 overlap 管線（PipelineOverlapRuntime）在動態 configure 與 AIWorker 並發時曾導致崩潰／死鎖風險。
    // 在具備與 Worker 協調的「先排空 future 再切換」之前，一律走同步 runDispatch（仍保留串行 ROI 與並行多模型邏輯）。
    // 若需重新啟用：在 VideoStream/AIWorker 於 configure 前排空 overlapFutures_，並驗證 PipelineOverlapRuntime 停機順序。
}

bool InferenceManager::runSingle(AX_VIDEO_FRAME_T* frame, AI_RESULT_T* out) {
    if (engines_.empty() || !engines_[0]) return false;
    return engines_[0]->run(frame, out);
}

bool InferenceManager::runParallel(AX_VIDEO_FRAME_T* frame, AI_RESULT_T* out) {
    std::vector<AI_RESULT_T> perModelResults(engines_.size());
    std::vector<std::future<bool>> futures;
    futures.reserve(engines_.size());
    for (size_t i = 0; i < engines_.size(); i++) {
        futures.push_back(std::async(std::launch::async, [this, frame, &perModelResults, i]() {
            memset(&perModelResults[i], 0, sizeof(AI_RESULT_T));
            if (!engines_[i]) return false;
            return engines_[i]->run(frame, &perModelResults[i]);
        }));
    }
    for (size_t i = 0; i < futures.size(); i++) {
        (void)futures[i].get();
    }
    inference_detail::mergeAIResults(out, perModelResults, true);
    return true;
}

bool InferenceManager::runSerialSync(AX_VIDEO_FRAME_T* frame, AI_RESULT_T* out) {
    if (!stages_.empty() && stages_.size() == engines_.size()) {
        return runWithStagesSync(frame, out);
    }
    if (engines_.empty() || !engines_[0]) return false;

    AI_RESULT_T currentROIs;
    memset(&currentROIs, 0, sizeof(currentROIs));
    if (!engines_[0]->run(frame, &currentROIs)) return false;

    std::vector<AI_RESULT_T> allResults;
    allResults.push_back(currentROIs);

    thread_local std::vector<char> cropBuf;
    thread_local std::vector<AX_U32> roiOrder;
    for (size_t stage = 1; stage < engines_.size() && engines_[stage]; stage++) {
        AI_RESULT_T nextStageMerged;
        memset(&nextStageMerged, 0, sizeof(nextStageMerged));
        AX_VIDEO_FRAME_T cropFrame;
        selectRoiIndicesForCascade(currentROIs, kMaxCascadeRois, kSerialRoiDedupIou, roiOrder);
        for (AX_U32 b : roiOrder) {
            AI_OBJ_T& box = currentROIs.objects[b];
            if (!inference_detail::cropFrameToROI(frame, box.x, box.y, box.w, box.h, &cropFrame, &cropBuf)) continue;
            AI_RESULT_T stageResult;
            memset(&stageResult, 0, sizeof(stageResult));
            if (engines_[stage]->run(&cropFrame, &stageResult)) {
                inference_detail::transformResultToFullFrame(&stageResult, box.x, box.y, box.w, box.h);
                allResults.push_back(stageResult);
                for (AX_U32 i = 0; i < stageResult.nObjSize && nextStageMerged.nObjSize < MAX_DETECT_OBJ_NUM; i++)
                    nextStageMerged.objects[nextStageMerged.nObjSize++] = stageResult.objects[i];
            }
        }
        currentROIs = nextStageMerged;
    }
    mergeSerialPrioritized(out, allResults, true);
    return true;
}

bool InferenceManager::runWithStagesSync(AX_VIDEO_FRAME_T* frame, AI_RESULT_T* out) {
    std::vector<size_t> firstRoundIndices;
    std::vector<AI_RESULT_T> modelResults(engines_.size());
    // 階段 0 必須在全圖首輪執行，否則「僅 ROI 階段」無前序框可裁，allResults 為空、畫面無任何框。
    // 若 JSON/舊客戶端誤將第一個模型標成 roi_from_previous，舊邏輯會讓 firstRoundIndices 為空。
    for (size_t i = 0; i < stages_.size(); i++) {
        const bool runFullFrame =
            (i == 0) || stages_[i].independent || !stages_[i].roiFromPrevious;
        if (runFullFrame) firstRoundIndices.push_back(i);
    }
    if (firstRoundIndices.size() <= 1) {
        for (size_t idx : firstRoundIndices) {
            memset(&modelResults[idx], 0, sizeof(AI_RESULT_T));
            if (engines_[idx]) (void)engines_[idx]->run(frame, &modelResults[idx]);
        }
    } else {
        std::vector<std::future<bool>> firstFutures;
        firstFutures.reserve(firstRoundIndices.size());
        for (size_t idx : firstRoundIndices) {
            firstFutures.push_back(std::async(std::launch::async, [this, frame, &modelResults, idx]() {
                memset(&modelResults[idx], 0, sizeof(AI_RESULT_T));
                if (!engines_[idx]) return false;
                return engines_[idx]->run(frame, &modelResults[idx]);
            }));
        }
        for (auto& f : firstFutures) (void)f.get();
    }

    std::vector<AI_RESULT_T> allResults;
    for (size_t idx : firstRoundIndices) {
        if (modelResults[idx].nObjSize > 0) allResults.push_back(modelResults[idx]);
    }

    thread_local std::vector<char> cropBuf;
    thread_local std::vector<AX_U32> roiOrder;
    for (size_t i = 0; i < stages_.size(); i++) {
        if (stages_[i].roiFromPrevious && !stages_[i].independent) {
            AI_RESULT_T* prevResult = nullptr;
            for (int j = (int)i - 1; j >= 0; j--) {
                if (modelResults[j].nObjSize > 0) {
                    prevResult = &modelResults[j];
                    break;
                }
            }
            if (!prevResult || prevResult->nObjSize == 0) continue;
            AI_RESULT_T stageMerged;
            memset(&stageMerged, 0, sizeof(stageMerged));
            AX_VIDEO_FRAME_T cropFrame;
            selectRoiIndicesForCascade(*prevResult, kMaxCascadeRois, kSerialRoiDedupIou, roiOrder);
            for (AX_U32 b : roiOrder) {
                AI_OBJ_T& box = prevResult->objects[b];
                if (!inference_detail::cropFrameToROI(frame, box.x, box.y, box.w, box.h, &cropFrame, &cropBuf)) continue;
                AI_RESULT_T stageResult;
                memset(&stageResult, 0, sizeof(stageResult));
                if (engines_[i] && engines_[i]->run(&cropFrame, &stageResult)) {
                    inference_detail::transformResultToFullFrame(&stageResult, box.x, box.y, box.w, box.h);
                    allResults.push_back(stageResult);
                    for (AX_U32 k = 0; k < stageResult.nObjSize && stageMerged.nObjSize < MAX_DETECT_OBJ_NUM; k++)
                        stageMerged.objects[stageMerged.nObjSize++] = stageResult.objects[k];
                }
            }
            modelResults[i] = stageMerged;
        }
    }
    mergeSerialPrioritized(out, allResults, true);
    return true;
}

bool InferenceManager::pipelineOverlapActive() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<bool>(pipeline_);
}

bool InferenceManager::runDispatch(AX_VIDEO_FRAME_T* frame, AI_RESULT_T* out) {
    if (engines_.empty()) return false;
    if (engines_.size() == 1) return runSingle(frame, out);
    if (mode_ == AIPipelineMode::Parallel) return runParallel(frame, out);
    return runSerialSync(frame, out);
}

bool InferenceManager::run(AX_VIDEO_FRAME_T* frame, AI_RESULT_T* out) {
    std::lock_guard<std::mutex> lk(mutex_);
    return runDispatch(frame, out);
}

std::shared_future<InferenceManager::FrameResult> InferenceManager::submitFrameAsync(std::vector<uint8_t> frameData,
                                                                                       uint32_t w, uint32_t h,
                                                                                       uint32_t stridePix,
                                                                                       uint32_t sz) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (pipeline_) {
        return pipeline_->submitAsync(std::move(frameData), w, h, stridePix, sz);
    }
    auto prom = std::make_shared<std::promise<FrameResult>>();
    std::shared_future<FrameResult> fut = prom->get_future().share();
    std::vector<uint8_t> buf = std::move(frameData);
    AX_VIDEO_FRAME_T tFrame = inference_mgr_detail::makeFrameFromNv12(buf, w, h, stridePix, sz);
    FrameResult fr;
    fr.ok = runDispatch(&tFrame, &fr.result);
    prom->set_value(fr);
    return fut;
}
