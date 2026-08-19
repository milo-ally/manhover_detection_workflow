#include "inference_manager.h"

#include "ai_processor.h"
#include "../../utilities/sample_log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
                                       std::vector<uint32_t>& outIndices) {
    outIndices.clear();
    std::vector<uint32_t> valid;
    valid.reserve(src.nObjSize);
    for (uint32_t i = 0; i < src.nObjSize; i++) {
        const AI_OBJ_T& o = src.objects[i];
        if (o.w > 0.f && o.h > 0.f) valid.push_back(i);
    }
    if (valid.empty()) return;

    std::stable_sort(valid.begin(), valid.end(), [&](uint32_t ia, uint32_t ib) {
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

    for (uint32_t idx : valid) {
        if (outIndices.size() >= maxRois) break;
        bool ok = true;
        for (uint32_t kept : outIndices) {
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
        uint32_t i;
    };
    std::vector<Ref> refs;
    refs.reserve(MAX_DETECT_OBJ_NUM * 2);
    for (size_t r = 0; r < results.size(); r++) {
        for (uint32_t i = 0; i < results[r].nObjSize; i++) {
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
        for (uint32_t i = 0; i < results[r].nObjSize && out->nObjSize < MAX_DETECT_OBJ_NUM; i++) {
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
    for (uint32_t i = 0; i < result->nObjSize; i++) {
        AI_OBJ_T& o = result->objects[i];
        o.x = roiX + o.x * roiW;
        o.y = roiY + o.y * roiH;
        o.w *= roiW;
        o.h *= roiH;
    }
}

bool cropFrameToROI(const AI_FRAME_T* src, float nx, float ny, float nw, float nh,
                    AI_FRAME_T* cropFrame, std::vector<char>* buf) {
    if (!src || !cropFrame || !buf || src->format != AI_FRAME_FORMAT_BGR24 || !src->data) return false;
    const int w = src->width, h = src->height;
    const int stride = src->stride > 0 ? src->stride : w * 3;
    int sx = (int)(nx * w), sy = (int)(ny * h);
    int sw = (int)(nw * w), sh = (int)(nh * h);
    if (sw <= 0 || sh <= 0) return false;
    sx = std::max(0, std::min(sx, w - 1));
    sy = std::max(0, std::min(sy, h - 1));
    if (sx + sw > w) sw = w - sx;
    if (sy + sh > h) sh = h - sy;
    if (sw <= 0 || sh <= 0) return false;

    buf->resize((size_t)sw * sh * 3);
    const unsigned char* srcData = (const unsigned char*)src->data;
    unsigned char* dst = (unsigned char*)buf->data();
    for (int row = 0; row < sh; row++)
        memcpy(dst + (size_t)row * sw * 3, srcData + (size_t)(sy + row) * stride + (size_t)sx * 3, (size_t)sw * 3);

    memset(cropFrame, 0, sizeof(AI_FRAME_T));
    cropFrame->data = buf->data();
    cropFrame->width = sw;
    cropFrame->height = sh;
    cropFrame->stride = sw * 3;
    cropFrame->format = AI_FRAME_FORMAT_BGR24;
    return true;
}

}  // namespace inference_detail

namespace {

AI_FRAME_T makeFrameFromBgr(std::vector<uint8_t>& bgr, uint32_t w, uint32_t h, uint32_t stridePix, uint32_t sz) {
    AI_FRAME_T tFrame = {0};
    tFrame.data = bgr.data();
    tFrame.width = (int)w;
    tFrame.height = (int)h;
    tFrame.stride = (int)stridePix * 3;
    tFrame.format = AI_FRAME_FORMAT_BGR24;
    (void)sz;
    return tFrame;
}

}  // namespace

InferenceManager::InferenceManager() = default;

InferenceManager::~InferenceManager() {
    std::lock_guard<std::mutex> lk(mutex_);
    stopPipelineThreads();
}

void InferenceManager::stopPipelineThreads() {
    // RK3588 当前无 overlap 流水线（与 AX650 现状一致：一律走同步 runDispatch）
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
}

bool InferenceManager::runSingle(AI_FRAME_T* frame, AI_RESULT_T* out) {
    if (engines_.empty() || !engines_[0]) return false;
    return engines_[0]->run(frame, out);
}

bool InferenceManager::runParallel(AI_FRAME_T* frame, AI_RESULT_T* out) {
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

bool InferenceManager::runSerialSync(AI_FRAME_T* frame, AI_RESULT_T* out) {
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
    thread_local std::vector<uint32_t> roiOrder;
    for (size_t stage = 1; stage < engines_.size() && engines_[stage]; stage++) {
        AI_RESULT_T nextStageMerged;
        memset(&nextStageMerged, 0, sizeof(nextStageMerged));
        AI_FRAME_T cropFrame;
        selectRoiIndicesForCascade(currentROIs, kMaxCascadeRois, kSerialRoiDedupIou, roiOrder);
        for (uint32_t b : roiOrder) {
            AI_OBJ_T& box = currentROIs.objects[b];
            if (!inference_detail::cropFrameToROI(frame, box.x, box.y, box.w, box.h, &cropFrame, &cropBuf)) continue;
            AI_RESULT_T stageResult;
            memset(&stageResult, 0, sizeof(stageResult));
            if (engines_[stage]->run(&cropFrame, &stageResult)) {
                inference_detail::transformResultToFullFrame(&stageResult, box.x, box.y, box.w, box.h);
                allResults.push_back(stageResult);
                for (uint32_t i = 0; i < stageResult.nObjSize && nextStageMerged.nObjSize < MAX_DETECT_OBJ_NUM; i++)
                    nextStageMerged.objects[nextStageMerged.nObjSize++] = stageResult.objects[i];
            }
        }
        currentROIs = nextStageMerged;
    }
    mergeSerialPrioritized(out, allResults, true);
    return true;
}

bool InferenceManager::runWithStagesSync(AI_FRAME_T* frame, AI_RESULT_T* out) {
    std::vector<size_t> firstRoundIndices;
    std::vector<AI_RESULT_T> modelResults(engines_.size());
    // 階段 0 必須在全圖首輪執行，否則「僅 ROI 階段」無前序框可裁，allResults 為空、畫面無任何框。
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
    thread_local std::vector<uint32_t> roiOrder;
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
            AI_FRAME_T cropFrame;
            selectRoiIndicesForCascade(*prevResult, kMaxCascadeRois, kSerialRoiDedupIou, roiOrder);
            for (uint32_t b : roiOrder) {
                AI_OBJ_T& box = prevResult->objects[b];
                if (!inference_detail::cropFrameToROI(frame, box.x, box.y, box.w, box.h, &cropFrame, &cropBuf)) continue;
                AI_RESULT_T stageResult;
                memset(&stageResult, 0, sizeof(stageResult));
                if (engines_[i] && engines_[i]->run(&cropFrame, &stageResult)) {
                    inference_detail::transformResultToFullFrame(&stageResult, box.x, box.y, box.w, box.h);
                    allResults.push_back(stageResult);
                    for (uint32_t k = 0; k < stageResult.nObjSize && stageMerged.nObjSize < MAX_DETECT_OBJ_NUM; k++)
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
    return false;  // RK3588 当前无 overlap 流水线（与 AX650 现状一致）
}

bool InferenceManager::runDispatch(AI_FRAME_T* frame, AI_RESULT_T* out) {
    if (engines_.empty()) return false;
    if (engines_.size() == 1) return runSingle(frame, out);
    if (mode_ == AIPipelineMode::Parallel) return runParallel(frame, out);
    return runSerialSync(frame, out);
}

bool InferenceManager::run(AI_FRAME_T* frame, AI_RESULT_T* out) {
    std::lock_guard<std::mutex> lk(mutex_);
    return runDispatch(frame, out);
}

std::shared_future<InferenceManager::FrameResult> InferenceManager::submitFrameAsync(std::vector<uint8_t> frameData,
                                                                                     uint32_t w, uint32_t h,
                                                                                     uint32_t stridePix,
                                                                                     uint32_t sz) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto prom = std::make_shared<std::promise<FrameResult>>();
    std::shared_future<FrameResult> fut = prom->get_future().share();
    std::vector<uint8_t> buf = std::move(frameData);
    AI_FRAME_T tFrame = makeFrameFromBgr(buf, w, h, stridePix, sz);
    FrameResult fr;
    fr.ok = runDispatch(&tFrame, &fr.result);
    prom->set_value(fr);
    return fut;
}
