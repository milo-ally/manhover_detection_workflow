#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "ai_interface.h"
#include "ai_pipeline_config.h"
#include "inference_engine.h"

class AIProcessor;

namespace inference_detail {

void mergeAIResults(AI_RESULT_T* out, const std::vector<AI_RESULT_T>& results, bool prefixLabel);
void transformResultToFullFrame(AI_RESULT_T* result, float roiX, float roiY, float roiW, float roiH);
bool cropFrameToROI(const AX_VIDEO_FRAME_T* src, float nx, float ny, float nw, float nh,
                    AX_VIDEO_FRAME_T* cropFrame, std::vector<char>* buf);

}  // namespace inference_detail

/**
 * 依 Stream 配置動態組裝 Single / Parallel / Serial 推理路徑。
 * Serial 且雙階段無「首輪獨立並行」需求時，可啟用內部 Worker+Queue 流水線以利幀級重疊。
 */
namespace inference_mgr_detail {
class PipelineOverlapRuntime;
}

class InferenceManager {
public:
    InferenceManager();
    ~InferenceManager();  // 定義於 .cpp：析構時安全釋放 pipeline

    InferenceManager(const InferenceManager&) = delete;
    InferenceManager& operator=(const InferenceManager&) = delete;

    void configure(const std::vector<std::shared_ptr<AIProcessor>>& processors,
                   const std::vector<ModelStageConfig>& stages,
                   AIPipelineMode mode);

    /** 單幀同步推理（沿用原 runAIInference 語義） */
    bool run(AX_VIDEO_FRAME_T* frame, AI_RESULT_T* out);

    struct FrameResult {
        bool ok = false;
        AI_RESULT_T result{};
    };

    /**
     * 非同步提交（僅在 pipelineOverlapActive()==true 時有意義：與下一幀在階段間重疊）。
     * 資料須為完整 NV12（與 AIWorker 一致）。
     */
    std::shared_future<FrameResult> submitFrameAsync(std::vector<uint8_t> frameData, uint32_t w, uint32_t h,
                                                     uint32_t stridePix, uint32_t sz);

    bool pipelineOverlapActive() const;

private:
    /** 實際調度邏輯；呼叫方須已持有 mutex_（run / submitFrameAsync 同步分支） */
    bool runDispatch(AX_VIDEO_FRAME_T* frame, AI_RESULT_T* out);

    bool runSingle(AX_VIDEO_FRAME_T* frame, AI_RESULT_T* out);
    bool runParallel(AX_VIDEO_FRAME_T* frame, AI_RESULT_T* out);
    bool runSerialSync(AX_VIDEO_FRAME_T* frame, AI_RESULT_T* out);
    bool runWithStagesSync(AX_VIDEO_FRAME_T* frame, AI_RESULT_T* out);

    void stopPipelineThreads();

    /** 序列化 configure / run / submit，避免熱切換模型或串並行時與 AI Worker 並發改寫 engines_（未定義行為 → segfault） */
    mutable std::mutex mutex_;

    std::vector<std::shared_ptr<AIProcessor>> processors_;
    std::vector<std::shared_ptr<BaseInferenceEngine>> engines_;
    std::vector<ModelStageConfig> stages_;
    AIPipelineMode mode_ = AIPipelineMode::Parallel;

    std::unique_ptr<inference_mgr_detail::PipelineOverlapRuntime> pipeline_;
};

#endif  // INFERENCE_MANAGER_H
