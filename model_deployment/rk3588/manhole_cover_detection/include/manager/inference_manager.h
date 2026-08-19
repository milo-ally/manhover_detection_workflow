#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

// 与 AX650 的 include/manager/inference_manager.h 同构；仅将 AX 帧类型替换为
// 平台无关的 AI_FRAME_T（Single / Parallel / Serial 調度語義一致）。

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

#include "ai_interface.h"
#include "ai_pipeline_config.h"
#include "inference_engine.h"

class AIProcessor;

namespace inference_detail {

void mergeAIResults(AI_RESULT_T* out, const std::vector<AI_RESULT_T>& results, bool prefixLabel);
void transformResultToFullFrame(AI_RESULT_T* result, float roiX, float roiY, float roiW, float roiH);
bool cropFrameToROI(const AI_FRAME_T* src, float nx, float ny, float nw, float nh,
                    AI_FRAME_T* cropFrame, std::vector<char>* buf);

}  // namespace inference_detail

class InferenceManager {
public:
    InferenceManager();
    ~InferenceManager();

    InferenceManager(const InferenceManager&) = delete;
    InferenceManager& operator=(const InferenceManager&) = delete;

    void configure(const std::vector<std::shared_ptr<AIProcessor>>& processors,
                   const std::vector<ModelStageConfig>& stages,
                   AIPipelineMode mode);

    /** 單幀同步推理 */
    bool run(AI_FRAME_T* frame, AI_RESULT_T* out);

    struct FrameResult {
        bool ok = false;
        AI_RESULT_T result{};
    };

    /**
     * 非同步提交（仅 API 兼容；RK3588 当前为同步执行并返回已就绪的 future）。
     * 資料須為完整 BGR24 幀（與 OpenCV 解碼輸出一致）。
     */
    std::shared_future<FrameResult> submitFrameAsync(std::vector<uint8_t> frameData, uint32_t w, uint32_t h,
                                                     uint32_t stridePix, uint32_t sz);

    bool pipelineOverlapActive() const;

private:
    /** 實際調度邏輯；呼叫方須已持有 mutex_ */
    bool runDispatch(AI_FRAME_T* frame, AI_RESULT_T* out);

    bool runSingle(AI_FRAME_T* frame, AI_RESULT_T* out);
    bool runParallel(AI_FRAME_T* frame, AI_RESULT_T* out);
    bool runSerialSync(AI_FRAME_T* frame, AI_RESULT_T* out);
    bool runWithStagesSync(AI_FRAME_T* frame, AI_RESULT_T* out);

    void stopPipelineThreads();

    /** 序列化 configure / run，避免熱切換模型或串並行時並發改寫 engines_ */
    mutable std::mutex mutex_;

    std::vector<std::shared_ptr<AIProcessor>> processors_;
    std::vector<std::shared_ptr<BaseInferenceEngine>> engines_;
    std::vector<ModelStageConfig> stages_;
    AIPipelineMode mode_ = AIPipelineMode::Parallel;
};

#endif  // INFERENCE_MANAGER_H
