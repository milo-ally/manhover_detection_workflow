#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

// 与 AX650 的 include/manager/inference_engine.h 同构；仅将 AX 帧类型替换为
// 平台无关的 AI_FRAME_T。

#include <memory>
#include <string>
#include "ai_interface.h"

class AIProcessor;

/**
 * 統一單一模型推理入口，便於 InferenceManager 以任務圖方式調度（Single / Parallel / Pipeline）。
 * 實作可包裝既有 AIProcessor，後續可替換為零拷貝 / 多輸入等進階引擎。
 */
class BaseInferenceEngine {
public:
    virtual ~BaseInferenceEngine() = default;

    virtual bool run(const AI_FRAME_T* frame, AI_RESULT_T* result) = 0;

    /** 用於除錯或管線選路；可為空 */
    virtual const char* name() const { return "BaseInferenceEngine"; }
};

/** 適配現有 AIProcessor（動態庫插件 + IAIModel） */
class ProcessorInferenceEngine : public BaseInferenceEngine {
public:
    explicit ProcessorInferenceEngine(std::shared_ptr<AIProcessor> processor);
    bool run(const AI_FRAME_T* frame, AI_RESULT_T* result) override;
    const char* name() const override { return "ProcessorInferenceEngine"; }
    std::shared_ptr<AIProcessor> processor() const { return processor_; }

private:
    std::shared_ptr<AIProcessor> processor_;
};

#endif  // INFERENCE_ENGINE_H
