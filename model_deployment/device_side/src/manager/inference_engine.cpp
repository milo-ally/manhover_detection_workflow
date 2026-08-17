#include "inference_engine.h"
#include "ai_processor.h"

ProcessorInferenceEngine::ProcessorInferenceEngine(std::shared_ptr<AIProcessor> processor)
    : processor_(std::move(processor)) {}

bool ProcessorInferenceEngine::run(const AX_VIDEO_FRAME_T* frame, AI_RESULT_T* result) {
    if (!processor_ || !frame || !result) return false;
    return processor_->processFrame(frame, result);
}
