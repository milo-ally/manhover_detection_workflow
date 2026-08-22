#pragma once

#include <memory>
#include <string>
#include <vector>
#include "ai_interface.h"
#include "yolo_postprocess.h"

namespace rkdl {

class RknnModel final : public IAIModel {
public:
    explicit RknnModel(std::string profile = "default");
    ~RknnModel() override;

    int Init(const char* model_path) override;
    void GetInputSize(int* width, int* height) override;
    int Inference(const VideoFrame* frame, AI_RESULT_T* result) override;
    void SetThresholds(float confidence, float nms) override;
    int Deinit() override;

    int RunRaw(const VideoFrame* frame, std::vector<TensorView>* outputs,
               LetterboxTransform* transform = nullptr);

    const std::string& lastError() const;
    bool available() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rkdl
