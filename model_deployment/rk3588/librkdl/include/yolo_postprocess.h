#pragma once

#include <string>
#include <vector>
#include "ai_interface.h"

namespace rkdl {

struct TensorView {
    std::vector<int> dims;
    std::vector<float> values;
    bool nchw = false;
};

struct LetterboxTransform {
    int source_width = 0;
    int source_height = 0;
    int network_width = 0;
    int network_height = 0;
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

int decode_yolo(const std::vector<TensorView>& outputs,
                const LetterboxTransform& transform,
                float confidence_threshold,
                float nms_threshold,
                const std::string& profile,
                AI_RESULT_T* result);

}  // namespace rkdl
