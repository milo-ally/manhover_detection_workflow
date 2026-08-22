#pragma once

#include <cstdint>
#include "rk3588/frame.h"

#define MAX_DETECT_OBJ_NUM 64
#define MAX_KEYPOINTS 17

struct AI_KEYPOINT_T {
    float x = 0.0f;
    float y = 0.0f;
    float conf = 0.0f;
};

struct AI_OBJ_T {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    std::int32_t class_id = 0;
    float score = 0.0f;
    std::uint64_t track_id = 0;
    char label[32]{};
    AI_KEYPOINT_T keypoints[MAX_KEYPOINTS]{};
    std::uint32_t nKeypoints = 0;
};

struct AI_RESULT_T {
    std::uint32_t nObjSize = 0;
    AI_OBJ_T objects[MAX_DETECT_OBJ_NUM]{};
};

class IAIModel {
public:
    virtual ~IAIModel() = default;
    virtual int Init(const char* model_path) = 0;
    virtual void GetInputSize(int* width, int* height) = 0;
    virtual int Inference(const VideoFrame* frame, AI_RESULT_T* result) = 0;
    virtual void SetThresholds(float confidence, float nms) = 0;
    virtual int Deinit() = 0;
};

using CreateAIModelFunc = IAIModel* (*)();
using DestroyAIModelFunc = void (*)(IAIModel*);
