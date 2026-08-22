#include "ai_interface.h"
#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include "letterbox_utils.hpp"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

// manhole_cover model contract: output [1, 9, 8400],
// channels = cx, cy, w, h, good, broke, lose, uncovered, circle.
namespace {
constexpr int kClassCount = 5;
constexpr int kInputSize = 640;
constexpr int kAnchorCount = 8400;
constexpr float kDefaultConf = 0.25f;
constexpr float kDefaultNms = 0.45f;
const char* kLabels[kClassCount] = {"good", "broke", "lose", "uncovered", "circle"};

float env_float(const char* name, float fallback) {
    const char* value = std::getenv(name);
    return value && *value ? static_cast<float>(std::atof(value)) : fallback;
}

std::vector<char> load_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0) return {};
    file.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    return file.read(data.data(), size) ? data : std::vector<char>{};
}
}

class ManholeCoverModel final : public IAIModel {
public:
    ~ManholeCoverModel() override { Deinit(); }

    int Init(const char* model_path) override {
        if (!model_path || !*model_path) return -1;
        conf_ = env_float("MANHOLE_CONF_THRESH", env_float("MODEL_CONF_THRESH", kDefaultConf));
        nms_ = env_float("MANHOLE_NMS_THRESH", env_float("MODEL_NMS_THRESH", kDefaultNms));
        const std::vector<char> model = load_file(model_path);
        if (model.empty()) { std::fprintf(stderr, "[ManholeCover] cannot read model: %s\n", model_path); return -1; }
        if (AX_ENGINE_CreateHandle(&handle_, model.data(), model.size()) != 0) return -1;
        AX_ENGINE_GetIOInfo(handle_, &io_info_);
        if (!io_info_ || io_info_->nInputSize == 0 || io_info_->nOutputSize == 0) return -1;

        io_.nInputSize = io_info_->nInputSize;
        io_.pInputs = new AX_ENGINE_IO_BUFFER_T[io_.nInputSize]{};
        io_.nOutputSize = io_info_->nOutputSize;
        io_.pOutputs = new AX_ENGINE_IO_BUFFER_T[io_.nOutputSize]{};
        for (unsigned int i = 0; i < io_.nInputSize; ++i) {
            const AX_U32 bytes = kInputSize * kInputSize * 3;
            if (AX_SYS_MemAlloc(&io_.pInputs[i].phyAddr, &io_.pInputs[i].pVirAddr, bytes, 128,
                                (const AX_S8*)"manhole_input") != 0) return -1;
            io_.pInputs[i].nSize = bytes;
            io_.pInputs[i].pStride = new AX_S32[4]{};
            io_.pInputs[i].pStride[0] = kInputSize * 3;
        }
        for (unsigned int i = 0; i < io_.nOutputSize; ++i) {
            const AX_U32 bytes = io_info_->pOutputs[i].nSize;
            if (AX_SYS_MemAlloc(&io_.pOutputs[i].phyAddr, &io_.pOutputs[i].pVirAddr, bytes, 128,
                                (const AX_S8*)"manhole_output") != 0) return -1;
            io_.pOutputs[i].nSize = bytes;
            io_.pOutputs[i].pStride = new AX_S32[4]{};
        }
        return 0;
    }

    void GetInputSize(int* w, int* h) override { if (w) *w = kInputSize; if (h) *h = kInputSize; }

    int Inference(const AX_VIDEO_FRAME_T* frame, AI_RESULT_T* result) override {
        if (!handle_ || !frame || !result || !frame->u64VirAddr[0] || frame->u32Width == 0 || frame->u32Height == 0) return -1;
        *result = {};
        cv::Mat nv12(frame->u32Height * 3 / 2, frame->u32Width, CV_8UC1,
                     (void*)frame->u64VirAddr[0], frame->u32PicStride[0]);
        cv::Mat rgb, input;
        cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
        const auto letterbox = ai_letterbox::letterbox(rgb, input, kInputSize, kInputSize);
        std::memcpy(io_.pInputs[0].pVirAddr, input.data, kInputSize * kInputSize * 3);
        AX_SYS_MflushCache(io_.pInputs[0].phyAddr, io_.pInputs[0].pVirAddr, io_.pInputs[0].nSize);
        if (AX_ENGINE_RunSync(handle_, &io_) != 0) return -1;

        float* output = nullptr;
        const size_t expected = static_cast<size_t>(kClassCount + 4) * kAnchorCount * sizeof(float);
        for (unsigned int i = 0; i < io_.nOutputSize; ++i)
            if (io_.pOutputs[i].pVirAddr && io_.pOutputs[i].nSize == expected) output = (float*)io_.pOutputs[i].pVirAddr;
        if (!output && io_.nOutputSize) output = (float*)io_.pOutputs[0].pVirAddr;
        if (!output) return -1;

        struct Candidate { float x, y, w, h, score; int label; };
        std::vector<Candidate> candidates;
        for (int i = 0; i < kAnchorCount; ++i) {
            int label = 0;
            float score = output[4 * kAnchorCount + i];
            for (int c = 1; c < kClassCount; ++c) {
                const float value = output[(4 + c) * kAnchorCount + i];
                if (value > score) { score = value; label = c; }
            }
            if (!std::isfinite(score) || score < conf_) continue;
            const float cx = output[i], cy = output[kAnchorCount + i];
            const float w = output[2 * kAnchorCount + i], h = output[3 * kAnchorCount + i];
            if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h) || w <= 0 || h <= 0) continue;
            candidates.push_back({cx - w * .5f, cy - h * .5f, w, h, score, label});
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
        std::vector<int> picked;
        for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
            bool keep = true;
            for (int j : picked) {
                if (candidates[i].label != candidates[j].label) continue;
                const auto& a = candidates[i];
                const auto& b = candidates[j];
                const float inter = std::max(0.f, std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x)) *
                                    std::max(0.f, std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y));
                const float uni = a.w * a.h + b.w * b.h - inter;
                if (inter / (uni + 1e-7f) > nms_) { keep = false; break; }
            }
            if (keep) picked.push_back(i);
        }
        for (int index : picked) {
            if (result->nObjSize >= MAX_DETECT_OBJ_NUM) break;
            const auto& candidate = candidates[index];
            auto& out = result->objects[result->nObjSize++];
            float x = candidate.x, y = candidate.y, w = candidate.w, h = candidate.h;
            ai_letterbox::scale_bbox_to_original(x, y, w, h, letterbox);
            x = std::max(0.f, std::min(x, (float)frame->u32Width));
            y = std::max(0.f, std::min(y, (float)frame->u32Height));
            w = std::max(0.f, std::min(w, (float)frame->u32Width - x));
            h = std::max(0.f, std::min(h, (float)frame->u32Height - y));
            out.x = x / frame->u32Width; out.y = y / frame->u32Height;
            out.w = w / frame->u32Width; out.h = h / frame->u32Height;
            out.class_id = candidate.label; out.score = candidate.score;
            std::snprintf(out.label, sizeof(out.label), "%s", kLabels[out.class_id]);
        }
        return 0;
    }

    int Deinit() override {
        if (io_.pInputs) for (unsigned int i = 0; i < io_.nInputSize; ++i) { if (io_.pInputs[i].pVirAddr) AX_SYS_MemFree(io_.pInputs[i].phyAddr, io_.pInputs[i].pVirAddr); delete[] io_.pInputs[i].pStride; }
        if (io_.pOutputs) for (unsigned int i = 0; i < io_.nOutputSize; ++i) { if (io_.pOutputs[i].pVirAddr) AX_SYS_MemFree(io_.pOutputs[i].phyAddr, io_.pOutputs[i].pVirAddr); delete[] io_.pOutputs[i].pStride; }
        delete[] io_.pInputs; delete[] io_.pOutputs; io_.pInputs = nullptr; io_.pOutputs = nullptr;
        if (handle_) AX_ENGINE_DestroyHandle(handle_);
        handle_ = nullptr;
        io_info_ = nullptr;
        return 0;
    }

private:
    AX_ENGINE_HANDLE handle_ = nullptr;
    AX_ENGINE_IO_INFO_T* io_info_ = nullptr;
    AX_ENGINE_IO_T io_{};
    float conf_ = kDefaultConf, nms_ = kDefaultNms;
};

extern "C" IAIModel* CreateAIModel() { return new ManholeCoverModel(); }
extern "C" void DestroyAIModel(IAIModel* model) { delete model; }
