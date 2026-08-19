#ifndef FRAME_BROKER_H
#define FRAME_BROKER_H

// 档位2 主码流与 AI 流之间的共享：
//   FrameBroker   —— 主码流解码出的 NV12 帧共享给 AI 流（latest-frame 语义，
//                     对应 AX650 共享 VDEC 组 + latest-frame 缓冲）
//   SharedAIResult —— AI 流推理结果共享给主码流 OSD（对应 AX650
//                     OSDAssociatedModel.latestResult）

#include <mutex>
#include <vector>
#include <cstdint>

#include "ai_interface.h"

struct Nv12FrameData {
    std::vector<uint8_t> data;  // NV12（stride 对齐）
    int width = 0;
    int height = 0;
    int stride = 0;
};

class FrameBroker {
public:
    // 主码流解码线程发布一帧（内部拷贝，latest 覆盖旧帧）
    void publish(const uint8_t* nv12, int w, int h, int stride) {
        if (!nv12 || w <= 0 || h <= 0 || stride <= 0) return;
        std::lock_guard<std::mutex> lk(mu_);
        latest_.width = w;
        latest_.height = h;
        latest_.stride = stride;
        const size_t sz = static_cast<size_t>(stride) * h * 3 / 2;
        latest_.data.assign(nv12, nv12 + sz);
        has_ = true;
    }

    // AI 流取最新帧（取走后清空，AI 慢时旧帧被丢弃）
    bool consumeLatest(Nv12FrameData& out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!has_) return false;
        out = latest_;
        has_ = false;
        return true;
    }

    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        has_ = false;
        latest_.data.clear();
    }

private:
    std::mutex mu_;
    Nv12FrameData latest_;
    bool has_ = false;
};

class SharedAIResult {
public:
    void set(const AI_RESULT_T& r) {
        std::lock_guard<std::mutex> lk(mu_);
        latest_ = r;
    }
    // 返回是否包含检测目标
    bool get(AI_RESULT_T& out) const {
        std::lock_guard<std::mutex> lk(mu_);
        out = latest_;
        return out.nObjSize > 0;
    }
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        AI_RESULT_T z{};
        latest_ = z;
    }

private:
    mutable std::mutex mu_;
    AI_RESULT_T latest_{};
};

#endif  // FRAME_BROKER_H
