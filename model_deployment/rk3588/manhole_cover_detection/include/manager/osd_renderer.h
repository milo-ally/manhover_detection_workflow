#ifndef OSD_RENDERER_H
#define OSD_RENDERER_H

// RK3588 版 OSD 渲染器。与 AX650 的 include/manager/osd_renderer.h 同构：
// 每路流一个 OSDRenderer，把 AI_RESULT_T 渲染到输出帧。
// AX650 通过 IVPS OSD Region（硬件叠加）实现；RK3588 直接绘制到 BGR 帧。

#include <mutex>
#include <memory>
#include <opencv2/core.hpp>
#include "../ai_interface.h"
#include "../osd_renderer_interface.h"

#ifndef OSD_MAX_DISP_NUM
#define OSD_MAX_DISP_NUM (32)
#endif

class OSDRenderer {
public:
    OSDRenderer();
    ~OSDRenderer();

    bool init();
    void update(const AI_RESULT_T* result, int srcW, int srcH, int dstW, int dstH, cv::Mat& frame);
    void clear();

private:
    std::mutex renderMutex_;
    std::shared_ptr<IOSDRenderer> renderer_;
    bool initialized_ = false;
};

#endif // OSD_RENDERER_H
