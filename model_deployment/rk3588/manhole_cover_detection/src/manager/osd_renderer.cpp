#include "osd_renderer.h"
#include "../../utilities/sample_log.h"

OSDRenderer::OSDRenderer() {
    renderer_ = std::make_shared<DefaultOSDRenderer>();
}

OSDRenderer::~OSDRenderer() {
    std::lock_guard<std::mutex> lock(renderMutex_);
    initialized_ = false;
}

// 初始化 OSD 渲染器（RK3588 上为纯软件绘制，无需硬件 region）
bool OSDRenderer::init() {
    std::lock_guard<std::mutex> lock(renderMutex_);
    initialized_ = true;
    ALOGN("[OSDRenderer] OSD renderer initialized");
    return true;
}

// 根据 AI 检测结果更新 OSD 显示（直接绘制到输出 BGR 帧）
void OSDRenderer::update(const AI_RESULT_T* pResult, int srcW, int srcH, int dstW, int dstH, cv::Mat& frame) {
    if (!pResult || !initialized_ || !renderer_) return;
    std::lock_guard<std::mutex> lock(renderMutex_);
    renderer_->render(pResult, dstW, dstH, frame, OSD_MAX_DISP_NUM);
    (void)srcW;
    (void)srcH;
}

// 清除 OSD 显示
void OSDRenderer::clear() {
    AI_RESULT_T emptyResult = {0};
    emptyResult.nObjSize = 0;
    (void)emptyResult;
    std::lock_guard<std::mutex> lock(renderMutex_);
    // 纯软件绘制无残留：无需额外操作
    ALOGN("[OSDRenderer] OSD cleared");
}
