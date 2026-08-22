#ifndef HELMET_DETECTION_OSD_RENDERER_H
#define HELMET_DETECTION_OSD_RENDERER_H

#include "../osd_renderer_interface.h"
#include <cstring>
#include <memory>
#include <vector>

// 安全帽檢測專用的 OSD 渲染器
// 繪製：矩形框（根據類別不同顏色）+ 文字標籤
// - "helmet" (有安全帽): 綠色
// - "no-helmet" (沒有安全帽): 紅色
// - "hat" (帽子): 黃色
class HelmetDetectionOSDRenderer : public IOSDRenderer {
public:
    HelmetDetectionOSDRenderer() = default;
    virtual ~HelmetDetectionOSDRenderer() = default;

    unsigned int render(const AI_RESULT_T* result,
                       int dstW, int dstH,
                       AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                       unsigned int maxElements) override;

    std::string getName() const override { return "HelmetDetectionOSDRenderer"; }

private:
    unsigned int getColorForClass(const char* label) const;

    std::vector<std::unique_ptr<unsigned char[]>> bitmapPtrs_;
    size_t previousBatchSize_ = 0;
    size_t lastBatchSize_ = 0;
};

#endif // HELMET_DETECTION_OSD_RENDERER_H
