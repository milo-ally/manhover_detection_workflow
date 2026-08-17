#ifndef BEHAVIOR_DETECTION_OSD_RENDERER_H
#define BEHAVIOR_DETECTION_OSD_RENDERER_H

#include "../osd_renderer_interface.h"
#include <cstring>
#include <vector>

// 行为检测专用的 OSD 渲染器
// 绘制：矩形框（根据类别不同颜色）+ 文字标签
// - "calling" (打电话): 黄色
// - "smoking" (吸烟): 红色
class BehaviorDetectionOSDRenderer : public IOSDRenderer {
public:
    BehaviorDetectionOSDRenderer() : bitmapPtrs_() {}
    virtual ~BehaviorDetectionOSDRenderer() {
        bitmapPtrs_.clear();
    }

    unsigned int render(const AI_RESULT_T* result,
                       int dstW, int dstH,
                       AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                       unsigned int maxElements) override;

    std::string getName() const override { return "BehaviorDetectionOSDRenderer"; }

private:
    // 根据类别获取颜色
    unsigned int getColorForClass(const char* label) const;

    // 追踪分配的位图内存指针，用于释放
    std::vector<std::unique_ptr<unsigned char[]>> bitmapPtrs_;
};

#endif // BEHAVIOR_DETECTION_OSD_RENDERER_H
