#ifndef HUMAN_DETECTION_OSD_RENDERER_H
#define HUMAN_DETECTION_OSD_RENDERER_H

#include "../osd_renderer_interface.h"

// 人員偵測專用 OSD 渲染器
// 額外繪製「人數」：P: N
class HumanDetectionOSDRenderer : public IOSDRenderer {
public:
    HumanDetectionOSDRenderer() = default;
    virtual ~HumanDetectionOSDRenderer() = default;

    unsigned int render(const AI_RESULT_T* result,
                        int dstW, int dstH,
                        AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                        unsigned int maxElements) override;

    std::string getName() const override { return "HumanDetectionOSDRenderer"; }

private:
    // IVPS 可能非同步讀取 pBitmap，所以要延遲釋放（沿用 DefaultOSDRenderer 的策略）
    std::vector<std::unique_ptr<unsigned char[]>> bitmapPtrs_;
    size_t previousBatchSize_ = 0;
    size_t lastBatchSize_ = 0;
};

#endif // HUMAN_DETECTION_OSD_RENDERER_H

