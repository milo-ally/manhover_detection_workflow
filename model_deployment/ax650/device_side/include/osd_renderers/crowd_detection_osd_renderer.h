#ifndef CROWD_DETECTION_OSD_RENDERER_H
#define CROWD_DETECTION_OSD_RENDERER_H

#include "../osd_renderer_interface.h"
#include <cstring>
#include <memory>
#include <vector>

// 人員聚集專用的 OSD 渲染器
// 繪製：群組矩形框（青色）+ 文字標籤 "G (成員數)"
class CrowdDetectionOSDRenderer : public IOSDRenderer {
public:
    CrowdDetectionOSDRenderer() = default;
    virtual ~CrowdDetectionOSDRenderer() = default;

    unsigned int render(const AI_RESULT_T* result,
                        int dstW, int dstH,
                        AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                        unsigned int maxElements) override;

    std::string getName() const override { return "CrowdDetectionOSDRenderer"; }

private:
    std::vector<std::unique_ptr<unsigned char[]>> bitmapPtrs_;
    size_t previousBatchSize_ = 0;
    size_t lastBatchSize_ = 0;
};

#endif // CROWD_DETECTION_OSD_RENDERER_H
