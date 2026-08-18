#ifndef FALL_DETECTION_OSD_RENDERER_H
#define FALL_DETECTION_OSD_RENDERER_H

#include "../osd_renderer_interface.h"
#include <memory>
#include <vector>
#include <cstring>

class FallDetectionOSDRenderer : public IOSDRenderer {
public:
    FallDetectionOSDRenderer();
    virtual ~FallDetectionOSDRenderer() = default;

    unsigned int render(const AI_RESULT_T* result,
                       int dstW, int dstH,
                       AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                       unsigned int maxElements) override;

    std::string getName() const override { return "FallDetectionOSDRenderer"; }

private:
    unsigned int getColorForState(const char* label) const;

    std::vector<std::unique_ptr<unsigned char[]>> bitmapPtrs_;
    size_t previousBatchSize_ = 0;
    size_t lastBatchSize_ = 0;
};

#endif
