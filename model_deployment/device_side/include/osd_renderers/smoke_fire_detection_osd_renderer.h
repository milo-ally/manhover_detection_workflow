#pragma once
#include "osd_renderer_interface.h"
#include <vector>

class SmokeFireDetectionOSDRenderer : public IOSDRenderer {
public:
    SmokeFireDetectionOSDRenderer() = default;
    ~SmokeFireDetectionOSDRenderer() = default;

    unsigned int render(const AI_RESULT_T* result, 
                        int dstW, int dstH,
                        AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                        unsigned int maxElements) override;
    
    std::string getName() const override { return "SmokeFireDetectionOSDRenderer"; }

private:
    unsigned int getColorForState(const char* label) const;
    std::vector<std::vector<unsigned char>> bitmap_cache_;
    size_t previousBatchSize_ = 0;
    size_t lastBatchSize_ = 0;
};
