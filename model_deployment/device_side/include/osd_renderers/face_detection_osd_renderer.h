#ifndef FACE_DETECTION_OSD_RENDERER_H
#define FACE_DETECTION_OSD_RENDERER_H

#include "../osd_renderer_interface.h"

class FaceDetectionOSDRenderer : public IOSDRenderer {
public:
    FaceDetectionOSDRenderer() = default;
    ~FaceDetectionOSDRenderer() override = default;

    unsigned int render(const AI_RESULT_T* result,
                        int dstW, int dstH,
                        AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                        unsigned int maxElements) override;

    std::string getName() const override { return "FaceDetectionOSDRenderer"; }

private:
    std::vector<std::unique_ptr<unsigned char[]>> bitmapPtrs_;
    size_t previousBatchSize_ = 0;
    size_t lastBatchSize_ = 0;
};

#endif // FACE_DETECTION_OSD_RENDERER_H

