#ifndef FACE_RECOGNITION_OSD_RENDERER_H
#define FACE_RECOGNITION_OSD_RENDERER_H

#include "../osd_renderer_interface.h"
#include <cstdint>
#include "../../../utilities/json.hpp"

class FaceRecognitionOSDRenderer : public IOSDRenderer {
public:
    explicit FaceRecognitionOSDRenderer(const nlohmann::json& params = nlohmann::json::object());
    ~FaceRecognitionOSDRenderer() override = default;

    unsigned int render(const AI_RESULT_T* result,
                        int dstW, int dstH,
                        AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                        unsigned int maxElements) override;

    std::string getName() const override { return "FaceRecognitionOSDRenderer"; }

private:
    struct RuntimeParams {
        float trackIouHold = 0.35f;
        float trackCenterDistMax = 0.12f;     // 中心距離門檻（歸一化）
        float trackAreaRatioMin = 0.45f;      // 尺度變化下限
        float trackAreaRatioMax = 2.20f;      // 尺度變化上限
        int trackMaxMiss = 12;
        int nameConfirmHits = 3;
        int unknownConfirmHits = 5;
        float showNameMinScore = 0.40f;
        int lockNameConfirmHits = 10;
        int unlockLockBreakHits = 8;
        bool showUnknownTrackId = false;
    };

    struct TrackState {
        std::uint64_t id = 0;
        float x = 0.f;
        float y = 0.f;
        float w = 0.f;
        float h = 0.f;
        std::string name;
        std::string lockedName;
        float score = 0.f;
        int miss = 0;
        std::string pendingName;
        int pendingNameHits = 0;
        int unknownHits = 0;
        int stableNameHits = 0;
        int lockBreakHits = 0;
        bool matchedInFrame = false;
    };

    static float iou(const TrackState& a, const TrackState& b);
    static float centerDistance(const TrackState& a, const TrackState& b);
    static float areaRatio(const TrackState& a, const TrackState& b);
    void applyRuntimeParams(const nlohmann::json& params);

    std::vector<std::unique_ptr<unsigned char[]>> bitmapPtrs_;
    size_t previousBatchSize_ = 0;
    size_t lastBatchSize_ = 0;
    std::vector<TrackState> tracks_;
    std::uint64_t nextTrackId_ = 1;
    RuntimeParams params_;
};

#endif // FACE_RECOGNITION_OSD_RENDERER_H

