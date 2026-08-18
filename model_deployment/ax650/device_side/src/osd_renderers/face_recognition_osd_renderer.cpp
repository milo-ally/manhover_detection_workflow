#include "../include/osd_renderers/face_recognition_osd_renderer.h"

#include "opencv2/opencv.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace {
float clampFloat(float v, float lo, float hi) {
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}

int clampInt(int v, int lo, int hi) {
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}
}

float FaceRecognitionOSDRenderer::iou(const TrackState& a, const TrackState& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.w, b.x + b.w);
    const float y2 = std::min(a.y + a.h, b.y + b.h);
    const float iw = std::max(0.0f, x2 - x1);
    const float ih = std::max(0.0f, y2 - y1);
    const float inter = iw * ih;
    const float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 1e-6f ? inter / uni : 0.0f;
}

float FaceRecognitionOSDRenderer::centerDistance(const TrackState& a, const TrackState& b) {
    const float acx = a.x + a.w * 0.5f;
    const float acy = a.y + a.h * 0.5f;
    const float bcx = b.x + b.w * 0.5f;
    const float bcy = b.y + b.h * 0.5f;
    const float dx = acx - bcx;
    const float dy = acy - bcy;
    return std::sqrt(dx * dx + dy * dy);
}

float FaceRecognitionOSDRenderer::areaRatio(const TrackState& a, const TrackState& b) {
    const float aa = std::max(1e-6f, a.w * a.h);
    const float ba = std::max(1e-6f, b.w * b.h);
    return aa / ba;
}

FaceRecognitionOSDRenderer::FaceRecognitionOSDRenderer(const nlohmann::json& params) {
    applyRuntimeParams(params);
}

void FaceRecognitionOSDRenderer::applyRuntimeParams(const nlohmann::json& params) {
    if (!params.is_object()) return;
    auto readFloat = [&](const char* key, float current, float lo, float hi) {
        if (!params.contains(key)) return current;
        try {
            return clampFloat(params.at(key).get<float>(), lo, hi);
        } catch (...) {
            return current;
        }
    };
    auto readInt = [&](const char* key, int current, int lo, int hi) {
        if (!params.contains(key)) return current;
        try {
            return clampInt(params.at(key).get<int>(), lo, hi);
        } catch (...) {
            return current;
        }
    };
    auto readBool = [&](const char* key, bool current) {
        if (!params.contains(key)) return current;
        try {
            if (params.at(key).is_boolean()) return params.at(key).get<bool>();
            return params.at(key).get<int>() != 0;
        } catch (...) {
            return current;
        }
    };

    params_.trackIouHold = readFloat("track_iou_hold", params_.trackIouHold, 0.1f, 0.9f);
    params_.trackCenterDistMax = readFloat("track_center_dist_max", params_.trackCenterDistMax, 0.02f, 0.5f);
    params_.trackAreaRatioMin = readFloat("track_area_ratio_min", params_.trackAreaRatioMin, 0.1f, 1.0f);
    params_.trackAreaRatioMax = readFloat("track_area_ratio_max", params_.trackAreaRatioMax, 1.0f, 6.0f);
    params_.trackMaxMiss = readInt("track_max_miss", params_.trackMaxMiss, 1, 120);
    params_.nameConfirmHits = readInt("name_confirm_hits", params_.nameConfirmHits, 1, 30);
    params_.unknownConfirmHits = readInt("unknown_confirm_hits", params_.unknownConfirmHits, 1, 60);
    params_.showNameMinScore = readFloat("show_name_min_score", params_.showNameMinScore, 0.1f, 1.0f);
    params_.lockNameConfirmHits = readInt("lock_name_confirm_hits", params_.lockNameConfirmHits, 2, 120);
    params_.unlockLockBreakHits = readInt("unlock_lock_break_hits", params_.unlockLockBreakHits, 1, 120);
    params_.showUnknownTrackId = readBool("show_unknown_track_id", params_.showUnknownTrackId);
}

unsigned int FaceRecognitionOSDRenderer::render(const AI_RESULT_T* result,
                                                 int dstW, int dstH,
                                                 AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                                                 unsigned int maxElements) {
    if (!result || !tDisp) return 0;

    if (previousBatchSize_ > 0 && bitmapPtrs_.size() >= previousBatchSize_) {
        bitmapPtrs_.erase(bitmapPtrs_.begin(), bitmapPtrs_.begin() + previousBatchSize_);
    }

    for (auto& tr : tracks_) tr.matchedInFrame = false;

    unsigned int dispIdx = 0;
    unsigned int newBitmapCount = 0;
    std::vector<TrackState> newTracks;
    newTracks.reserve(result->nObjSize);

    for (unsigned int i = 0; i < result->nObjSize && dispIdx < maxElements - 1; ++i) {
        const auto& obj = result->objects[i];

        int x = static_cast<int>(obj.x * dstW);
        int y = static_cast<int>(obj.y * dstH);
        int w = static_cast<int>(obj.w * dstW);
        int h = static_cast<int>(obj.h * dstH);

        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + w >= dstW) w = dstW - x - 2;
        if (y + h >= dstH) h = dstH - y - 2;
        if (w % 2 != 0) w--;
        if (h % 2 != 0) h--;
        if (w <= 4 || h <= 4) continue;

        TrackState cur;
        cur.id = nextTrackId_++;
        cur.x = obj.x;
        cur.y = obj.y;
        cur.w = obj.w;
        cur.h = obj.h;
        cur.score = obj.score;
        const bool lowScoreKnown = (obj.class_id >= 0) && (obj.score < params_.showNameMinScore);
        const bool rawUnknown = (obj.class_id < 0) || lowScoreKnown ||
                                (std::string(obj.label) == "unknown") || (std::string(obj.label) == "unknown_face");
        const std::string rawName = rawUnknown ? "unknown" : std::string(obj.label);

        // 一對一匹配，避免舊 track 被多次復用
        int bestTrack = -1;
        float bestCost = std::numeric_limits<float>::max();
        for (int t = 0; t < static_cast<int>(tracks_.size()); ++t) {
            if (tracks_[t].miss > params_.trackMaxMiss) continue;
            if (tracks_[t].matchedInFrame) continue;

            const float ov = iou(cur, tracks_[t]);
            if (ov < params_.trackIouHold) continue;
            const float cdist = centerDistance(cur, tracks_[t]);
            if (cdist > params_.trackCenterDistMax) continue;
            const float ar = areaRatio(cur, tracks_[t]);
            if (ar < params_.trackAreaRatioMin || ar > params_.trackAreaRatioMax) continue;

            // cost 越小越好
            const float cost = (1.0f - ov) + 0.35f * cdist;
            if (cost < bestCost) {
                bestCost = cost;
                bestTrack = t;
            }
        }
        if (bestTrack >= 0) {
            auto& prev = tracks_[bestTrack];
            prev.matchedInFrame = true;
            cur.id = prev.id;
            cur.name = prev.name;
            cur.lockedName = prev.lockedName;
            cur.pendingName = prev.pendingName;
            cur.pendingNameHits = prev.pendingNameHits;
            cur.unknownHits = prev.unknownHits;
            cur.stableNameHits = prev.stableNameHits;
            cur.lockBreakHits = prev.lockBreakHits;
            cur.miss = 0;
            // 保留當前幀原始分數
        }

        if (rawUnknown) {
            cur.unknownHits++;
            if (cur.unknownHits >= params_.unknownConfirmHits) {
                cur.name = "unknown";
            }
        } else {
            cur.unknownHits = 0;
            if (cur.name.empty() || cur.name == "unknown") {
                if (cur.pendingName == rawName) {
                    cur.pendingNameHits++;
                } else {
                    cur.pendingName = rawName;
                    cur.pendingNameHits = 1;
                }
                if (cur.pendingNameHits >= params_.nameConfirmHits) {
                    cur.name = rawName;
                    cur.pendingName.clear();
                    cur.pendingNameHits = 0;
                }
            } else if (cur.name != rawName) {
                if (cur.pendingName == rawName) {
                    cur.pendingNameHits++;
                } else {
                    cur.pendingName = rawName;
                    cur.pendingNameHits = 1;
                }
                if (cur.pendingNameHits >= params_.nameConfirmHits) {
                    cur.name = rawName;
                    cur.pendingName.clear();
                    cur.pendingNameHits = 0;
                }
            } else {
                cur.pendingName.clear();
                cur.pendingNameHits = 0;
            }
        }

        if (cur.name.empty()) cur.name = rawName;
        if (cur.score < params_.showNameMinScore) {
            cur.name = "unknown";
        }
        if (!cur.lockedName.empty()) {
            if (rawUnknown || rawName != cur.lockedName) {
                cur.lockBreakHits++;
            } else {
                cur.lockBreakHits = 0;
            }
            if (cur.lockBreakHits >= params_.unlockLockBreakHits) {
                cur.lockedName.clear();
                cur.lockBreakHits = 0;
                cur.stableNameHits = 0;
            } else {
                if (cur.score >= params_.showNameMinScore) {
                    cur.name = cur.lockedName;
                } else {
                    cur.name = "unknown";
                }
            }
        } else {
            if (!rawUnknown && cur.name == rawName && cur.name != "unknown") {
                cur.stableNameHits++;
            } else {
                cur.stableNameHits = 0;
            }
            if (cur.stableNameHits >= params_.lockNameConfirmHits) {
                cur.lockedName = cur.name;
                cur.lockBreakHits = 0;
            }
        }
        const bool finalUnknown = (cur.name == "unknown");
        const std::string finalName = cur.name;
        const unsigned int rectColor = finalUnknown ? 0x0000FF : 0x00FF00; // unknown=紅, known=綠

        tDisp->arrDisp[dispIdx].bShow = AX_TRUE;
        tDisp->arrDisp[dispIdx].eType = AX_IVPS_RGN_TYPE_RECT;
        tDisp->arrDisp[dispIdx].uDisp.tPolygon.tRect.nX = x;
        tDisp->arrDisp[dispIdx].uDisp.tPolygon.tRect.nY = y;
        tDisp->arrDisp[dispIdx].uDisp.tPolygon.tRect.nW = w;
        tDisp->arrDisp[dispIdx].uDisp.tPolygon.tRect.nH = h;
        tDisp->arrDisp[dispIdx].uDisp.tPolygon.nLineWidth = 4;
        tDisp->arrDisp[dispIdx].uDisp.tPolygon.nColor = rectColor;
        tDisp->arrDisp[dispIdx].uDisp.tPolygon.nAlpha = 255;
        dispIdx++;

        if (dispIdx >= maxElements - 1) continue;

        std::string namePart = finalUnknown ? "unknown" : finalName;
        if (!finalUnknown || params_.showUnknownTrackId) {
            namePart += " #" + std::to_string(static_cast<unsigned long long>(cur.id));
        }
        std::string labelText = namePart + " " + std::to_string(static_cast<int>(cur.score * 100)) + "%";
        int baseLine = 0;
        float fontScale = 0.6f;
        int thickness = 2;
        cv::Size labelSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);

        int textW = (labelSize.width + 4 + 1) & ~1;
        if (textW % 4) textW = (textW + 3) & ~3;
        int textH = (labelSize.height + baseLine + 4 + 1) & ~1;
        if (textH % 4) textH = (textH + 3) & ~3;

        std::unique_ptr<unsigned char[]> textBitmap(new unsigned char[textW * textH * 4]);
        memset(textBitmap.get(), 0xFF, textW * textH * 4);
        cv::Mat textMat(textH, textW, CV_8UC4, textBitmap.get());
        cv::putText(textMat, labelText, cv::Point(2, labelSize.height + 2), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                    cv::Scalar(0, 0, 0, 255), thickness);

        for (int row = 0; row < textH; row++) {
            unsigned char* p = textBitmap.get() + row * textW * 4;
            for (int col = 0; col < textW; col++, p += 4) {
                unsigned char t = p[0];
                p[0] = p[2];
                p[2] = t;
            }
        }

        tDisp->arrDisp[dispIdx].bShow = AX_TRUE;
        tDisp->arrDisp[dispIdx].eType = AX_IVPS_RGN_TYPE_OSD;
        tDisp->arrDisp[dispIdx].uDisp.tOSD.enRgbFormat = AX_FORMAT_RGBA8888;
        tDisp->arrDisp[dispIdx].uDisp.tOSD.u32BmpWidth = static_cast<AX_U32>(textW);
        tDisp->arrDisp[dispIdx].uDisp.tOSD.u32BmpHeight = static_cast<AX_U32>(textH);
        tDisp->arrDisp[dispIdx].uDisp.tOSD.u32DstXoffset = std::max(0, x);
        tDisp->arrDisp[dispIdx].uDisp.tOSD.u32DstYoffset = std::max(0, y - textH - 2);
        tDisp->arrDisp[dispIdx].uDisp.tOSD.pBitmap = textBitmap.get();
        tDisp->arrDisp[dispIdx].uDisp.tOSD.u64PhyAddr = 0;
        tDisp->arrDisp[dispIdx].uDisp.tOSD.u16Alpha = 255;

        bitmapPtrs_.push_back(std::move(textBitmap));
        newBitmapCount++;
        dispIdx++;
        newTracks.push_back(cur);
    }

    std::vector<TrackState> carry;
    carry.reserve(tracks_.size());
    for (auto& tr : tracks_) {
        if (!tr.matchedInFrame) {
            tr.miss++;
            if (tr.miss <= params_.trackMaxMiss) carry.push_back(tr);
        }
    }
    tracks_ = std::move(newTracks);
    tracks_.insert(tracks_.end(), carry.begin(), carry.end());

    previousBatchSize_ = lastBatchSize_;
    lastBatchSize_ = newBitmapCount;
    tDisp->nNum = dispIdx;
    return dispIdx;
}

