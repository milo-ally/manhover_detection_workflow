#include "../include/osd_renderers/fall_detection_osd_renderer.h"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <cstring>
#include <string>

FallDetectionOSDRenderer::FallDetectionOSDRenderer() = default;

unsigned int FallDetectionOSDRenderer::getColorForState(const char* label) const {
    // 支持帶前綴的標籤（如 [0]standing），使用 strstr 檢查是否包含關鍵字
    if (strstr(label, "standing") != nullptr) {
        return 0x00FF00;  // 綠色 (站立)
    } else if (strstr(label, "walking") != nullptr) {
        return 0x00FFFF;  // 青色 (行走)
    } else if (strstr(label, "sitting") != nullptr) {
        return 0xFFFF00;  // 黃色 (坐姿)
    } else if (strstr(label, "falling") != nullptr) {
        return 0xFFA500;  // 橙色 (風險/預警)
    } else if (strstr(label, "FALLEN") != nullptr) {
        return 0xFF0000;  // 紅色 (跌倒報警)
    } else if (strstr(label, "unknown") != nullptr) {
        return 0x808080;  // 灰色
    }
    return 0x00FF00;
}

unsigned int FallDetectionOSDRenderer::render(const AI_RESULT_T* result, 
                                              int dstW, int dstH,
                                              AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                                              unsigned int maxElements) {
    if (!result || !tDisp) return 0;

    if (previousBatchSize_ > 0 && bitmapPtrs_.size() >= previousBatchSize_)
        bitmapPtrs_.erase(bitmapPtrs_.begin(), bitmapPtrs_.begin() + previousBatchSize_);

    unsigned int dispIdx = 0;
    unsigned int newBitmapCount = 0;

    for (unsigned int i = 0; i < result->nObjSize && dispIdx < maxElements - 1; ++i) {
        const auto& obj = result->objects[i];
        
        int x = (int)(obj.x * dstW);
        int y = (int)(obj.y * dstH);
        int w = (int)(obj.w * dstW);
        int h = (int)(obj.h * dstH);
        
        // 邊界檢查
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + w >= dstW) w = dstW - x - 2;
        if (y + h >= dstH) h = dstH - y - 2;
        
        // [Axera IVPS 要求] 寬度和X坐標通常建議偶數對齊，避免顏色異常
        if (x % 2 != 0) x--;
        if (w % 2 != 0) w--;
        if (h % 2 != 0) h--;
        
        if (w <= 4 || h <= 4) continue;
        
        unsigned int rectColor = getColorForState(obj.label);
        
        // 1. 繪製 BBox (矩形框)
        if (dispIdx < maxElements) {
            tDisp->arrDisp[dispIdx].bShow = AX_TRUE;
            tDisp->arrDisp[dispIdx].eType = AX_IVPS_RGN_TYPE_RECT;
            tDisp->arrDisp[dispIdx].uDisp.tPolygon.tRect.nX = x;
            tDisp->arrDisp[dispIdx].uDisp.tPolygon.tRect.nY = y;
            tDisp->arrDisp[dispIdx].uDisp.tPolygon.tRect.nW = w;
            tDisp->arrDisp[dispIdx].uDisp.tPolygon.tRect.nH = h;
            
            // 如果是跌倒，加粗框線
            bool isAlarm = (rectColor == 0xFF0000);
            tDisp->arrDisp[dispIdx].uDisp.tPolygon.nLineWidth = isAlarm ? 6 : 4;
            tDisp->arrDisp[dispIdx].uDisp.tPolygon.nColor = rectColor;
            tDisp->arrDisp[dispIdx].uDisp.tPolygon.nAlpha = 255;
            dispIdx++;
        } else {
            continue;
        }

        // 2. 繪製文字標籤（跌倒偵測不繪製骨架，僅框與文字）
        if (dispIdx < maxElements - 1) {
            std::string labelText = std::string(obj.label);
            if (labelText.empty()) labelText = "unknown";
            labelText += " " + std::to_string((int)(obj.score * 100)) + "%";

            int baseLine = 0;
            float fontScale = 0.8f;
            int thickness = 2;
            cv::Size labelSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);

            int textW = (labelSize.width + 4 + 1) & ~1;
            if (textW % 4) textW = (textW + 3) & ~3;
            int textH = (labelSize.height + baseLine + 4 + 1) & ~1;
            if (textH % 4) textH = (textH + 3) & ~3;

            std::unique_ptr<unsigned char[]> textBitmap(new unsigned char[textW * textH * 4]);
            memset(textBitmap.get(), 0xFF, textW * textH * 4);  // 白底
            cv::Mat textMat(textH, textW, CV_8UC4, textBitmap.get());
            cv::putText(textMat, labelText, cv::Point(2, labelSize.height + 2),
                        cv::FONT_HERSHEY_SIMPLEX, fontScale,
                        cv::Scalar(0, 0, 0, 255), thickness);  // 黑字 BGR
            for (int row = 0; row < textH; row++) {
                unsigned char* p = textBitmap.get() + row * textW * 4;
                for (int col = 0; col < textW; col++, p += 4) {
                    unsigned char t = p[0];
                    p[0] = p[2];
                    p[2] = t;
                }
            }

            int textX = std::max(0, std::min(x, dstW - textW));
            int textY = std::max(0, std::min(y - textH - 2, dstH - textH));
            if (textX % 2 != 0) textX--;

            tDisp->arrDisp[dispIdx].bShow = AX_TRUE;
            tDisp->arrDisp[dispIdx].eType = AX_IVPS_RGN_TYPE_OSD;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.enRgbFormat = AX_FORMAT_RGBA8888;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32BmpWidth = (AX_U32)textW;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32BmpHeight = (AX_U32)textH;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32DstXoffset = (AX_U32)textX;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32DstYoffset = (AX_U32)textY;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.pBitmap = textBitmap.get();
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u64PhyAddr = 0;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u16Alpha = 255;

            bitmapPtrs_.push_back(std::move(textBitmap));
            newBitmapCount++;
            dispIdx++;
        }
    }
    previousBatchSize_ = lastBatchSize_;
    lastBatchSize_ = newBitmapCount;
    return dispIdx;
}
