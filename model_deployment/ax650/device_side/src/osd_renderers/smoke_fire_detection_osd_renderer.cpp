#include "../include/osd_renderers/smoke_fire_detection_osd_renderer.h"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <cstring>
#include <string>

unsigned int SmokeFireDetectionOSDRenderer::getColorForState(const char* label) const {
    // 支持帶前綴的標籤（如 [0]fire），使用 strstr 檢查是否包含關鍵字
    if (strstr(label, "fire") != nullptr) {
        return 0xFF0000;  // Red for Fire
    } else if (strstr(label, "smoke") != nullptr) {
        return 0xFFA500;  // Orange for Smoke (visible against dark/light backgrounds)
    }
    return 0x00FF00; // Default Green
}

unsigned int SmokeFireDetectionOSDRenderer::render(const AI_RESULT_T* result, 
                                              int dstW, int dstH,
                                              AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                                              unsigned int maxElements) {
    if (!result || !tDisp) return 0;

    if (previousBatchSize_ > 0 && bitmap_cache_.size() >= previousBatchSize_)
        bitmap_cache_.erase(bitmap_cache_.begin(), bitmap_cache_.begin() + previousBatchSize_);

    unsigned int dispIdx = 0;
    unsigned int newBitmapCount = 0;
    
    // Estimate elements per object: Rect + Text = 2
    unsigned int elementsPerObject = 2;
    
    for (unsigned int i = 0; i < result->nObjSize && dispIdx < maxElements - 1; ++i) {
        const auto& obj = result->objects[i];
        
        int x = (int)(obj.x * dstW);
        int y = (int)(obj.y * dstH);
        int w = (int)(obj.w * dstW);
        int h = (int)(obj.h * dstH);
        
        // Boundary checks
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + w >= dstW) w = dstW - x - 2;
        if (y + h >= dstH) h = dstH - y - 2;
        
        // Alignment
        if (x % 2 != 0) x--;
        if (w % 2 != 0) w--;
        if (h % 2 != 0) h--;
        
        if (w <= 4 || h <= 4) continue;
        
        unsigned int rectColor = getColorForState(obj.label);
        
        // 1. Draw BBox (Rect)
        if (dispIdx < maxElements) {
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
        }
        
        // 2. Draw Text Label
        if (dispIdx < maxElements - 1) {
            std::string labelText = std::string(obj.label);
            if (labelText.empty()) {
                labelText = "unknown";
            }
            labelText += " " + std::to_string((int)(obj.score * 100)) + "%";
            
            int baseLine = 0;
            float fontScale = 0.8f;
            int thickness = 2;
            cv::Size labelSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);
            
            int textW = (labelSize.width + 4 + 1) & ~1;
            if (textW % 4) textW = (textW + 3) & ~3;
            int textH = labelSize.height + baseLine + 4;
            if (textH % 4) textH = (textH + 3) & ~3;
            
            std::vector<unsigned char> bitmap_buffer(textW * textH * 4, 0);
            cv::Mat textMat(textH, textW, CV_8UC4, bitmap_buffer.data());
            
            // Extract RGB from color (0xRRGGBB)
            unsigned char r = (rectColor >> 16) & 0xFF;
            unsigned char g = (rectColor >> 8) & 0xFF;
            unsigned char b = rectColor & 0xFF;
            
            cv::putText(textMat, labelText, cv::Point(2, labelSize.height + 2),
                        cv::FONT_HERSHEY_SIMPLEX, fontScale,
                        cv::Scalar(b, g, r, 255), thickness);  // OpenCV BGR
            // 轉成 RGBA8888（IVPS 期望 R,G,B,A），避免錯誤解讀導致色偏/洋紅線
            for (int row = 0; row < textH; row++) {
                unsigned char* p = bitmap_buffer.data() + row * textW * 4;
                for (int col = 0; col < textW; col++, p += 4) {
                    unsigned char t = p[0];
                    p[0] = p[2];
                    p[2] = t;
                }
            }
            bitmap_cache_.push_back(std::move(bitmap_buffer));
            newBitmapCount++;

            int textX = std::max(0, std::min(x, dstW - textW));
            int textY = std::max(0, std::min(y - textH - 2, dstH - textH));
            if (textX % 2 != 0) textX--;

            tDisp->arrDisp[dispIdx].bShow = AX_TRUE;
            tDisp->arrDisp[dispIdx].eType = AX_IVPS_RGN_TYPE_OSD;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.enRgbFormat = AX_FORMAT_RGBA8888;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32BmpWidth = textW;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32BmpHeight = textH;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32DstXoffset = textX;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32DstYoffset = textY;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.pBitmap = bitmap_cache_.back().data();
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u64PhyAddr = 0;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u16Alpha = 255;
            
            dispIdx++;
        }
    }
    previousBatchSize_ = lastBatchSize_;
    lastBatchSize_ = newBitmapCount;
    return dispIdx;
}
