#include "../include/osd_renderers/helmet_detection_osd_renderer.h"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <cstring>

unsigned int HelmetDetectionOSDRenderer::getColorForClass(const char* label) const {
    // 支持帶前綴的標籤（如 [0]no-helmet），使用 strstr 檢查是否包含關鍵字
    if (strstr(label, "helmet") != nullptr && strstr(label, "no-helmet") == nullptr) {
        return 0x00FF00;  // 綠色 - 有安全帽
    } else if (strstr(label, "no-helmet") != nullptr) {
        return 0xFF0000;  // 紅色 - 沒有安全帽
    } else if (strstr(label, "hat") != nullptr) {
        return 0xFFFF00;  // 黃色 - 帽子
    }
    return 0x00FF00;
}

unsigned int HelmetDetectionOSDRenderer::render(const AI_RESULT_T* result,
                                                int dstW, int dstH,
                                                AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                                                unsigned int maxElements) {
    if (!result || !tDisp) return 0;

    // 僅釋放「兩幀前」的位圖，避免 IVPS 非同步使用 pBitmap 時 use-after-free（文字亂碼/洋紅線）
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

        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + w >= dstW) w = dstW - x - 2;
        if (y + h >= dstH) h = dstH - y - 2;
        if (w % 2 != 0) w--;
        if (h % 2 != 0) h--;

        if (w <= 4 || h <= 4) continue;

        unsigned int rectColor = getColorForClass(obj.label);

        // 1. 矩形框
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

        // 2. 文字 OSD：與原本可顯示時相同，僅用虛擬位址 pBitmap，u64PhyAddr=0
        if (dispIdx < maxElements - 1) {
            std::string labelText = std::string(obj.label) + " " + std::to_string((int)(obj.score * 100)) + "%";

            int baseLine = 0;
            float fontScale = 0.6f;
            int thickness = 2;
            cv::Size labelSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);

            int textW = ((labelSize.width + 4 + 1) & ~1);
            if (textW % 4) textW = (textW + 3) & ~3;
            int textH = (labelSize.height + baseLine + 4 + 1) & ~1;
            if (textH % 4) textH = (textH + 3) & ~3;

            std::unique_ptr<unsigned char[]> textBitmap(new unsigned char[textW * textH * 4]);
            memset(textBitmap.get(), 0xFF, textW * textH * 4);  // 白底
            cv::Mat textMat(textH, textW, CV_8UC4, textBitmap.get());
            cv::putText(textMat, labelText, cv::Point(2, labelSize.height + 2),
                        cv::FONT_HERSHEY_SIMPLEX, fontScale,
                        cv::Scalar(0, 0, 0, 255), thickness);  // 黑字 BGR
            // 宣告為 RGBA8888，須與緩衝區一致：OpenCV 輸出 BGRA，逐像素換成 RGBA
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
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32BmpWidth = (AX_U32)textW;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32BmpHeight = (AX_U32)textH;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32DstXoffset = (AX_U32)std::max(0, x);
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32DstYoffset = (AX_U32)std::max(0, y - textH - 2);
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
