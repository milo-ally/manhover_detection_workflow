#include "../include/osd_renderers/behavior_detection_osd_renderer.h"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <cstring>

unsigned int BehaviorDetectionOSDRenderer::getColorForClass(const char* label) const {
    // 根据类别设置颜色
    // 0xRRGGBB 格式
    if (strcmp(label, "calling") == 0) {
        return 0xFFFF00;  // 黄色 - 打电话
    } else if (strcmp(label, "smoking") == 0) {
        return 0x0000FF;  // 红色 - 吸烟（需要警告）
    }
    return 0x00FF00;  // 默认绿色
}

unsigned int BehaviorDetectionOSDRenderer::render(const AI_RESULT_T* result,
                                                int dstW, int dstH,
                                                AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                                                unsigned int maxElements) {
    if (!result || !tDisp) return 0;

    // Clear previous bitmaps; unique_ptr handles deletion automatically
    bitmapPtrs_.clear();

    unsigned int dispIdx = 0;

    for (unsigned int i = 0; i < result->nObjSize && dispIdx < maxElements - 1; ++i) {
        const auto& obj = result->objects[i];

        // 计算矩形框坐标
        int x = (int)(obj.x * dstW);
        int y = (int)(obj.y * dstH);
        int w = (int)(obj.w * dstW);
        int h = (int)(obj.h * dstH);

        // 边界检查
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + w >= dstW) w = dstW - x - 2;
        if (y + h >= dstH) h = dstH - y - 2;
        if (w % 2 != 0) w--;
        if (h % 2 != 0) h--;

        if (w <= 4 || h <= 4) continue;

        // 根据类别获取颜色
        unsigned int rectColor = getColorForClass(obj.label);

        // 1. 绘制矩形框
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

        // 2. 绘制文字标签（使用 OSD 位图）
        if (dispIdx < maxElements - 1) {
            std::string labelText = std::string(obj.label) + " " + std::to_string((int)(obj.score * 100)) + "%";

            // 使用 OpenCV 创建文字位图
            int baseLine = 0;
            float fontScale = 0.6f;
            int thickness = 2;
            cv::Size labelSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);

            int textW = labelSize.width + 4;
            int textH = labelSize.height + baseLine + 4;

            // 使用 unique_ptr
            std::unique_ptr<unsigned char[]> textBitmap(new unsigned char[textW * textH * 4]);
            memset(textBitmap.get(), 0, textW * textH * 4);  // 透明背景

            cv::Mat textMat(textH, textW, CV_8UC4, textBitmap.get());

            // 绘制文字（使用矩形框颜色）
            // rectColor 是 0xRRGGBB 格式
            // OpenCV 的 cv::Scalar 使用 BGR 格式，所以需要正确转换
            // 0xRRGGBB -> BGR: (B, G, R) = (rectColor & 0xFF, (rectColor >> 8) & 0xFF, (rectColor >> 16) & 0xFF)
            unsigned char b = rectColor & 0xFF;          // B分量
            unsigned char g = (rectColor >> 8) & 0xFF;   // G分量
            unsigned char r = (rectColor >> 16) & 0xFF;  // R分量
            cv::putText(textMat, labelText, cv::Point(2, labelSize.height + 2),
                        cv::FONT_HERSHEY_SIMPLEX, fontScale,
                        cv::Scalar(b, g, r, 255), thickness);

            // 设置 OSD 位图
            tDisp->arrDisp[dispIdx].bShow = AX_TRUE;
            tDisp->arrDisp[dispIdx].eType = AX_IVPS_RGN_TYPE_OSD;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.enRgbFormat = AX_FORMAT_RGBA8888;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32BmpWidth = textW;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32BmpHeight = textH;
            
            // 计算标签位置，确保在图像边界内
            int labelX = std::max(0, x);
            int labelY = std::max(0, y - textH - 2);
            // 如果标签上方没有空间，则放在框的下方
            if (labelY + textH > dstH) {
                labelY = std::min(y + h + 2, dstH - textH);
            }
            
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32DstXoffset = labelX;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u32DstYoffset = labelY;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.pBitmap = textBitmap.get();
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u64PhyAddr = 0;
            tDisp->arrDisp[dispIdx].uDisp.tOSD.u16Alpha = 255;

            // 追踪位图指针，以便在下次渲染或析构时释放
            bitmapPtrs_.push_back(std::move(textBitmap));

            dispIdx++;
        }
    }

    // 位图内存将在下一次 render 调用或析构函数中释放
    // AX_IVPS_RGN_Update 会复制位图数据到内部缓冲区，所以我们可以保留指针直到下次更新
    return dispIdx;
}
