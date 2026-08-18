/**
 * YOLO 推理前預處理：等比例縮放 + 灰邊 (114)，與 Ultralytics / 常見 YOLO 訓練流程對齊。
 * 輸出 bbox 若在 letterbox 後的 target 座標系，需呼叫 scale_bbox_to_original 映回原始影像像素。
 */
#pragma once

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

namespace ai_letterbox {

struct LetterboxInfo {
    float scale = 1.f;
    int pad_w = 0;
    int pad_h = 0;
    int target_w = 640;
    int target_h = 640;
};

/** 將 src letterbox 至 target_w x target_h；dst 若尺寸不符則 create，否則寫入既有緩衝（如外部映射）。 */
inline LetterboxInfo letterbox(const cv::Mat& src, cv::Mat& dst, int target_w, int target_h) {
    LetterboxInfo lb;
    lb.target_w = target_w;
    lb.target_h = target_h;
    const int w = src.cols;
    const int h = src.rows;
    lb.scale = std::min((float)target_w / (float)w, (float)target_h / (float)h);
    const int new_w = (int)std::round((float)w * lb.scale);
    const int new_h = (int)std::round((float)h * lb.scale);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    if (dst.empty() || dst.cols != target_w || dst.rows != target_h || dst.type() != src.type())
        dst.create(target_h, target_w, src.type());
    dst.setTo(cv::Scalar(114, 114, 114));
    lb.pad_w = (target_w - new_w) / 2;
    lb.pad_h = (target_h - new_h) / 2;
    resized.copyTo(dst(cv::Rect(lb.pad_w, lb.pad_h, new_w, new_h)));
    return lb;
}

/** letterbox 影像座標系下的 (x,y,w,h) 像素 → 原圖像素 */
inline void scale_bbox_to_original(float& x, float& y, float& w, float& h, const LetterboxInfo& lb) {
    x = (x - (float)lb.pad_w) / lb.scale;
    y = (y - (float)lb.pad_h) / lb.scale;
    w /= lb.scale;
    h /= lb.scale;
}

}  // namespace ai_letterbox
