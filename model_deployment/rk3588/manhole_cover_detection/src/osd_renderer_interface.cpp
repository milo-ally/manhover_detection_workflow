#include "osd_renderer_interface.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>

unsigned int DefaultOSDRenderer::render(const AI_RESULT_T* result,
                                        int dstW, int dstH,
                                        cv::Mat& frame,
                                        unsigned int maxElements) {
    if (!result || frame.empty()) return 0;
    const cv::Scalar colors[] = {
        {46, 204, 113}, {52, 73, 235}, {0, 165, 255}, {0, 0, 255}, {255, 191, 0}
    };
    const int width = frame.cols;
    const int height = frame.rows;
    unsigned int drawn = 0;
    const unsigned int count = std::min(result->nObjSize, maxElements);
    for (unsigned int i = 0; i < count; ++i) {
        const AI_OBJ_T& obj = result->objects[i];
        const int x1 = static_cast<int>(obj.x * width);
        const int y1 = static_cast<int>(obj.y * height);
        const int x2 = static_cast<int>((obj.x + obj.w) * width);
        const int y2 = static_cast<int>((obj.y + obj.h) * height);
        if (x2 <= x1 || y2 <= y1) continue;
        const cv::Scalar color = colors[obj.class_id % 5];
        cv::rectangle(frame, cv::Rect(x1, y1, x2 - x1, y2 - y1), color, 2);
        char text[128];
        snprintf(text, sizeof(text), "%s %.2f", obj.label, obj.score);
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        const int ty = std::max(text_size.height + 4, y1);
        cv::rectangle(frame, cv::Rect(x1, ty - text_size.height - 6,
                                      text_size.width + 6, text_size.height + 6), color, -1);
        cv::putText(frame, text, cv::Point(x1 + 3, ty - 4), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        ++drawn;
    }
    (void)dstW;
    (void)dstH;
    return drawn;
}
