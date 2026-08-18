#pragma once

#include <opencv2/core.hpp>
#include <rknn_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Detection {
    cv::Rect box;
    float confidence = 0.0f;
    int class_id = -1;
};

class ManholeRknn {
public:
    ~ManholeRknn();

    bool init(const std::string &model_path);
    bool infer(const cv::Mat &bgr, std::vector<Detection> &detections,
               float conf_threshold, float iou_threshold, int max_det);
    void release();

    int input_width() const { return input_width_; }
    int input_height() const { return input_height_; }

private:
    bool prepare_input(const cv::Mat &bgr, cv::Mat &letterboxed,
                       float &gain, float &pad_x, float &pad_y,
                       std::vector<uint8_t> &nchw_buffer);
    bool decode_output(const float *output, size_t float_count,
                       const cv::Size &original_size, float gain,
                       float pad_x, float pad_y, float conf_threshold,
                       float iou_threshold, int max_det,
                       std::vector<Detection> &detections) const;

    rknn_context context_ = 0;
    rknn_tensor_attr input_attr_{};
    rknn_tensor_attr output_attr_{};
    int input_width_ = 0;
    int input_height_ = 0;
    int input_channels_ = 0;
    int output_channels_ = 0;
    int output_anchors_ = 0;
    std::vector<uint8_t> model_data_;
};

const char *class_name(int class_id);
