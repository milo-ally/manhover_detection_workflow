#include "rknpu_manhole.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {

constexpr int kClassCount = 5;
constexpr int kModelSize = 640;
constexpr int kBoxFields = 4;
const char *kClassNames[kClassCount] = {"good", "broke", "lose", "uncovered", "circle"};

float iou(const cv::Rect &a, const cv::Rect &b) {
    const int intersection = (a & b).area();
    const int union_area = a.area() + b.area() - intersection;
    return union_area > 0 ? static_cast<float>(intersection) / union_area : 0.0f;
}

void dump_attr(const char *prefix, const rknn_tensor_attr &attr) {
    std::cout << prefix << " name=" << attr.name << " dims=";
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        std::cout << (i == 0 ? "[" : ",") << attr.dims[i];
    }
    std::cout << "] type=" << attr.type << " fmt=" << attr.fmt
              << " qnt_type=" << attr.qnt_type << " zp=" << attr.zp
              << " scale=" << attr.scale << std::endl;
}

int clamp_int(int value, int low, int high) {
    return std::max(low, std::min(value, high));
}

}  // namespace

const char *class_name(int class_id) {
    return class_id >= 0 && class_id < kClassCount ? kClassNames[class_id] : "unknown";
}

ManholeRknn::~ManholeRknn() {
    release();
}

bool ManholeRknn::init(const std::string &model_path) {
    release();

    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "cannot open model: " << model_path << std::endl;
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::cerr << "invalid model size" << std::endl;
        return false;
    }
    model_data_.resize(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char *>(model_data_.data()), size)) {
        std::cerr << "cannot read model: " << model_path << std::endl;
        return false;
    }

    int ret = rknn_init(&context_, model_data_.data(), model_data_.size(), 0, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_init failed: " << ret << std::endl;
        release();
        return false;
    }

    rknn_input_output_num io_num{};
    ret = rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC || io_num.n_input != 1 || io_num.n_output != 1) {
        std::cerr << "expected one input and one output, got input="
                  << io_num.n_input << " output=" << io_num.n_output << std::endl;
        release();
        return false;
    }

    input_attr_.index = 0;
    output_attr_.index = 0;
    if (rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_)) != RKNN_SUCC ||
        rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &output_attr_, sizeof(output_attr_)) != RKNN_SUCC) {
        std::cerr << "failed to query tensor attributes" << std::endl;
        release();
        return false;
    }
    dump_attr("input:", input_attr_);
    dump_attr("output:", output_attr_);

    if (input_attr_.n_dims != 4 || input_attr_.dims[0] != 1) {
        std::cerr << "unsupported input dimensions" << std::endl;
        release();
        return false;
    }
    if (input_attr_.fmt == RKNN_TENSOR_NCHW) {
        input_channels_ = input_attr_.dims[1];
        input_height_ = input_attr_.dims[2];
        input_width_ = input_attr_.dims[3];
    } else if (input_attr_.fmt == RKNN_TENSOR_NHWC) {
        input_height_ = input_attr_.dims[1];
        input_width_ = input_attr_.dims[2];
        input_channels_ = input_attr_.dims[3];
    } else {
        std::cerr << "unsupported input format: " << input_attr_.fmt << std::endl;
        release();
        return false;
    }
    if (input_channels_ != 3 || input_width_ <= 0 || input_height_ <= 0) {
        std::cerr << "expected a 3-channel image input" << std::endl;
        release();
        return false;
    }

    // The converted model is [1, 9, 8400]. Also accept [1, 8400, 9].
    int first = 0;
    int second = 0;
    if (output_attr_.n_dims == 3) {
        first = output_attr_.dims[1];
        second = output_attr_.dims[2];
    } else if (output_attr_.n_dims == 4 && output_attr_.dims[1] == 1) {
        first = output_attr_.dims[2];
        second = output_attr_.dims[3];
    }
    if (first == kBoxFields + kClassCount) {
        output_channels_ = first;
        output_anchors_ = second;
        output_channel_first_ = true;
    } else if (second == kBoxFields + kClassCount) {
        output_channels_ = second;
        output_anchors_ = first;
        output_channel_first_ = false;
    } else {
        std::cerr << "expected output with 9 channels, got dims="
                  << first << "x" << second << std::endl;
        release();
        return false;
    }
    return true;
}

void ManholeRknn::release() {
    if (context_ != 0) {
        rknn_destroy(context_);
        context_ = 0;
    }
    model_data_.clear();
    input_attr_ = {};
    output_attr_ = {};
    input_width_ = input_height_ = input_channels_ = 0;
    output_channels_ = output_anchors_ = 0;
    output_channel_first_ = false;
    output_debug_printed_ = false;
    no_detection_debug_printed_ = false;
}

bool ManholeRknn::prepare_input(const cv::Mat &bgr, cv::Mat &letterboxed,
                                float &gain, float &pad_x, float &pad_y,
                                std::vector<uint8_t> &nchw_buffer) {
    if (bgr.empty() || bgr.channels() != 3) {
        return false;
    }
    gain = std::min(static_cast<float>(input_width_) / bgr.cols,
                    static_cast<float>(input_height_) / bgr.rows);
    const int resized_width = static_cast<int>(std::round(bgr.cols * gain));
    const int resized_height = static_cast<int>(std::round(bgr.rows * gain));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resized_width, resized_height), 0, 0,
               gain > 1.0f ? cv::INTER_LINEAR : cv::INTER_AREA);

    pad_x = (input_width_ - resized_width) / 2.0f;
    pad_y = (input_height_ - resized_height) / 2.0f;
    const int left = static_cast<int>(std::round(pad_x - 0.1f));
    const int right = static_cast<int>(std::round(pad_x + 0.1f));
    const int top = static_cast<int>(std::round(pad_y - 0.1f));
    const int bottom = static_cast<int>(std::round(pad_y + 0.1f));
    cv::copyMakeBorder(resized, letterboxed, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) {
        rgb = rgb.clone();
    }
    if (input_attr_.fmt == RKNN_TENSOR_NHWC) {
        nchw_buffer.assign(rgb.data, rgb.data + rgb.total() * rgb.elemSize());
    } else {
        nchw_buffer.resize(rgb.total() * rgb.elemSize());
        const size_t plane = static_cast<size_t>(input_width_) * input_height_;
        for (int c = 0; c < input_channels_; ++c) {
            for (int y = 0; y < input_height_; ++y) {
                for (int x = 0; x < input_width_; ++x) {
                    nchw_buffer[static_cast<size_t>(c) * plane + y * input_width_ + x] =
                        rgb.at<cv::Vec3b>(y, x)[c];
                }
            }
        }
    }
    return true;
}

bool ManholeRknn::decode_output(const float *output, size_t float_count,
                                const cv::Size &original_size, float gain,
                                float pad_x, float pad_y, float conf_threshold,
                                float iou_threshold, int max_det,
                                std::vector<Detection> &detections) const {
    if (!output || float_count < static_cast<size_t>(output_channels_) * output_anchors_) {
        return false;
    }
    struct Candidate { cv::Rect2f box; float score; int class_id; };
    std::vector<Candidate> candidates;
    candidates.reserve(output_anchors_);

    auto value = [&](int channel, int anchor) {
        if (output_channel_first_) {
            return output[static_cast<size_t>(channel) * output_anchors_ + anchor];
        }
        return output[static_cast<size_t>(anchor) * output_channels_ + channel];
    };

    if (!output_debug_printed_) {
        float minimum = output[0];
        float maximum = output[0];
        for (size_t i = 1; i < static_cast<size_t>(output_channels_) * output_anchors_; ++i) {
            minimum = std::min(minimum, output[i]);
            maximum = std::max(maximum, output[i]);
        }
        std::cerr << "output decode: floats=" << float_count
                  << " channels=" << output_channels_
                  << " anchors=" << output_anchors_
                  << " layout=" << (output_channel_first_ ? "[channels,anchors]" : "[anchors,channels]")
                  << " range=[" << minimum << "," << maximum << "]"
                  << " sample_scores=" << value(4, 0) << "," << value(5, 0)
                  << "," << value(6, 0) << "," << value(7, 0)
                  << "," << value(8, 0) << std::endl;
        output_debug_printed_ = true;
    }

    int score_candidates = 0;
    for (int anchor = 0; anchor < output_anchors_; ++anchor) {
        int class_id = 0;
        float score = value(4, anchor);
        for (int c = 1; c < kClassCount; ++c) {
            const float candidate = value(4 + c, anchor);
            if (candidate > score) {
                score = candidate;
                class_id = c;
            }
        }
        if (!std::isfinite(score) || score < conf_threshold) {
            continue;
        }
        ++score_candidates;
        const float cx = value(0, anchor);
        const float cy = value(1, anchor);
        const float width = value(2, anchor);
        const float height = value(3, anchor);
        candidates.push_back({cv::Rect2f(cx - width / 2.0f, cy - height / 2.0f,
                                         width, height), score, class_id});
    }

    std::vector<int> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return candidates[a].score > candidates[b].score;
    });
    std::vector<bool> suppressed(candidates.size(), false);
    for (int index : order) {
        if (suppressed[index]) {
            continue;
        }
        const auto &candidate = candidates[index];
        Detection result;
        const float x1 = (candidate.box.x - pad_x) / gain;
        const float y1 = (candidate.box.y - pad_y) / gain;
        const float x2 = (candidate.box.x + candidate.box.width - pad_x) / gain;
        const float y2 = (candidate.box.y + candidate.box.height - pad_y) / gain;
        result.box = cv::Rect(clamp_int(static_cast<int>(std::round(x1)), 0, original_size.width - 1),
                              clamp_int(static_cast<int>(std::round(y1)), 0, original_size.height - 1),
                              0, 0);
        const int right = clamp_int(static_cast<int>(std::round(x2)), 0, original_size.width);
        const int bottom = clamp_int(static_cast<int>(std::round(y2)), 0, original_size.height);
        result.box.width = std::max(0, right - result.box.x);
        result.box.height = std::max(0, bottom - result.box.y);
        result.confidence = candidate.score;
        result.class_id = candidate.class_id;
        if (result.box.area() > 0) {
            detections.push_back(result);
        }
        if (static_cast<int>(detections.size()) >= max_det) {
            break;
        }
        for (int other : order) {
            if (!suppressed[other] && candidates[other].class_id == candidate.class_id &&
                iou(result.box, cv::Rect(cvRound((candidates[other].box.x - pad_x) / gain),
                                         cvRound((candidates[other].box.y - pad_y) / gain),
                                         cvRound(candidates[other].box.width / gain),
                                         cvRound(candidates[other].box.height / gain))) > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
    if (!no_detection_debug_printed_ && detections.empty()) {
        std::cerr << "output decode: candidates_above_conf=" << score_candidates
                  << ", detections_after_nms=0, conf_threshold=" << conf_threshold
                  << ", iou_threshold=" << iou_threshold << std::endl;
        no_detection_debug_printed_ = true;
    }
    return true;
}

bool ManholeRknn::infer(const cv::Mat &bgr, std::vector<Detection> &detections,
                        float conf_threshold, float iou_threshold, int max_det) {
    detections.clear();
    cv::Mat letterboxed;
    float gain = 0.0f, pad_x = 0.0f, pad_y = 0.0f;
    std::vector<uint8_t> input_buffer;
    if (!prepare_input(bgr, letterboxed, gain, pad_x, pad_y, input_buffer)) {
        return false;
    }

    rknn_input input{};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = input_attr_.fmt;
    input.size = input_buffer.size();
    input.buf = input_buffer.data();
    int ret = rknn_inputs_set(context_, 1, &input);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_inputs_set failed: " << ret << std::endl;
        return false;
    }
    ret = rknn_run(context_, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_run failed: " << ret << std::endl;
        return false;
    }

    rknn_output output{};
    output.index = 0;
    output.want_float = 1;
    ret = rknn_outputs_get(context_, 1, &output, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
        return false;
    }
    const bool ok = decode_output(static_cast<const float *>(output.buf),
                                  output.size / sizeof(float), bgr.size(), gain,
                                  pad_x, pad_y, conf_threshold, iou_threshold,
                                  max_det, detections);
    rknn_outputs_release(context_, 1, &output);
    return ok;
}
