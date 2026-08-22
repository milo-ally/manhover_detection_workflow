#pragma once

#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include "rk3588/frame.h"

namespace rkdl {

class FaceAligner {
public:
    FaceAligner();
    ~FaceAligner();
    bool load(const std::string& predictor_path);
    bool align112Rgb(const VideoFrame& face, cv::Mat& aligned_rgb) const;
    bool available() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rkdl
