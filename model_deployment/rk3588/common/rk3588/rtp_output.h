#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <opencv2/core.hpp>

namespace rkmedia {

class RtpOutput {
public:
    RtpOutput();
    ~RtpOutput();
    bool open(const std::string& host_port, int width, int height, int fps);
    bool write(const cv::Mat& bgr, std::int64_t pts);
    void close();
    const std::string& lastError() const;
    bool usingRockchipEncoder() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rkmedia
