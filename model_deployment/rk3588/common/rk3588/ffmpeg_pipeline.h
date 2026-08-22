#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <opencv2/core.hpp>

namespace rkmedia {

struct DecodedFrame {
    cv::Mat bgr;
    std::int64_t pts = 0;
};

class VideoSource {
public:
    VideoSource();
    ~VideoSource();
    bool open(const std::string& url, const std::string& codec_hint = "auto");
    int run(const std::function<bool(DecodedFrame&&)>& callback,
            const std::atomic<bool>& stop_requested);
    const std::string& lastError() const;
    bool usingRockchipDecoder() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rkmedia
