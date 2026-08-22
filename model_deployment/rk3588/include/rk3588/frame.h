#pragma once

#include <cstddef>
#include <cstdint>

enum class PixelFormat {
    NV12,
    RGB888,
    BGR888,
};

// Platform-neutral frame passed across the application/plugin ABI.  dma_fd is
// populated for DRM/RGA zero-copy paths on RK3588; host frames use data only.
struct VideoFrame {
    int width = 0;
    int height = 0;
    int stride = 0;
    PixelFormat format = PixelFormat::NV12;
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    int dma_fd = -1;
    std::int64_t pts = 0;
};
