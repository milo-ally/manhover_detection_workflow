#ifndef RK_MEDIA_H
#define RK_MEDIA_H

// 档位2 硬件层封装：对应 AX650 的 VDEC / VENC / IVPS。
//   VDEC -> MPP mpi_dec（RkDecoder，H.264 硬件解码 → NV12）
//   VENC -> MPP mpi_enc（RkEncoder，NV12 → H.264 硬件编码）
//   IVPS -> RGA（rga_ops，NV12 缩放 / NV12<->BGR 格式转换）
// 头文件/运行库来源与可复现下载地址见工程 readme.txt「MPP/RGA 依赖」章节。

#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>

// NV12 帧描述（Y 平面后紧跟 UV 交错平面）
struct RkNv12Frame {
    const uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;   // Y 平面行字节数（对齐后）
    size_t size = 0;  // stride * height * 3 / 2
    bool valid() const { return data && width > 0 && height > 0; }
};

// ===== MPP 解码器（对应 AX650 VDEC）=====
class RkDecoder {
public:
    RkDecoder();
    ~RkDecoder();
    RkDecoder(const RkDecoder&) = delete;
    RkDecoder& operator=(const RkDecoder&) = delete;

    // 初始化 H.264 解码（MPP 内部缓冲，split mode=1 由 MPP 自行解析码流）
    bool init();

    // 帧回调：在 sendPacket() 内同步触发，NV12 数据仅在回调返回前有效
    using FrameCallback = std::function<void(const RkNv12Frame&)>;
    void setFrameCallback(FrameCallback cb) { frameCb_ = std::move(cb); }

    // 喂 H.264 ES（AnnexB）包
    bool sendPacket(const uint8_t* data, size_t size);
    void deinit();

private:
    bool drainFrames();

    void* ctx_ = nullptr;   // MppCtx
    void* mpi_ = nullptr;   // MppApi*
    FrameCallback frameCb_;
    std::vector<uint8_t> outBuf_;  // 当前帧拷贝缓冲（MPP buffer 释放前拷贝出来）
    bool init_ = false;
};

// ===== MPP 编码器（对应 AX650 VENC）=====
class RkEncoder {
public:
    RkEncoder();
    ~RkEncoder();
    RkEncoder(const RkEncoder&) = delete;
    RkEncoder& operator=(const RkEncoder&) = delete;

    // 初始化 H.264 编码（CBR；NV12 输入；stride 为输入行字节数）
    bool init(int width, int height, int stride, int fps, int bitrateKbps);

    // 编码包回调（AnnexB，含起始码；在 encodeFrame() 内同步触发）
    using PacketCallback = std::function<void(const uint8_t* data, size_t size)>;
    void setPacketCallback(PacketCallback cb) { pktCb_ = std::move(cb); }

    // 取 SPS/PPS 头（init 后、首帧前调用一次）
    bool getHeader(std::vector<uint8_t>& header);

    // 送 NV12 帧编码
    bool encodeFrame(const RkNv12Frame& frame);
    void deinit();

private:
    bool drainPackets();

    void* ctx_ = nullptr;
    void* mpi_ = nullptr;
    PacketCallback pktCb_;
    bool init_ = false;
};

// ===== RGA 2D 加速（对应 AX650 IVPS）=====
namespace rga_ops {

// NV12 → NV12 缩放（主码流 1080p 输出 / AI 流 640x640 支路）
bool resizeNv12(const uint8_t* src, int sw, int sh, int sstride,
                uint8_t* dst, int dw, int dh, int dstride);

// NV12 → BGR24（OSD 画框前的格式转换；bgr 需 w*h*3 字节）
bool nv12ToBgr(const uint8_t* src, int w, int h, int stride, uint8_t* bgr);

// BGR24 → NV12（画框后送回编码器；nv12 需 stride*h*3/2 字节）
bool bgrToNv12(const uint8_t* bgr, int w, int h, uint8_t* nv12, int stride);

}  // namespace rga_ops

#endif  // RK_MEDIA_H
