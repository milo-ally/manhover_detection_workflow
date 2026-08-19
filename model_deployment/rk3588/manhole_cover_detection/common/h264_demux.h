#ifndef H264_DEMUX_H
#define H264_DEMUX_H

// 档位2 解封装：用 FFmpeg(libavformat) 把 RTSP/本地视频解成 H.264 AnnexB 包，
// 喂给 MPP 解码器（对应 AX650 的 VideoDemux + VDEC 输入链路）。
// 依赖：libavformat / libavcodec / libavutil（板端 ffmpeg 运行库）。

#include <cstdint>
#include <string>
#include <functional>

class H264Demux {
public:
    H264Demux();
    ~H264Demux();
    H264Demux(const H264Demux&) = delete;
    H264Demux& operator=(const H264Demux&) = delete;

    // 打开 RTSP（强制 tcp）或本地文件
    bool open(const std::string& url);

    // 包回调：返回 false 表示停止读取
    using PacketCallback = std::function<bool(const uint8_t* data, size_t size)>;
    void setPacketCallback(PacketCallback cb) { cb_ = std::move(cb); }

    // 读下一个视频包并回调；返回 false 表示 EOF 或回调要求停止
    bool readOnce();
    bool eof() const { return eof_; }

    void close();

private:
    void* fmtCtx_ = nullptr;   // AVFormatContext*
    void* bsfCtx_ = nullptr;   // AVBSFContext* (h264_mp4toannexb)
    int streamIdx_ = -1;
    PacketCallback cb_;
    bool eof_ = false;
    bool opened_ = false;
    unsigned long long deliveredCount_ = 0;   // 已送达视频包计数（诊断）
    unsigned long long readCalls_ = 0;        // readOnce 调用次数（诊断）
};

#endif  // H264_DEMUX_H
