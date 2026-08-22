#include "rtp_output.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <opencv2/imgproc.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

namespace rkmedia {
namespace {
std::string av_error(int code) {
    char text[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(code, text, sizeof(text));
    return text;
}
}

struct RtpOutput::Impl {
    AVFormatContext* format = nullptr;
    AVCodecContext* encoder = nullptr;
    AVStream* stream = nullptr;
    SwsContext* scaler = nullptr;
    AVFrame* frame = nullptr;
    int width = 0, height = 0, fps = 30;
    std::int64_t next_pts = 0;
    std::string error;
    bool rockchip = false;
    bool header_written = false;

    bool drain() {
        AVPacket* packet = av_packet_alloc();
        if (!packet) return false;
        int rc;
        while ((rc = avcodec_receive_packet(encoder, packet)) >= 0) {
            av_packet_rescale_ts(packet, encoder->time_base, stream->time_base);
            packet->stream_index = stream->index;
            rc = av_interleaved_write_frame(format, packet);
            av_packet_unref(packet);
            if (rc < 0) { error = "RTP write failed: " + av_error(rc); av_packet_free(&packet); return false; }
        }
        av_packet_free(&packet);
        return rc == AVERROR(EAGAIN) || rc == AVERROR_EOF;
    }
};

RtpOutput::RtpOutput() : impl_(new Impl) {}
RtpOutput::~RtpOutput() { close(); }

bool RtpOutput::open(const std::string& host_port, int width, int height, int fps) {
    close();
    impl_->width = width & ~1;
    impl_->height = height & ~1;
    impl_->fps = std::max(1, fps);
    const std::string url = host_port.rfind("rtp://", 0) == 0 ? host_port : "rtp://" + host_port;
    int rc = avformat_alloc_output_context2(&impl_->format, nullptr, "rtp", url.c_str());
    if (rc < 0 || !impl_->format) { impl_->error = "cannot create RTP output: " + av_error(rc); return false; }
    // The AX deployment uses one RTP-only UDP port per path (raw occupies the
    // adjacent port), so RTCP must not claim port+1 here.
    av_opt_set(impl_->format->priv_data, "rtpflags", "skip_rtcp", 0);
    av_opt_set_int(impl_->format->priv_data, "payload_type", 96, 0);
    const AVCodec* codec = avcodec_find_encoder_by_name("h264_rkmpp");
    impl_->rockchip = codec != nullptr;
#if defined(__aarch64__)
    const char* allow_software = std::getenv("RK3588_ALLOW_SOFTWARE_CODEC");
    if (!codec && (!allow_software || std::strcmp(allow_software, "1") != 0)) {
        impl_->error = "Rockchip MPP encoder h264_rkmpp is unavailable; install Rockchip FFmpeg "
                       "or set RK3588_ALLOW_SOFTWARE_CODEC=1";
        return false;
    }
#endif
    if (!codec) codec = avcodec_find_encoder_by_name("libx264");
    if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) { impl_->error = "no H.264 encoder is available"; return false; }
    impl_->encoder = avcodec_alloc_context3(codec);
    impl_->stream = avformat_new_stream(impl_->format, nullptr);
    if (!impl_->encoder || !impl_->stream) { impl_->error = "cannot allocate H.264 output"; return false; }
    impl_->encoder->codec_id = AV_CODEC_ID_H264;
    impl_->encoder->width = impl_->width;
    impl_->encoder->height = impl_->height;
    impl_->encoder->time_base = AVRational{1, impl_->fps};
    impl_->encoder->framerate = AVRational{impl_->fps, 1};
    impl_->encoder->gop_size = impl_->fps;
    impl_->encoder->max_b_frames = 0;
    impl_->encoder->bit_rate = std::max<std::int64_t>(1500000, static_cast<std::int64_t>(width) * height * fps / 18);
    impl_->encoder->pix_fmt = AV_PIX_FMT_NV12;
    if (!impl_->rockchip) {
        impl_->encoder->pix_fmt = AV_PIX_FMT_YUV420P;
        av_opt_set(impl_->encoder->priv_data, "preset", "veryfast", 0);
        av_opt_set(impl_->encoder->priv_data, "tune", "zerolatency", 0);
    }
    if (impl_->format->oformat->flags & AVFMT_GLOBALHEADER) impl_->encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    rc = avcodec_open2(impl_->encoder, codec, nullptr);
    if (rc < 0) { impl_->error = std::string("open encoder ") + codec->name + " failed: " + av_error(rc); return false; }
    avcodec_parameters_from_context(impl_->stream->codecpar, impl_->encoder);
    impl_->stream->time_base = impl_->encoder->time_base;
    impl_->frame = av_frame_alloc();
    impl_->frame->format = impl_->encoder->pix_fmt;
    impl_->frame->width = impl_->width;
    impl_->frame->height = impl_->height;
    rc = av_frame_get_buffer(impl_->frame, 64);
    if (rc < 0) { impl_->error = "encoder frame allocation failed: " + av_error(rc); return false; }
    impl_->scaler = sws_getContext(impl_->width, impl_->height, AV_PIX_FMT_BGR24,
                                   impl_->width, impl_->height, impl_->encoder->pix_fmt,
                                   SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (!(impl_->format->oformat->flags & AVFMT_NOFILE)) {
        rc = avio_open(&impl_->format->pb, url.c_str(), AVIO_FLAG_WRITE);
        if (rc < 0) { impl_->error = "open RTP socket failed: " + av_error(rc); return false; }
    }
    AVDictionary* options = nullptr;
    av_dict_set(&options, "pkt_size", "1200", 0);
    rc = avformat_write_header(impl_->format, &options);
    av_dict_free(&options);
    if (rc < 0) { impl_->error = "RTP header failed: " + av_error(rc); return false; }
    impl_->header_written = true;
    return true;
}

bool RtpOutput::write(const cv::Mat& bgr, std::int64_t pts) {
    if (!impl_->encoder || !impl_->frame || bgr.empty()) return false;
    cv::Mat resized;
    const cv::Mat* image = &bgr;
    if (bgr.cols != impl_->width || bgr.rows != impl_->height) {
        cv::resize(bgr, resized, cv::Size(impl_->width, impl_->height));
        image = &resized;
    }
    int rc = av_frame_make_writable(impl_->frame);
    if (rc < 0) { impl_->error = "encoder frame is not writable"; return false; }
    const uint8_t* source[] = {image->data, nullptr, nullptr, nullptr};
    int strides[] = {static_cast<int>(image->step), 0, 0, 0};
    sws_scale(impl_->scaler, source, strides, 0, impl_->height, impl_->frame->data, impl_->frame->linesize);
    impl_->frame->pts = pts >= 0 ? pts : impl_->next_pts;
    impl_->next_pts = impl_->frame->pts + 1;
    rc = avcodec_send_frame(impl_->encoder, impl_->frame);
    if (rc < 0) { impl_->error = "H.264 encode failed: " + av_error(rc); return false; }
    return impl_->drain();
}

void RtpOutput::close() {
    if (!impl_) return;
    if (impl_->encoder && impl_->format && impl_->header_written) {
        avcodec_send_frame(impl_->encoder, nullptr);
        impl_->drain();
        av_write_trailer(impl_->format);
    }
    sws_freeContext(impl_->scaler); impl_->scaler = nullptr;
    av_frame_free(&impl_->frame);
    avcodec_free_context(&impl_->encoder);
    if (impl_->format) {
        if (!(impl_->format->oformat->flags & AVFMT_NOFILE) && impl_->format->pb) avio_closep(&impl_->format->pb);
        avformat_free_context(impl_->format); impl_->format = nullptr;
    }
    impl_->stream = nullptr;
    impl_->header_written = false;
}

const std::string& RtpOutput::lastError() const { return impl_->error; }
bool RtpOutput::usingRockchipEncoder() const { return impl_->rockchip; }

}  // namespace rkmedia
