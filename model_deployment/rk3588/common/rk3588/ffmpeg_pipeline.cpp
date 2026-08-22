#include "ffmpeg_pipeline.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace rkmedia {
namespace {
std::once_flag network_once;

std::string av_error(int code) {
    char text[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(code, text, sizeof(text));
    return text;
}

const AVCodec* rockchip_decoder(AVCodecID id) {
    const char* force_software = std::getenv("RK3588_FORCE_SOFTWARE_CODEC");
    if (force_software && std::strcmp(force_software, "1") == 0) return nullptr;
    if (id == AV_CODEC_ID_H264) return avcodec_find_decoder_by_name("h264_rkmpp");
    if (id == AV_CODEC_ID_HEVC) return avcodec_find_decoder_by_name("hevc_rkmpp");
    return nullptr;
}

bool software_codec_allowed() {
#if defined(__aarch64__)
    const char* value = std::getenv("RK3588_ALLOW_SOFTWARE_CODEC");
    return value && std::strcmp(value, "1") == 0;
#else
    return true;
#endif
}
}  // namespace

struct VideoSource::Impl {
    AVFormatContext* format = nullptr;
    AVCodecContext* decoder = nullptr;
    int stream_index = -1;
    SwsContext* scaler = nullptr;
    AVPixelFormat scaler_source = AV_PIX_FMT_NONE;
    int scaler_width = 0;
    int scaler_height = 0;
    std::string error;
    bool rockchip = false;

    ~Impl() {
        sws_freeContext(scaler);
        avcodec_free_context(&decoder);
        if (format) avformat_close_input(&format);
    }

    bool convert(AVFrame* source, DecodedFrame& output) {
        AVFrame* usable = source;
        AVFrame* transferred = nullptr;
        if (source->format == AV_PIX_FMT_DRM_PRIME || source->hw_frames_ctx) {
            transferred = av_frame_alloc();
            if (!transferred) { error = "cannot allocate hardware transfer frame"; return false; }
            int rc = av_hwframe_transfer_data(transferred, source, 0);
            if (rc < 0) { error = "MPP/DRM frame transfer failed: " + av_error(rc); av_frame_free(&transferred); return false; }
            usable = transferred;
        }
        const auto pixel_format = static_cast<AVPixelFormat>(usable->format);
        if (!scaler || scaler_source != pixel_format || scaler_width != usable->width || scaler_height != usable->height) {
            sws_freeContext(scaler);
            scaler = sws_getContext(usable->width, usable->height, pixel_format,
                                    usable->width, usable->height, AV_PIX_FMT_BGR24,
                                    SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
            scaler_source = pixel_format;
            scaler_width = usable->width;
            scaler_height = usable->height;
        }
        if (!scaler) { error = "unsupported decoded pixel format"; av_frame_free(&transferred); return false; }
        output.bgr.create(usable->height, usable->width, CV_8UC3);
        uint8_t* planes[] = {output.bgr.data, nullptr, nullptr, nullptr};
        int strides[] = {static_cast<int>(output.bgr.step), 0, 0, 0};
        sws_scale(scaler, usable->data, usable->linesize, 0, usable->height, planes, strides);
        output.pts = usable->best_effort_timestamp;
        av_frame_free(&transferred);
        return true;
    }
};

VideoSource::VideoSource() : impl_(new Impl) { std::call_once(network_once, [] { avformat_network_init(); }); }
VideoSource::~VideoSource() = default;

bool VideoSource::open(const std::string& url, const std::string& codec_hint) {
    AVDictionary* options = nullptr;
    if (url.rfind("rtsp://", 0) == 0) {
        av_dict_set(&options, "rtsp_transport", "tcp", 0);
        av_dict_set(&options, "stimeout", "5000000", 0);
        av_dict_set(&options, "fflags", "nobuffer", 0);
        av_dict_set(&options, "flags", "low_delay", 0);
    }
    int rc = avformat_open_input(&impl_->format, url.c_str(), nullptr, &options);
    av_dict_free(&options);
    if (rc < 0) { impl_->error = "open input failed: " + av_error(rc); return false; }
    rc = avformat_find_stream_info(impl_->format, nullptr);
    if (rc < 0) { impl_->error = "stream discovery failed: " + av_error(rc); return false; }
    impl_->stream_index = av_find_best_stream(impl_->format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (impl_->stream_index < 0) { impl_->error = "input has no video stream"; return false; }
    AVCodecParameters* parameters = impl_->format->streams[impl_->stream_index]->codecpar;
    if (codec_hint == "h264" && parameters->codec_id != AV_CODEC_ID_H264) {
        impl_->error = "configured h264 input does not contain H.264"; return false;
    }
    if ((codec_hint == "h265" || codec_hint == "hevc") && parameters->codec_id != AV_CODEC_ID_HEVC) {
        impl_->error = "configured h265 input does not contain HEVC"; return false;
    }
    const AVCodec* codec = rockchip_decoder(parameters->codec_id);
    impl_->rockchip = codec != nullptr;
    if (!codec && (parameters->codec_id == AV_CODEC_ID_H264 || parameters->codec_id == AV_CODEC_ID_HEVC) &&
        !software_codec_allowed()) {
        impl_->error = "Rockchip MPP decoder is unavailable (expected h264_rkmpp/hevc_rkmpp); "
                       "install Rockchip FFmpeg or set RK3588_ALLOW_SOFTWARE_CODEC=1";
        return false;
    }
    if (!codec) codec = avcodec_find_decoder(parameters->codec_id);
    if (!codec) { impl_->error = "no decoder is available for the input codec"; return false; }
    impl_->decoder = avcodec_alloc_context3(codec);
    if (!impl_->decoder) { impl_->error = "cannot allocate decoder context"; return false; }
    avcodec_parameters_to_context(impl_->decoder, parameters);
    impl_->decoder->flags |= AV_CODEC_FLAG_LOW_DELAY;
    impl_->decoder->thread_count = impl_->rockchip ? 1 : 2;
    rc = avcodec_open2(impl_->decoder, codec, nullptr);
    if (rc < 0) { impl_->error = std::string("open decoder ") + codec->name + " failed: " + av_error(rc); return false; }
    return true;
}

int VideoSource::run(const std::function<bool(DecodedFrame&&)>& callback,
                     const std::atomic<bool>& stop_requested) {
    if (!impl_->decoder || !impl_->format) return -1;
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    if (!packet || !frame) { av_packet_free(&packet); av_frame_free(&frame); return -1; }
    int status = 0;
    while (!stop_requested.load()) {
        int rc = av_read_frame(impl_->format, packet);
        if (rc == AVERROR_EOF) break;
        if (rc < 0) { impl_->error = "input read failed: " + av_error(rc); status = rc; break; }
        if (packet->stream_index == impl_->stream_index) {
            rc = avcodec_send_packet(impl_->decoder, packet);
            while (rc >= 0) {
                rc = avcodec_receive_frame(impl_->decoder, frame);
                if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) break;
                if (rc < 0) { impl_->error = "decode failed: " + av_error(rc); status = rc; break; }
                DecodedFrame decoded;
                if (impl_->convert(frame, decoded) && !callback(std::move(decoded))) {
                    status = 0;
                    av_frame_unref(frame);
                    av_packet_unref(packet);
                    goto done;
                }
                av_frame_unref(frame);
            }
        }
        av_packet_unref(packet);
        if (status < 0) break;
    }
done:
    av_packet_free(&packet);
    av_frame_free(&frame);
    return status;
}

const std::string& VideoSource::lastError() const { return impl_->error; }
bool VideoSource::usingRockchipDecoder() const { return impl_->rockchip; }

}  // namespace rkmedia
