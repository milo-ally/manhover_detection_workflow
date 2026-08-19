extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavutil/opt.h>
#include <libavutil/error.h>
}

#include <cstdio>

#include "h264_demux.h"

H264Demux::H264Demux() = default;
H264Demux::~H264Demux() { close(); }

bool H264Demux::open(const std::string& url) {
    close();
    if (url.empty()) return false;

    AVFormatContext* fmt = nullptr;
    AVDictionary* opts = nullptr;
    if (url.rfind("rtsp://", 0) == 0 || url.rfind("rtsps://", 0) == 0) {
        av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    }
    const int openRet = avformat_open_input(&fmt, url.c_str(), nullptr, &opts);
    av_dict_free(&opts);
    if (openRet != 0) {
        char errbuf[128] = {0};
        av_strerror(openRet, errbuf, sizeof(errbuf));
        fprintf(stderr, "[H264Demux] avformat_open_input failed: %s (%s)\n",
                url.c_str(), errbuf);
        return false;
    }
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        fprintf(stderr, "[H264Demux] avformat_find_stream_info failed: %s\n", url.c_str());
        avformat_close_input(&fmt);
        return false;
    }

    for (unsigned i = 0; i < fmt->nb_streams; ++i) {
        const AVCodecParameters* par = fmt->streams[i]->codecpar;
        if (par && par->codec_type == AVMEDIA_TYPE_VIDEO &&
            par->codec_id == AV_CODEC_ID_H264) {
            streamIdx_ = static_cast<int>(i);
            break;
        }
    }
    if (streamIdx_ < 0) {
        // 输出流信息，便于排查（常见：输入是 H.265，而本工程只支持 H.264）
        for (unsigned i = 0; i < fmt->nb_streams; ++i) {
            const AVCodecParameters* par = fmt->streams[i]->codecpar;
            if (!par) continue;
            fprintf(stderr, "[H264Demux] stream[%u] type=%d codec_id=%d (%s)\n",
                    i, par->codec_type, par->codec_id,
                    par->codec_type == AVMEDIA_TYPE_VIDEO ? "video" : "other");
        }
        fprintf(stderr,
                "[H264Demux] no H.264 video stream found in %s. "
                "This project supports H.264 input only; convert first, e.g.:\n"
                "  ffmpeg -i <input> -c:v libx264 -pix_fmt yuv420p <input_h264>.mp4\n",
                url.c_str());
        avformat_close_input(&fmt);
        return false;
    }

    // 统一转 AnnexB（RTSP 已为 AnnexB，mp4 源会转换；对 AnnexB 输入为透传）
    const AVBitStreamFilter* bsf = av_bsf_get_by_name("h264_mp4toannexb");
    if (bsf) {
        AVBSFContext* bsfCtx = nullptr;
        if (av_bsf_alloc(bsf, &bsfCtx) == 0) {
            if (avcodec_parameters_copy(bsfCtx->par_in,
                                        fmt->streams[streamIdx_]->codecpar) == 0 &&
                av_bsf_init(bsfCtx) == 0) {
                bsfCtx_ = bsfCtx;
            } else {
                av_bsf_free(&bsfCtx);
            }
        }
    }

    fmtCtx_ = fmt;
    opened_ = true;
    eof_ = false;
    if (fmt->streams[streamIdx_]->codecpar) {
        const AVCodecParameters* par = fmt->streams[streamIdx_]->codecpar;
        fprintf(stderr, "[H264Demux] opened %s: H.264 video stream idx=%d %dx%d, "
                "duration=%.2fs, nb_frames=%lld\n",
                url.c_str(), streamIdx_,
                par->width, par->height,
                fmt->duration > 0 ? (double)fmt->duration / AV_TIME_BASE : 0.0,
                (long long)fmt->streams[streamIdx_]->nb_frames);
    }
    return true;
}

bool H264Demux::readOnce() {
    if (!opened_) return false;

    AVPacket* pkt = av_packet_alloc();
    const int r = av_read_frame(static_cast<AVFormatContext*>(fmtCtx_), pkt);
    if (r < 0) {
        if (r != AVERROR_EOF) {
            char errbuf[128] = {0};
            av_strerror(r, errbuf, sizeof(errbuf));
            fprintf(stderr, "[H264Demux] av_read_frame error: %s\n", errbuf);
        }
        eof_ = true;
        av_packet_free(&pkt);
        return false;
    }

    bool delivered = false;
    if (pkt->stream_index == streamIdx_) {
        if (bsfCtx_) {
            AVBSFContext* b = static_cast<AVBSFContext*>(bsfCtx_);
            if (av_bsf_send_packet(b, pkt) == 0) {
                while (av_bsf_receive_packet(b, pkt) == 0) {
                    if (cb_ && !cb_(pkt->data, pkt->size)) {
                        delivered = false;
                        break;
                    }
                    delivered = true;
                    av_packet_unref(pkt);
                }
            }
        } else {
            if (cb_) delivered = cb_(pkt->data, pkt->size);
        }
    }
    av_packet_free(&pkt);
    return delivered;
}

void H264Demux::close() {
    if (bsfCtx_) {
        AVBSFContext* b = static_cast<AVBSFContext*>(bsfCtx_);
        av_bsf_free(&b);
        bsfCtx_ = nullptr;
    }
    if (fmtCtx_) {
        AVFormatContext* f = static_cast<AVFormatContext*>(fmtCtx_);
        avformat_close_input(&f);
        fmtCtx_ = nullptr;
    }
    streamIdx_ = -1;
    opened_ = false;
    eof_ = false;
}
