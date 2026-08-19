extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavutil/opt.h>
}

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
    if (avformat_open_input(&fmt, url.c_str(), nullptr, &opts) != 0) {
        av_dict_free(&opts);
        return false;
    }
    av_dict_free(&opts);
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
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
    return true;
}

bool H264Demux::readOnce() {
    if (!opened_) return false;

    AVPacket* pkt = av_packet_alloc();
    const int r = av_read_frame(static_cast<AVFormatContext*>(fmtCtx_), pkt);
    if (r < 0) {
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
