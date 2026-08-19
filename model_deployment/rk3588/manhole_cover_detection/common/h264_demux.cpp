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
    if (!bsf) {
        fprintf(stderr, "[H264Demux] WARNING: h264_mp4toannexb bsf not found, "
                "packets will be sent as-is (MP4 源可能是 AVCC，MPP 可能解不出)\n");
    } else {
        AVBSFContext* bsfCtx = nullptr;
        if (av_bsf_alloc(bsf, &bsfCtx) == 0) {
            if (avcodec_parameters_copy(bsfCtx->par_in,
                                        fmt->streams[streamIdx_]->codecpar) == 0 &&
                av_bsf_init(bsfCtx) == 0) {
                bsfCtx_ = bsfCtx;
                fprintf(stderr, "[H264Demux] h264_mp4toannexb bsf initialized\n");
            } else {
                av_bsf_free(&bsfCtx);
                fprintf(stderr, "[H264Demux] WARNING: bsf init failed, packets as-is\n");
            }
        }
    }

    fmtCtx_ = fmt;
    opened_ = true;
    eof_ = false;
    // 从 extradata（avcC 或 SPS/PPS 序列）提取 SPS/PPS 为 AnnexB，供 MPP 首帧前初始化。
    // RTSP 流的 SPS/PPS 在 SDP/extradata，数据包内通常不含，MPP 需先拿到。
    if (fmt->streams[streamIdx_]->codecpar) {
        const AVCodecParameters* par = fmt->streams[streamIdx_]->codecpar;
        spsPpsNals_.clear();
        // avcC 格式：extradata 前 5 字节是配置，之后是 [len4 SPS]*N + [len4 PPS]*M
        if (par->extradata && par->extradata_size > 7 && par->extradata[0] == 1) {
            const uint8_t* ed = par->extradata;
            int off = 5;
            if (ed[4] != 0xFF) {
                // 标准 avcC：profile/level 后直接是 SPS 数量（低 5 位）
                off = 5;
            }
            const int numSps = ed[off++] & 0x1F;
            for (int i = 0; i < numSps && off + 2 <= par->extradata_size; ++i) {
                const int len = (ed[off] << 8) | ed[off + 1];
                off += 2;
                if (off + len <= par->extradata_size) {
                    spsPpsNals_.push_back(0x00); spsPpsNals_.push_back(0x00);
                    spsPpsNals_.push_back(0x00); spsPpsNals_.push_back(0x01);
                    spsPpsNals_.insert(spsPpsNals_.end(), ed + off, ed + off + len);
                    off += len;
                }
            }
            if (off + 1 <= par->extradata_size) {
                const int numPps = ed[off++] & 0x1F;
                for (int i = 0; i < numPps && off + 2 <= par->extradata_size; ++i) {
                    const int len = (ed[off] << 8) | ed[off + 1];
                    off += 2;
                    if (off + len <= par->extradata_size) {
                        spsPpsNals_.push_back(0x00); spsPpsNals_.push_back(0x00);
                        spsPpsNals_.push_back(0x00); spsPpsNals_.push_back(0x01);
                        spsPpsNals_.insert(spsPpsNals_.end(), ed + off, ed + off + len);
                        off += len;
                    }
                }
            }
        }
        if (!spsPpsNals_.empty())
            fprintf(stderr, "[H264Demux] extracted SPS/PPS from extradata: %zu bytes\n",
                    spsPpsNals_.size());
        else if (par->extradata && par->extradata_size >= 4 &&
                 par->extradata[0] == 0 && par->extradata[1] == 0 &&
                 par->extradata[2] == 0 && par->extradata[3] == 1) {
            // extradata 本身已是 AnnexB（RTSP 常见），直接透传
            spsPpsNals_.assign(par->extradata, par->extradata + par->extradata_size);
            fprintf(stderr, "[H264Demux] extradata is AnnexB SPS/PPS: %d bytes\n",
                    par->extradata_size);
        } else {
            fprintf(stderr, "[H264Demux] no SPS/PPS extracted (extradata_size=%d, first=%02x)\n",
                    par->extradata ? par->extradata_size : 0,
                    par->extradata && par->extradata_size ? par->extradata[0] : 0);
        }
        fprintf(stderr, "[H264Demux] opened %s: H.264 video stream idx=%d %dx%d, "
                "duration=%.2fs, nb_frames=%lld\n",
                url.c_str(), streamIdx_,
                par->width, par->height,
                fmt->duration > 0 ? (double)fmt->duration / AV_TIME_BASE : 0.0,
                (long long)fmt->streams[streamIdx_]->nb_frames);
    }
    return true;
}

bool H264Demux::getHeaderNals(std::vector<uint8_t>& out) const {
    out = spsPpsNals_;
    return !out.empty();
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
    // 首个视频包前，先注入从 extradata 提取的 SPS/PPS，让 MPP 能初始化
    //（RTSP 数据包内通常不含 SPS/PPS，若不先发，MPP 永远等不到关键帧）
    if (pkt->stream_index == streamIdx_) {
        if (!headerSent_ && !spsPpsNals_.empty()) {
            headerSent_ = true;
            if (cb_) cb_(spsPpsNals_.data(), spsPpsNals_.size());
            fprintf(stderr, "[H264Demux] injected %zu bytes SPS/PPS before first frame\n",
                    spsPpsNals_.size());
        }
        static int pktLog = 0;
        if (bsfCtx_) {
            AVBSFContext* b = static_cast<AVBSFContext*>(bsfCtx_);
            if (av_bsf_send_packet(b, pkt) == 0) {
                while (av_bsf_receive_packet(b, pkt) == 0) {
                    if (pktLog++ < 10)
                        fprintf(stderr, "[H264Demux] cb pkt size=%d head=%02x%02x%02x%02x\n",
                                pkt->size, pkt->data[0], pkt->data[1], pkt->data[2], pkt->data[3]);
                    if (cb_ && !cb_(pkt->data, pkt->size)) {
                        delivered = false;
                        break;
                    }
                    delivered = true;
                    deliveredCount_++;
                    av_packet_unref(pkt);
                }
            }
        } else {
            if (pktLog++ < 10)
                fprintf(stderr, "[H264Demux] cb pkt(no bsf) size=%d\n", pkt->size);
            if (cb_) delivered = cb_(pkt->data, pkt->size);
            if (delivered) deliveredCount_++;
        }
    } else {
        if (pktLog++ < 5)
            fprintf(stderr, "[H264Demux] skip stream_idx=%d\n", pkt->stream_index);
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
    spsPpsNals_.clear();
    headerSent_ = false;
}
