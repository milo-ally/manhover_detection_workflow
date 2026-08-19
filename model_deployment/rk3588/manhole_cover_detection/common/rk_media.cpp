#include "rk_media.h"

#include <rk_mpi.h>
#include <cstring>
#include <cstdio>

#include <cstddef>
#include <im2d.h>
#include <rga.h>

namespace {

inline MppCtx ctx_of(void* p) { return reinterpret_cast<MppCtx>(p); }
inline MppApi* mpi_of(void* p) { return reinterpret_cast<MppApi*>(p); }

}  // namespace

// ===================== RkDecoder (MPP mpi_dec) =====================

RkDecoder::RkDecoder() = default;
RkDecoder::~RkDecoder() { deinit(); }

bool RkDecoder::init() {
    deinit();
    MppCtx ctx = nullptr;
    MppApi* mpi = nullptr;
    if (mpp_create(&ctx, &mpi) != MPP_OK) {
        fprintf(stderr, "[RkMedia] mpp_create(dec) failed\n");
        return false;
    }
    // split mode=1：MPP 自行解析 H.264 码流（不需要外部按 NAL 分包）
    RK_U32 split = 1;
    if (mpi->control(ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &split) != MPP_OK) {
        fprintf(stderr, "[RkMedia] MPP_DEC_SET_PARSER_SPLIT_MODE failed\n");
        mpp_destroy(ctx);
        return false;
    }
    if (mpp_init(ctx, MPP_CTX_DEC, MPP_VIDEO_CodingAVC) != MPP_OK) {
        fprintf(stderr, "[RkMedia] mpp_init(dec) failed\n");
        mpp_destroy(ctx);
        return false;
    }
    ctx_ = ctx;
    mpi_ = mpi;
    init_ = true;
    return true;
}

bool RkDecoder::sendPacket(const uint8_t* data, size_t size) {
    if (!init_ || !data || size == 0) return false;
    MppPacket packet = nullptr;
    mpp_packet_init(&packet, const_cast<uint8_t*>(data), size);
    MPP_RET ret = mpi_of(mpi_)->decode_put_packet(ctx_of(ctx_), packet);
    mpp_packet_deinit(&packet);
    if (ret != MPP_OK) return false;
    return drainFrames();
}

bool RkDecoder::drainFrames() {
    for (int i = 0; i < 8; ++i) {
        MppFrame frame = nullptr;
        MPP_RET ret = mpi_of(mpi_)->decode_get_frame(ctx_of(ctx_), &frame);
        if (ret != MPP_OK || !frame) break;

        const int w = static_cast<int>(mpp_frame_get_width(frame));
        const int h = static_cast<int>(mpp_frame_get_height(frame));
        const int hstride = static_cast<int>(mpp_frame_get_hor_stride(frame));
        MppBuffer buf = mpp_frame_get_buffer(frame);
        if (buf && w > 0 && h > 0 && hstride > 0) {
            const size_t ysz = static_cast<size_t>(hstride) * h;
            outBuf_.resize(ysz + ysz / 2);
            memcpy(outBuf_.data(), mpp_buffer_get_ptr(buf), ysz + ysz / 2);
            RkNv12Frame f;
            f.data = outBuf_.data();
            f.width = w;
            f.height = h;
            f.stride = hstride;
            f.size = outBuf_.size();
            if (frameCb_) frameCb_(f);
        }
        mpp_frame_deinit(&frame);
    }
    return true;
}

void RkDecoder::deinit() {
    if (ctx_) {
        if (mpi_) mpi_of(mpi_)->reset(ctx_of(ctx_));
        mpp_destroy(ctx_of(ctx_));
    }
    ctx_ = nullptr;
    mpi_ = nullptr;
    outBuf_.clear();
    init_ = false;
}

// ===================== RkEncoder (MPP mpi_enc) =====================

RkEncoder::RkEncoder() = default;
RkEncoder::~RkEncoder() { deinit(); }

bool RkEncoder::init(int width, int height, int stride, int fps, int bitrateKbps) {
    deinit();
    MppCtx ctx = nullptr;
    MppApi* mpi = nullptr;
    if (mpp_create(&ctx, &mpi) != MPP_OK) {
        fprintf(stderr, "[RkMedia] mpp_create(enc) failed\n");
        return false;
    }
    if (mpp_init(ctx, MPP_CTX_ENC, MPP_VIDEO_CodingAVC) != MPP_OK) {
        fprintf(stderr, "[RkMedia] mpp_init(enc) failed\n");
        mpp_destroy(ctx);
        return false;
    }

    MppEncCfg cfg = nullptr;
    mpp_enc_cfg_init(&cfg);
    mpp_enc_cfg_set_s32(cfg, "prep:width", width);
    mpp_enc_cfg_set_s32(cfg, "prep:height", height);
    mpp_enc_cfg_set_s32(cfg, "prep:hor_stride", stride);
    mpp_enc_cfg_set_s32(cfg, "prep:ver_stride", stride);
    mpp_enc_cfg_set_s32(cfg, "prep:format", MPP_FMT_YUV420SP);
    mpp_enc_cfg_set_s32(cfg, "rc:mode", MPP_ENC_RC_MODE_CBR);
    const int bps = bitrateKbps * 1000;
    mpp_enc_cfg_set_s32(cfg, "rc:bps", bps);
    mpp_enc_cfg_set_s32(cfg, "rc:bps_max", bps);
    mpp_enc_cfg_set_s32(cfg, "rc:bps_min", bps);
    mpp_enc_cfg_set_s32(cfg, "codec:type", MPP_VIDEO_CodingAVC);
    mpp_enc_cfg_set_s32(cfg, "codec:gop", fps * 2);
    mpp_enc_cfg_set_s32(cfg, "codec:fps", fps);
    mpp_enc_cfg_set_s32(cfg, "codec:fps_in_flex", 0);
    mpp_enc_cfg_set_s32(cfg, "codec:header_mode", MPP_ENC_HEADER_MODE_EACH_IDR);

    if (mpi->control(ctx, MPP_ENC_SET_CFG, cfg) != MPP_OK) {
        fprintf(stderr, "[RkMedia] MPP_ENC_SET_CFG failed\n");
        mpp_enc_cfg_deinit(cfg);
        mpp_destroy(ctx);
        return false;
    }
    mpp_enc_cfg_deinit(cfg);

    ctx_ = ctx;
    mpi_ = mpi;
    init_ = true;
    return true;
}

bool RkEncoder::getHeader(std::vector<uint8_t>& header) {
    header.clear();
    if (!init_) return false;
    MppPacket pkt = nullptr;
    if (mpi_of(mpi_)->control(ctx_of(ctx_), MPP_ENC_GET_HDR_SYNC, &pkt) != MPP_OK || !pkt) {
        return false;
    }
    const uint8_t* data = static_cast<const uint8_t*>(mpp_packet_get_data(pkt));
    const size_t size = mpp_packet_get_size(pkt);
    if (data && size > 0) header.assign(data, data + size);
    mpp_packet_deinit(&pkt);
    return !header.empty();
}

bool RkEncoder::encodeFrame(const RkNv12Frame& frame) {
    if (!init_ || !frame.valid()) return false;
    const size_t size = static_cast<size_t>(frame.stride) * frame.height * 3 / 2;

    MppFrame mf = nullptr;
    mpp_frame_init(&mf);
    mpp_frame_set_width(mf, frame.width);
    mpp_frame_set_height(mf, frame.height);
    mpp_frame_set_hor_stride(mf, frame.stride);
    mpp_frame_set_ver_stride(mf, frame.stride);
    mpp_frame_set_fmt(mf, MPP_FMT_YUV420SP);

    MppBuffer buf = nullptr;
    if (mpp_buffer_get(nullptr, &buf, size) != MPP_OK) {
        mpp_frame_deinit(&mf);
        return false;
    }
    memcpy(mpp_buffer_get_ptr(buf), frame.data, size);
    mpp_frame_set_buffer(mf, buf);

    MPP_RET ret = mpi_of(mpi_)->encode_put_frame(ctx_of(ctx_), mf);
    mpp_frame_deinit(&mf);
    mpp_buffer_put(buf);
    if (ret != MPP_OK) return false;
    return drainPackets();
}

bool RkEncoder::drainPackets() {
    for (int i = 0; i < 8; ++i) {
        MppPacket pkt = nullptr;
        MPP_RET ret = mpi_of(mpi_)->encode_get_packet(ctx_of(ctx_), &pkt);
        if (ret != MPP_OK || !pkt) break;
        const uint8_t* data = static_cast<const uint8_t*>(mpp_packet_get_data(pkt));
        const size_t size = mpp_packet_get_size(pkt);
        if (data && size > 0 && pktCb_) pktCb_(data, size);
        mpp_packet_deinit(&pkt);
    }
    return true;
}

void RkEncoder::deinit() {
    if (ctx_) {
        if (mpi_) mpi_of(mpi_)->reset(ctx_of(ctx_));
        mpp_destroy(ctx_of(ctx_));
    }
    ctx_ = nullptr;
    mpi_ = nullptr;
    init_ = false;
}

// ===================== RGA ops（对应 AX650 IVPS）=====================

namespace rga_ops {

bool resizeNv12(const uint8_t* src, int sw, int sh, int sstride,
                uint8_t* dst, int dw, int dh, int dstride) {
    if (!src || !dst || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return false;
    rga_buffer_t s = wrapbuffer_virtualaddr(const_cast<uint8_t*>(src), sw, sh,
                                            sstride, sstride, RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t d = wrapbuffer_virtualaddr(dst, dw, dh,
                                            dstride, dstride, RK_FORMAT_YCbCr_420_SP);
    IM_STATUS st = imresize(s, d, 0, 0, INTER_LINEAR, 1);
    return st == IM_STATUS_SUCCESS;
}

bool nv12ToBgr(const uint8_t* src, int w, int h, int stride, uint8_t* bgr) {
    if (!src || !bgr || w <= 0 || h <= 0) return false;
    rga_buffer_t s = wrapbuffer_virtualaddr(const_cast<uint8_t*>(src), w, h,
                                            stride, stride, RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t d = wrapbuffer_virtualaddr(bgr, w, h, w * 3, h, RK_FORMAT_BGR_888);
    IM_STATUS st = imcvtcolor(s, d, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_BGR_888,
                              IM_COLOR_SPACE_DEFAULT, 1);
    return st == IM_STATUS_SUCCESS;
}

bool bgrToNv12(const uint8_t* bgr, int w, int h, uint8_t* nv12, int stride) {
    if (!bgr || !nv12 || w <= 0 || h <= 0) return false;
    rga_buffer_t s = wrapbuffer_virtualaddr(const_cast<uint8_t*>(bgr), w, h,
                                            w * 3, h, RK_FORMAT_BGR_888);
    rga_buffer_t d = wrapbuffer_virtualaddr(nv12, w, h, stride, stride, RK_FORMAT_YCbCr_420_SP);
    IM_STATUS st = imcvtcolor(s, d, RK_FORMAT_BGR_888, RK_FORMAT_YCbCr_420_SP,
                              IM_COLOR_SPACE_DEFAULT, 1);
    return st == IM_STATUS_SUCCESS;
}

}  // namespace rga_ops
