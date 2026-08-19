#include "rk_media.h"

#include <rk_mpi.h>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// RGA/OpenCV 2D（对应 AX650 IVPS）。
// 注意：RGA 对虚拟地址 buffer 的约束较严（需对齐/特殊内存），在纯软件路径下先用
// OpenCV 完成 NV12<->BGR 与缩放（稳定可靠），接口保持不变，后续可替换回 RGA 硬件路径。

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

// 首帧 info-change：解码器会先发一个仅含 SPS/PPS 信息的帧，需用该分辨率
// 配置解码器（内部 buffer 模式下 MPP 自管输出缓冲），然后 mark info-change-ready
// 让它继续正常解码。此模式不需要外部 buffer group，最简单可靠。
bool RkDecoder::setupBufferGroup(int bufSize) {
    (void)bufSize;
    return true;
}

bool RkDecoder::sendPacket(const uint8_t* data, size_t size) {
    if (!init_ || !data || size == 0) return false;

    MppPacket packet = nullptr;
    mpp_packet_init(&packet, const_cast<uint8_t*>(data), size);

    // decode_put_packet 是异步接口；buffer 满时返回 MPP_ERR_BUFFER_FULL(-1012)，
    // 需先 decode_get_frame 清空输出缓冲（含 info-change 处理），短暂等待后重试。
    const int kMaxRetry = 200;  // 累计约 4s 上限
    MPP_RET ret = MPP_OK;
    for (int attempt = 0; attempt < kMaxRetry; ++attempt) {
        ret = mpi_of(mpi_)->decode_put_packet(ctx_of(ctx_), packet);
        if (ret == MPP_OK) break;
        if (ret == MPP_ERR_BUFFER_FULL || ret == MPP_ERR_DISPLAY_FULL) {
            drainFrames();  // 处理 info-change + 清输出缓冲
            struct timespec ts = {0, 1 * 1000 * 1000};  // 1ms
            nanosleep(&ts, nullptr);
            continue;
        }
        break;  // 其他错误，重试无意义
    }
    mpp_packet_deinit(&packet);

    if (ret != MPP_OK) {
        static int errLogged = 0;
        if (errLogged++ < 10)
            fprintf(stderr, "[RkMedia] decode_put_packet failed ret=%d\n", ret);
        return false;
    }
    return drainFrames();
}

bool RkDecoder::drainFrames() {
    for (int i = 0; i < 32; ++i) {
        MppFrame frame = nullptr;
        MPP_RET ret = mpi_of(mpi_)->decode_get_frame(ctx_of(ctx_), &frame);
        if (ret != MPP_OK || !frame) break;

        const int w = static_cast<int>(mpp_frame_get_width(frame));
        const int h = static_cast<int>(mpp_frame_get_height(frame));
        const int hstride = static_cast<int>(mpp_frame_get_hor_stride(frame));
        const int vstride = static_cast<int>(mpp_frame_get_ver_stride(frame));

        // 首帧 info-change：解码器请求确定输出分辨率，需建立外部 buffer group
        if (mpp_frame_get_info_change(frame)) {
            const int bufSize = mpp_frame_get_buf_size(frame);
            fprintf(stderr, "[RkMedia] decoder info-change: %dx%d hstride=%d vstride=%d buf_size=%d\n",
                    w, h, hstride, vstride, bufSize);
            if (!setupBufferGroup(bufSize)) {
                fprintf(stderr, "[RkMedia] setupBufferGroup failed\n");
                mpp_frame_deinit(&frame);
                return false;
            }
            // 让解码器知道外部 buffer 已就绪，继续解码
            ret = mpi_of(mpi_)->control(ctx_of(ctx_), MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
            if (ret != MPP_OK) {
                fprintf(stderr, "[RkMedia] MPP_DEC_SET_INFO_CHANGE_READY failed ret=%d\n", ret);
                mpp_frame_deinit(&frame);
                return false;
            }
            bufferPrepared_ = true;
            mpp_frame_deinit(&frame);
            continue;  // info-change 帧不含像素，不回调
        }

        MppBuffer buf = mpp_frame_get_buffer(frame);
        static int frameLogCount = 0;
        if (frameLogCount++ < 3 || (frameLogCount % 300) == 0)
            fprintf(stderr, "[RkMedia] dec frame %d: %dx%d hstride=%d\n",
                    frameLogCount, w, h, hstride);
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
    if (bufGroup_) {
        mpp_buffer_group_put(static_cast<MppBufferGroup>(bufGroup_));
        bufGroup_ = nullptr;
    }
    ctx_ = nullptr;
    mpi_ = nullptr;
    outBuf_.clear();
    bufferPrepared_ = false;
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
    // 官方 mpi_enc_test：必须先 GET_CFG 取默认配置（否则未设字段是垃圾值，SET 会失败），
    // 再修改字段，最后 SET_CFG。
    if (mpi->control(ctx, MPP_ENC_GET_CFG, cfg) != MPP_OK) {
        fprintf(stderr, "[RkMedia] MPP_ENC_GET_CFG failed\n");
        mpp_enc_cfg_deinit(cfg);
        mpp_destroy(ctx);
        return false;
    }
    mpp_enc_cfg_set_s32(cfg, "prep:width", width);
    mpp_enc_cfg_set_s32(cfg, "prep:height", height);
    mpp_enc_cfg_set_s32(cfg, "prep:hor_stride", stride);   // 行宽（像素）
    mpp_enc_cfg_set_s32(cfg, "prep:ver_stride", height);   // 行数（高度），不是 stride！
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
    // EACH_IDR：每个 IDR 帧内联携带 SPS/PPS，raw 流自包含可解码
    // （DEFAULT 模式 SPS/PPS 可能未随流输出导致 "non-existing PPS"）
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
    // 注意：用 get_pos/get_length（实际编码长度），不能用 get_size（缓冲区容量）
    const uint8_t* data = static_cast<const uint8_t*>(mpp_packet_get_pos(pkt));
    const size_t size = mpp_packet_get_length(pkt);
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
    mpp_frame_set_hor_stride(mf, frame.stride);   // 行宽（像素）
    mpp_frame_set_ver_stride(mf, frame.height);   // 行数（高度）
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
        // 用 get_pos/get_length（实际编码长度），不用 get_size（那是缓冲区容量，会写入垃圾）
        const uint8_t* data = static_cast<const uint8_t*>(mpp_packet_get_pos(pkt));
        const size_t size = mpp_packet_get_length(pkt);
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

// NV12 -> BGR24（OpenCV：YUV420 半平面 -> BGR）
bool nv12ToBgr(const uint8_t* src, int w, int h, int stride, uint8_t* bgr) {
    if (!src || !bgr || w <= 0 || h <= 0) return false;
    cv::Mat nv12((h * 3) / 2, w, CV_8UC1, const_cast<uint8_t*>(src), stride);
    cv::Mat out(h, w, CV_8UC3, bgr);
    cv::cvtColor(nv12, out, cv::COLOR_YUV2BGR_NV12);
    return true;
}

// BGR24 -> NV12（OpenCV：BGR -> I420 平面，再重排为 NV12 半平面；stride 为 Y 行字节数）
bool bgrToNv12(const uint8_t* bgr, int w, int h, uint8_t* nv12, int stride) {
    if (!bgr || !nv12 || w <= 0 || h <= 0) return false;
    cv::Mat b(h, w, CV_8UC3, const_cast<uint8_t*>(bgr));
    cv::Mat i420;
    cv::cvtColor(b, i420, cv::COLOR_BGR2YUV_I420);
    // Y 平面
    memcpy(nv12, i420.data, static_cast<size_t>(w) * h);
    // UV 平面：I420 是 U、V 分开平面，NV12 是 U/V 交错半平面
    const uint8_t* u = i420.data + static_cast<size_t>(w) * h;
    const uint8_t* v = u + static_cast<size_t>(w) * h / 4;
    uint8_t* uv = nv12 + static_cast<size_t>(stride) * h;
    const int halfW = w / 2;
    const int halfH = h / 2;
    for (int row = 0; row < halfH; ++row) {
        const uint8_t* ur = u + static_cast<size_t>(row) * halfW;
        const uint8_t* vr = v + static_cast<size_t>(row) * halfW;
        uint8_t* ov = uv + static_cast<size_t>(row) * stride;
        for (int col = 0; col < halfW; ++col) {
            ov[col * 2] = ur[col];
            ov[col * 2 + 1] = vr[col];
        }
    }
    return true;
}

// NV12 -> NV12 缩放（OpenCV：NV12->BGR->resize->BGR->NV12）
bool resizeNv12(const uint8_t* src, int sw, int sh, int sstride,
                uint8_t* dst, int dw, int dh, int dstride) {
    if (!src || !dst || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return false;
    std::vector<uint8_t> bgrIn(static_cast<size_t>(sw) * sh * 3);
    std::vector<uint8_t> bgrOut(static_cast<size_t>(dw) * dh * 3);
    if (!nv12ToBgr(src, sw, sh, sstride, bgrIn.data())) return false;
    cv::Mat in(sh, sw, CV_8UC3, bgrIn.data());
    cv::Mat out(dh, dw, CV_8UC3, bgrOut.data());
    cv::resize(in, out, cv::Size(dw, dh), 0, 0, cv::INTER_LINEAR);
    return bgrToNv12(bgrOut.data(), dw, dh, dst, dstride);
}

}  // namespace rga_ops
