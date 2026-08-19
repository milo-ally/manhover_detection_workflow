/*
 * RTP Pusher - 直接推送 H.264/H.265 NAL 單元到 MediaMTX
 * 使用 RTP over UDP，無需 FFmpeg
 * 遵循 RFC 6184 (H.264) 和 RFC 7798 (H.265)
 */

#ifndef _RTP_PUSHER_H_
#define _RTP_PUSHER_H_

#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int sock_fd;                    // UDP socket
    struct sockaddr_in dest_addr;   // MediaMTX 地址
    uint16_t rtp_seq;              // RTP 序列號
    uint32_t rtp_ssrc;             // RTP SSRC
    uint32_t rtp_timestamp;        // RTP 時間戳
    uint32_t frame_count;          // 幀計數器（用於生成時間戳）
    int is_h265;                   // 是否為 H.265 (0=H.264, 1=H.265)
    int clock_rate;                // 時鐘頻率 (90000 for video)
} rtp_pusher_t;

/**
 * 初始化 RTP 推送器
 * @param pusher 推送器指針
 * @param dest_ip MediaMTX 服務器 IP 地址
 * @param dest_port MediaMTX RTP 接收端口（通常為 8000）
 * @param is_h265 是否為 H.265 編碼（0=H.264, 1=H.265）
 * @return 0 成功，-1 失敗
 */
int rtp_pusher_init(rtp_pusher_t* pusher, const char* dest_ip, uint16_t dest_port, int is_h265);

/**
 * 推送 H.264/H.265 NAL 單元（封裝成 RTP 包）
 * @param pusher 推送器指針
 * @param nalu_data NAL 單元數據（包含起始碼 0x00000001 或 0x000001）
 * @param nalu_size NAL 單元大小
 * @param pts 時間戳（微秒）
 * @return 0 成功，-1 失敗
 */
int rtp_pusher_push_nalu(rtp_pusher_t* pusher, const uint8_t* nalu_data, uint32_t nalu_size, uint64_t pts);

/**
 * 推送 H.264/H.265 NAL 單元（不包含起始碼）
 * @param pusher 推送器指針
 * @param nalu_data NAL 單元數據（不包含起始碼）
 * @param nalu_size NAL 單元大小
 * @param pts 時間戳（微秒）
 * @return 0 成功，-1 失敗
 */
int rtp_pusher_push_nalu_no_startcode(rtp_pusher_t* pusher, const uint8_t* nalu_data, uint32_t nalu_size, uint64_t pts);

/**
 * 推送 H.264/H.265 NAL 單元（不包含起始碼），可指定 marker 位
 * @param pusher 推送器指針
 * @param nalu_data NAL 單元數據（不包含起始碼）
 * @param nalu_size NAL 單元大小
 * @param pts 時間戳（微秒）
 * @param marker RTP marker 位（1=幀結束，0=幀繼續）
 * @return 0 成功，-1 失敗
 */
int rtp_pusher_push_nalu_no_startcode_with_marker(rtp_pusher_t* pusher, const uint8_t* nalu_data, uint32_t nalu_size, uint64_t pts, int marker);

/**
 * 釋放 RTP 推送器
 * @param pusher 推送器指針
 */
void rtp_pusher_deinit(rtp_pusher_t* pusher);

#ifdef __cplusplus
}
#endif

#endif // _RTP_PUSHER_H_

