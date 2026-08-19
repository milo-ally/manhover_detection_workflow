/*
 * RTP Pusher - 直接推送 H.264/H.265 NAL 單元到 MediaMTX
 * 使用 RTP over UDP，無需 FFmpeg
 * 遵循 RFC 6184 (H.264) 和 RFC 7798 (H.265)
 */

#include "rtp_pusher.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>

#define RTP_VERSION 2
#define RTP_PAYLOAD_TYPE_H264 96
#define RTP_PAYLOAD_TYPE_H265 96
#define RTP_MTU 1400  // 最大傳輸單元，避免 IP 分片
#define H264_CLOCK_RATE 90000
#define H265_CLOCK_RATE 90000
/* 設為 0 以保證流暢與低延遲；若僅為減輕丟包可設 50–120，會增加延遲與卡頓感 */
#define RTP_INTER_PACKET_DELAY_US 0

// RTP 頭部結構（12 bytes）
typedef struct {
    uint8_t cc:4;          // CSRC count
    uint8_t x:1;           // extension
    uint8_t p:1;           // padding
    uint8_t version:2;     // version
    uint8_t pt:7;          // payload type
    uint8_t m:1;           // marker
    uint16_t seq;          // sequence number
    uint32_t timestamp;    // timestamp
    uint32_t ssrc;         // synchronization source
} __attribute__((packed)) rtp_header_t;

// H.264 FU-A 指示字節
typedef struct {
    uint8_t s:1;  // start
    uint8_t e:1;  // end
    uint8_t r:1;  // reserved
    uint8_t type:5;  // NAL unit type
} __attribute__((packed)) fu_indicator_t;

// H.265 FU 指示字節（類似 H.264）
typedef struct {
    uint8_t s:1;  // start
    uint8_t e:1;  // end
    uint8_t r:1;  // reserved
    uint8_t type:6;  // NAL unit type (H.265 有 6 bits)
} __attribute__((packed)) fu_indicator_h265_t;

static uint32_t generate_ssrc(void) {
    return (uint32_t)rand();
}

static int find_nalu_start(const uint8_t* data, uint32_t size, uint32_t* start_pos, uint32_t* startcode_len) {
    // 查找 NAL 單元起始碼：0x00000001 或 0x000001
    for (uint32_t i = 0; i < size - 3; i++) {
        if (data[i] == 0x00 && data[i+1] == 0x00) {
            if (data[i+2] == 0x00 && data[i+3] == 0x01) {
                *start_pos = i;
                *startcode_len = 4;
                return 0;
            } else if (data[i+2] == 0x01) {
                *start_pos = i;
                *startcode_len = 3;
                return 0;
            }
        }
    }
    return -1;
}

static int send_rtp_packet(rtp_pusher_t* pusher, const uint8_t* payload, uint32_t payload_size, int marker) {
    // 手動構建 RTP 頭部（12 bytes），確保字節序正確
    uint8_t packet[RTP_MTU];
    
    // RTP 頭部第一個字節：V(2) + P(0) + X(0) + CC(0) = 0x80
    packet[0] = (RTP_VERSION << 6) | 0x00;  // V=2, P=0, X=0, CC=0
    
    // RTP 頭部第二個字節：M(1 if marker) + PT(96)
    packet[1] = (marker ? 0x80 : 0x00) | (RTP_PAYLOAD_TYPE_H264 & 0x7F);
    
    // 序列號（16 bits，網絡字節序）
    uint16_t seq_net = htons(pusher->rtp_seq++);
    memcpy(&packet[2], &seq_net, 2);
    
    // 時間戳（32 bits，網絡字節序）
    uint32_t ts_net = htonl(pusher->rtp_timestamp);
    memcpy(&packet[4], &ts_net, 4);
    
    // SSRC（32 bits，網絡字節序）
    uint32_t ssrc_net = htonl(pusher->rtp_ssrc);
    memcpy(&packet[8], &ssrc_net, 4);
    
    // 複製 payload
    uint32_t header_size = 12;
    uint32_t available = RTP_MTU - header_size;
    uint32_t copy_size = (payload_size < available) ? payload_size : available;
    memcpy(packet + header_size, payload, copy_size);
    
    // 發送 UDP 包（非阻塞 socket：EAGAIN 時重試 2 次、每次 0.5ms，減少剛訂閱時因緩衝未排空而全丟）
    ssize_t sent = sendto(pusher->sock_fd, packet, header_size + copy_size, 0,
                         (struct sockaddr*)&pusher->dest_addr, sizeof(pusher->dest_addr));
    if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        for (int retry = 0; retry < 2 && sent < 0; retry++) {
            usleep(500);
            sent = sendto(pusher->sock_fd, packet, header_size + copy_size, 0,
                         (struct sockaddr*)&pusher->dest_addr, sizeof(pusher->dest_addr));
        }
    }
    
    if (sent < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            static int drop_log = 0;
            if (drop_log++ % 500 == 0) {
                fprintf(stderr, "[RTP] sendto EAGAIN after retries, dropping packet, dest=%s:%d\n",
                        inet_ntoa(pusher->dest_addr.sin_addr), ntohs(pusher->dest_addr.sin_port));
            }
            return -1;
        }
        static int error_count = 0;
        if (error_count++ < 5) {
            fprintf(stderr, "[RTP] sendto() failed: %s (errno=%d), dest=%s:%d\n",
                    strerror(errno), errno,
                    inet_ntoa(pusher->dest_addr.sin_addr),
                    ntohs(pusher->dest_addr.sin_port));
        }
        return -1;
    }
    
    // 發送成功，記錄統計信息
    static int packet_count = 0;
    packet_count++;
    if (packet_count % 100 == 0) {
        char dest_ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &pusher->dest_addr.sin_addr, dest_ip_str, INET_ADDRSTRLEN);
        printf("[RTP] Sent %d packets, last packet size=%zd, seq=%d, ts=%u, dest=%s:%d\n", 
               packet_count, sent, pusher->rtp_seq - 1, pusher->rtp_timestamp,
               dest_ip_str, ntohs(pusher->dest_addr.sin_port));
    }
    
    // 前 10 個包也打印，確認發送開始
    if (packet_count <= 10) {
        char dest_ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &pusher->dest_addr.sin_addr, dest_ip_str, INET_ADDRSTRLEN);
        printf("[RTP] Sent packet #%d: size=%zd, seq=%d, dest=%s:%d\n", 
               packet_count, sent, pusher->rtp_seq - 1, dest_ip_str, ntohs(pusher->dest_addr.sin_port));
    }
    
    /* 包間小延遲，攤平單幀內突發，減輕接收端丟包與 DTS 重排失敗 */
    if (RTP_INTER_PACKET_DELAY_US > 0) {
        usleep(RTP_INTER_PACKET_DELAY_US);
    }
    return copy_size;
}

static int send_h264_nalu(rtp_pusher_t* pusher, const uint8_t* nalu_data, uint32_t nalu_size, int marker) {
    if (nalu_size == 0) return 0;
    
    // 獲取 NAL 單元類型（第一個字節的低 5 位）
    uint8_t nalu_type = nalu_data[0] & 0x1F;
    
    // 單個 RTP 包可以容納的 NAL 單元大小
    // RTP header 是 12 bytes，在單包模式下，NAL 單元（包括 NAL header）直接作為 payload
    uint32_t max_single_packet = RTP_MTU - 12;  // RTP header is 12 bytes
    
    if (nalu_size <= max_single_packet) {
        // 單包模式：直接發送 NAL 單元
        return send_rtp_packet(pusher, nalu_data, nalu_size, marker) >= 0 ? 0 : -1;
    } else {
        // FU-A 分片模式（RFC 6184）
        // FU indicator: 保留 F 和 NRI，類型設置為 28 (FU-A)
        uint8_t fu_indicator = (nalu_data[0] & 0x60) | 28;  // 保留 F(1bit) + NRI(2bits)，類型=28
        
        const uint8_t* data = nalu_data + 1;
        uint32_t remaining = nalu_size - 1;
        uint32_t offset = 0;
        
        while (remaining > 0) {
            uint32_t fragment_size = (remaining > max_single_packet - 2) ? 
                                    (max_single_packet - 2) : remaining;
            
            uint8_t payload[RTP_MTU];
            payload[0] = fu_indicator;  // FU indicator
            
            // FU header: S(1) + E(1) + R(1) + Type(5)
            // 注意：FU header 是一個完整的字節，不是位域結構
            uint8_t fu_header = 0;
            fu_header |= (offset == 0) ? 0x80 : 0x00;  // Start bit (bit 7)
            fu_header |= (remaining <= fragment_size) ? 0x40 : 0x00;  // End bit (bit 6)
            fu_header |= (nalu_type & 0x1F);  // Type (bits 0-4)
            payload[1] = fu_header;
            
            memcpy(payload + 2, data + offset, fragment_size);
            
            int is_marker = (remaining <= fragment_size) ? marker : 0;
            if (send_rtp_packet(pusher, payload, 2 + fragment_size, is_marker) < 0) {
                return -1;
            }
            
            offset += fragment_size;
            remaining -= fragment_size;
        }
        
        return 0;
    }
}

int rtp_pusher_init(rtp_pusher_t* pusher, const char* dest_ip, uint16_t dest_port, int is_h265) {
    if (!pusher || !dest_ip) {
        fprintf(stderr, "[RTP] rtp_pusher_init: invalid parameters\n");
        return -1;
    }
    
    memset(pusher, 0, sizeof(rtp_pusher_t));
    
    // 創建 UDP socket
    pusher->sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (pusher->sock_fd < 0) {
        fprintf(stderr, "[RTP] socket() failed: %s\n", strerror(errno));
        return -1;
    }
    
    // 增大發送緩衝區，減少突發送包時本機丟包（有助於緩解 MediaMTX 端 RTP packets lost）
    {
        int sendbuf_size = 512 * 1024;  // 512KB
        if (setsockopt(pusher->sock_fd, SOL_SOCKET, SO_SNDBUF, &sendbuf_size, sizeof(sendbuf_size)) != 0) {
            fprintf(stderr, "[RTP] setsockopt(SO_SNDBUF) failed: %s\n", strerror(errno));
        }
    }

    /* 非阻塞發送：當無人訂閱該路時 MediaMTX 可能不讀，sendto 會阻塞導致該路 VENC 線程卡死、另一路正常，
     * 表現為「開 live1 時 live2 斷、開 live2 時 live1 斷」。設為 O_NONBLOCK 後緩衝滿則 EAGAIN，丟包不阻塞。 */
    {
        int flags = fcntl(pusher->sock_fd, F_GETFL, 0);
        if (flags >= 0 && fcntl(pusher->sock_fd, F_SETFL, flags | O_NONBLOCK) != 0) {
            fprintf(stderr, "[RTP] fcntl(O_NONBLOCK) failed: %s\n", strerror(errno));
        }
    }

    // 設置目標地址
    memset(&pusher->dest_addr, 0, sizeof(pusher->dest_addr));
    pusher->dest_addr.sin_family = AF_INET;
    pusher->dest_addr.sin_port = htons(dest_port);
    if (inet_aton(dest_ip, &pusher->dest_addr.sin_addr) == 0) {
        fprintf(stderr, "[RTP] inet_aton() failed for IP: %s\n", dest_ip);
        close(pusher->sock_fd);
        return -1;
    }
    
    // 初始化 RTP 參數
    pusher->rtp_seq = (uint16_t)(rand() & 0xFFFF);
    pusher->rtp_ssrc = generate_ssrc();
    pusher->rtp_timestamp = 0;
    pusher->is_h265 = is_h265;
    pusher->clock_rate = is_h265 ? H265_CLOCK_RATE : H264_CLOCK_RATE;
    
    srand((unsigned int)time(NULL));
    
    printf("[RTP] Pusher initialized: %s -> %s:%d (socket=%d, %s)\n", 
           is_h265 ? "H.265" : "H.264", dest_ip, dest_port, 
           pusher->sock_fd,
           pusher->sock_fd >= 0 ? "OK" : "FAILED");
    
    // 打印目標地址信息
    char dest_ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &pusher->dest_addr.sin_addr, dest_ip_str, INET_ADDRSTRLEN);
    printf("[RTP] Destination address: %s:%d\n", dest_ip_str, ntohs(pusher->dest_addr.sin_port));
    
    return 0;
}

int rtp_pusher_push_nalu(rtp_pusher_t* pusher, const uint8_t* nalu_data, uint32_t nalu_size, uint64_t pts) {
    if (!pusher || !nalu_data || nalu_size == 0) {
        return -1;
    }
    
    // 查找 NAL 單元起始碼
    uint32_t start_pos = 0;
    uint32_t startcode_len = 0;
    
    if (find_nalu_start(nalu_data, nalu_size, &start_pos, &startcode_len) == 0) {
        // 跳過起始碼
        nalu_data += start_pos + startcode_len;
        nalu_size -= (start_pos + startcode_len);
    }
    
    // 更新 RTP 時間戳（PTS 轉換為 RTP 時間戳）
    pusher->rtp_timestamp = (uint32_t)((pts * pusher->clock_rate) / 1000000);
    
    // 發送 NAL 單元
    return send_h264_nalu(pusher, nalu_data, nalu_size, 1);  // marker=1 表示一幀結束
}

int rtp_pusher_push_nalu_no_startcode(rtp_pusher_t* pusher, const uint8_t* nalu_data, uint32_t nalu_size, uint64_t pts) {
    // 默認使用 marker=1（假設每個 NAL 都是幀的最後一個）
    return rtp_pusher_push_nalu_no_startcode_with_marker(pusher, nalu_data, nalu_size, pts, 1);
}

int rtp_pusher_push_nalu_no_startcode_with_marker(rtp_pusher_t* pusher, const uint8_t* nalu_data, uint32_t nalu_size, uint64_t pts, int marker) {
    if (!pusher || !nalu_data || nalu_size == 0) {
        return -1;
    }
    
    // 時間戳應該已經在調用前設置好了
    // 這裡不再更新時間戳，直接使用已設置的值
    
    // 發送 NAL 單元（使用指定的 marker 位）
    return send_h264_nalu(pusher, nalu_data, nalu_size, marker);
}

void rtp_pusher_deinit(rtp_pusher_t* pusher) {
    if (pusher && pusher->sock_fd >= 0) {
        close(pusher->sock_fd);
        pusher->sock_fd = -1;
    }
}

