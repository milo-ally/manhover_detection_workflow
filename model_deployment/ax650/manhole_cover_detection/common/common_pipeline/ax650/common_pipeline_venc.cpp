/*
 * AXERA is pleased to support the open source community by making ax-samples available.
 *
 * Copyright (c) 2022, AXERA Semiconductor (Shanghai) Co., Ltd. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

/*
 * Author: ZHEQIUSHUI
 */

#include "../common_pipeline.h"
#include "../rtp_pusher.h"
#include "../../utilities/sample_log.h"

#include <mutex>
#include <chrono>

#include "ax_venc_api.h"
#include "ax_ivps_api.h"
extern "C"
{
#include "common_venc.h"
}

// RTP 推送器（每個 pipeid 一個）
rtp_pusher_t* g_rtp_pushers[64] = {0};  // 最多支持 64 個管道
// 每路流一把鎖：避免一路大 I 幀（數百個 RTP 包）長時間佔用全局鎖，導致其他路無法送包、
// VENC 緩衝堆積、整體卡死，進而觸發 MediaMTX read udp i/o timeout（30s 無包）
static std::mutex g_rtp_push_mutex_per_pipe[64];

void *_venc_get_frame_thread(void *arg)
{
    pipeline_t *pipe = (pipeline_t *)arg;
    ALOGI("[VENC] Thread started for pipeid %d, venc_chn=%d, output_type=%d", 
          pipe->pipeid, pipe->m_venc_attr.n_venc_chn, pipe->m_output_type);
    AX_S16 syncType = 200;
    AX_VENC_STREAM_T stStream = {0};
    AX_VENC_RECV_PIC_PARAM_T stRecvParam;
    stRecvParam.s32RecvPicNum = 1;
    int s32Ret = AX_VENC_StartRecvFrame(pipe->m_venc_attr.n_venc_chn, &stRecvParam);
    if (AX_SUCCESS != s32Ret)
    {
        ALOGE("[VENC] AX_VENC_StartRecvFrame failed for pipeid %d, venc_chn=%d, s32Ret:0x%x", 
              pipe->pipeid, pipe->m_venc_attr.n_venc_chn, s32Ret);
        return NULL;
    }
    ALOGI("[VENC] Started receiving frames for pipeid %d, venc_chn=%d", 
          pipe->pipeid, pipe->m_venc_attr.n_venc_chn);

    static int frame_count_per_pipe[64] = {0};
    static int consecutive_fail_count[64] = {0};  // 連續 GetStream 失敗次數，成功時重置，用於 60s 無幀 log
    while (!pipe->n_loog_exit)
    {
        s32Ret = AX_VENC_GetStream(pipe->m_venc_attr.n_venc_chn, &stStream, syncType);
        // printf("%d\n",stStream.stPack.u32Len);
        if (AX_SUCCESS == s32Ret)
        {
            frame_count_per_pipe[pipe->pipeid]++;
            consecutive_fail_count[pipe->pipeid] = 0;
            if (frame_count_per_pipe[pipe->pipeid] % 30 == 0) {
                ALOGI("[VENC] Received frame #%d for pipeid %d, venc_chn=%d, size=%d, output_type=%d", 
                      frame_count_per_pipe[pipe->pipeid], pipe->pipeid, 
                      pipe->m_venc_attr.n_venc_chn, stStream.stPack.u32Len, pipe->m_output_type);
            }
            switch (pipe->m_output_type)
            {
            case po_mediamtx_h264:
            case po_mediamtx_h265:
            {
                // 直接推送 H.264/H.265 NAL 單元到 MediaMTX（通過 RTP）
                // 只有主碼流（po_mediamtx_h264）會進此分支；AI 流為 po_buff_nv12，無 VENC、不送 RTP。
#if RTP_MAIN_PUSH_ODD_FRAME_ONLY
                // 若一 VDEC 兩 Link 導致主碼流每源幀編碼兩次，可啟用「僅奇數幀推 RTP」去重（編譯時 RTP_MAIN_PUSH_ODD_FRAME_ONLY=1）。
                bool do_rtp_push = (pipe->pipeid >= 0 && pipe->pipeid < 64 && g_rtp_pushers[pipe->pipeid]);
                if (do_rtp_push && (frame_count_per_pipe[pipe->pipeid] % 2) != 1)
                    do_rtp_push = false;
#else
                bool do_rtp_push = (pipe->pipeid >= 0 && pipe->pipeid < 64 && g_rtp_pushers[pipe->pipeid]);
#endif
                if (do_rtp_push) {
                    {  // 每路流獨立鎖，避免一路大 I 幀長時間佔鎖導致其他路卡死與 MediaMTX i/o timeout
                        int pid = (pipe->pipeid >= 0 && pipe->pipeid < 64) ? pipe->pipeid : 0;
                        std::lock_guard<std::mutex> rtp_lock(g_rtp_push_mutex_per_pipe[pid]);
                    static int log_count[64] = {0};
                    if (++log_count[pipe->pipeid] % 30 == 0) {
                        char dest_ip_str[INET_ADDRSTRLEN];
                        inet_ntop(AF_INET, &g_rtp_pushers[pipe->pipeid]->dest_addr.sin_addr, 
                                 dest_ip_str, INET_ADDRSTRLEN);
                        ALOGI("[RTP] Pushing frame for pipeid %d, venc_chn=%d, pusher=%p, dest=%s:%d", 
                              pipe->pipeid, pipe->m_venc_attr.n_venc_chn, g_rtp_pushers[pipe->pipeid],
                              dest_ip_str, ntohs(g_rtp_pushers[pipe->pipeid]->dest_addr.sin_port));
                    }
                    // stStream.stPack 可能包含多個 NAL 單元（用起始碼分隔）
                    // 使用 u32NaluNum 和 stNaluInfo 來獲取每個 NAL 單元的信息
                    // [修復] 使用 per-pipeid 的靜態變量，避免多線程衝突
                    static uint64_t last_pts[64] = {0};
                    static uint32_t last_rtp_timestamp[64] = {0};
                    bool is_new_frame = (stStream.stPack.u64PTS != last_pts[pipe->pipeid] || stStream.stPack.u64PTS == 0);
                    
                    if (is_new_frame) {
                        // 直播推 MediaMTX：一律用牆鐘作為 RTP 時間戳，避免編碼佇列積壓導致 PTS 滯後、接收端緩衝累積造成久播延遲
                        using namespace std::chrono;
                        static steady_clock::time_point base_wall[64];
                        if (g_rtp_pushers[pipe->pipeid]->frame_count == 0) {
                            base_wall[pipe->pipeid] = steady_clock::now();
                            g_rtp_pushers[pipe->pipeid]->rtp_timestamp = 0;
                        } else {
                            auto elapsed_ns = duration_cast<nanoseconds>(steady_clock::now() - base_wall[pipe->pipeid]).count();
                            g_rtp_pushers[pipe->pipeid]->rtp_timestamp = (uint32_t)((elapsed_ns * 90) / 1000000000);
                        }
                        g_rtp_pushers[pipe->pipeid]->frame_count++;
                        last_rtp_timestamp[pipe->pipeid] = g_rtp_pushers[pipe->pipeid]->rtp_timestamp;
                        last_pts[pipe->pipeid] = stStream.stPack.u64PTS;
                    }
                    // 僅在新 PTS 時推送 RTP：同一 PTS 重複送會導致畫面來回跳與拖影，故跳過重複 PTS 的編碼幀
                    if (is_new_frame) {
                    // 記錄 SPS/PPS 的發送（每個 pipeid 獨立）
                    static bool sps_sent[64] = {false};
                    static bool pps_sent[64] = {false};
                    
                    // 如果編碼器提供了 NAL 單元信息，使用它來分割 NAL 單元
                    if (stStream.stPack.u32NaluNum > 0 && stStream.stPack.u32NaluNum <= VENC_MAX_NALU_NUM) {
                        // 使用編碼器提供的 NAL 單元信息
                        if (frame_count_per_pipe[pipe->pipeid] % 30 == 0) {
                            ALOGI("[RTP] Frame #%d: u32NaluNum=%d, total_size=%d", 
                                  frame_count_per_pipe[pipe->pipeid], stStream.stPack.u32NaluNum, stStream.stPack.u32Len);
                        }
                        for (AX_U32 i = 0; i < stStream.stPack.u32NaluNum; i++) {
                            const AX_VENC_NALU_INFO_T* nalu_info = &stStream.stPack.stNaluInfo[i];
                            const uint8_t* nalu_data = stStream.stPack.pu8Addr + nalu_info->u32NaluOffset;
                            uint32_t nalu_size = nalu_info->u32NaluLength;
                            
                            // 檢查數據是否包含起始碼（編碼器可能返回包含起始碼的數據）
                            uint32_t startcode_len = 0;
                            if (nalu_size >= 4 && nalu_data[0] == 0x00 && nalu_data[1] == 0x00) {
                                if (nalu_data[2] == 0x00 && nalu_data[3] == 0x01) {
                                    startcode_len = 4;
                                } else if (nalu_data[2] == 0x01) {
                                    startcode_len = 3;
                                }
                            }
                            
                            // 跳過起始碼
                            if (startcode_len > 0 && nalu_size > startcode_len) {
                                nalu_data += startcode_len;
                                nalu_size -= startcode_len;
                            }
                            
                            // 調試：打印前幾個字節（跳過起始碼後）
                            if (frame_count_per_pipe[pipe->pipeid] % 30 == 0 && i == 0) {
                                ALOGI("[RTP] First NAL data (after startcode): 0x%02X 0x%02X 0x%02X 0x%02X 0x%02X, startcode_len=%d", 
                                      nalu_data[0], nalu_data[1], nalu_data[2], nalu_data[3], nalu_data[4], startcode_len);
                            }
                            
                            // 讀取 NAL type（第一個字節的低 5 位）
                            uint8_t nalu_type = (nalu_data[0] & 0x1F);
                            
                            // 如果 NAL type 仍然是 0，打印更多調試信息
                            if (nalu_type == 0 && (frame_count_per_pipe[pipe->pipeid] % 30 == 0 || i == 0)) {
                                ALOGW("[RTP] Warning: NAL type=0 after skipping startcode, first 16 bytes:");
                                for (int j = 0; j < 16 && j < nalu_size; j++) {
                                    ALOGW("  [%d] = 0x%02X", j, nalu_data[j]);
                                }
                            }
                            bool is_sps = (nalu_type == 7);
                            bool is_pps = (nalu_type == 8);
                            bool is_idr = (nalu_type == 5);
                            bool is_slice = (nalu_type == 1);
                            
                            if (is_sps) {
                                sps_sent[pipe->pipeid] = true;
                                ALOGI("[RTP] Sending SPS (NAL type 7), size=%d", nalu_size);
                            }
                            if (is_pps) {
                                pps_sent[pipe->pipeid] = true;
                                ALOGI("[RTP] Sending PPS (NAL type 8), size=%d", nalu_size);
                            }
                            
                            // 判斷是否為最後一個 NAL
                            bool is_last_nal = (i == stStream.stPack.u32NaluNum - 1) || 
                                               (is_idr || (is_slice && i == stStream.stPack.u32NaluNum - 1));
                            
                            if (frame_count_per_pipe[pipe->pipeid] % 30 == 0 || is_sps || is_pps || is_idr) {
                                ALOGI("[RTP] Pushing NAL #%d/%d: type=%d, size=%d, marker=%d", 
                                      i+1, stStream.stPack.u32NaluNum, nalu_type, nalu_size, is_last_nal ? 1 : 0);
                            }
                            
                            // 發送 NAL 單元
                            int ret = rtp_pusher_push_nalu_no_startcode_with_marker(
                                g_rtp_pushers[pipe->pipeid],
                                nalu_data,
                                nalu_size,
                                stStream.stPack.u64PTS,
                                is_last_nal ? 1 : 0);
                            
                            if (ret != 0 && (frame_count_per_pipe[pipe->pipeid] % 30 == 0 || is_sps || is_pps)) {
                                ALOGE("[RTP] Failed to push NAL #%d, type=%d, ret=%d", i, nalu_type, ret);
                            }
                        }
                    } else {
                        // 編碼器沒有提供 NAL 單元信息，手動解析起始碼
                        // 手動解析起始碼（編碼器沒有提供 NAL 單元信息，或數據格式異常）
                        if (frame_count_per_pipe[pipe->pipeid] % 30 == 0) {
                            ALOGI("[RTP] Frame #%d: using manual parsing, u32NaluNum=%d, total_size=%d", 
                                  frame_count_per_pipe[pipe->pipeid], stStream.stPack.u32NaluNum, stStream.stPack.u32Len);
                            // 打印前幾個字節用於調試
                            if (stStream.stPack.u32Len >= 5) {
                                ALOGI("[RTP] Stream data start: 0x%02X 0x%02X 0x%02X 0x%02X 0x%02X", 
                                      stStream.stPack.pu8Addr[0], stStream.stPack.pu8Addr[1], 
                                      stStream.stPack.pu8Addr[2], stStream.stPack.pu8Addr[3], 
                                      stStream.stPack.pu8Addr[4]);
                            }
                        }
                        
                        const uint8_t* stream_data = stStream.stPack.pu8Addr;
                        uint32_t stream_size = stStream.stPack.u32Len;
                        uint32_t offset = 0;
                        uint32_t nalu_index = 0;
                        
                        while (offset < stream_size) {
                            // 查找起始碼
                            uint32_t startcode_len = 0;
                            if (offset + 3 < stream_size && stream_data[offset] == 0x00 && stream_data[offset+1] == 0x00) {
                                if (stream_data[offset+2] == 0x00 && stream_data[offset+3] == 0x01) {
                                    startcode_len = 4;
                                } else if (stream_data[offset+2] == 0x01) {
                                    startcode_len = 3;
                                }
                            }
                            
                            if (startcode_len == 0) {
                                offset++;
                                continue;
                            }
                            
                            offset += startcode_len;
                            if (offset >= stream_size) break;
                            
                            // 查找下一個起始碼
                            uint32_t nalu_start = offset;
                            uint32_t nalu_end = offset;
                            while (nalu_end < stream_size) {
                                if (nalu_end + 3 < stream_size && 
                                    stream_data[nalu_end] == 0x00 && stream_data[nalu_end+1] == 0x00 &&
                                    (stream_data[nalu_end+2] == 0x01 || 
                                     (stream_data[nalu_end+2] == 0x00 && stream_data[nalu_end+3] == 0x01))) {
                                    break;
                                }
                                nalu_end++;
                            }
                            
                            uint32_t nalu_size = nalu_end - nalu_start;
                            if (nalu_size > 0) {
                                const uint8_t* nalu_data = stream_data + nalu_start;
                                uint8_t nalu_type = (nalu_data[0] & 0x1F);
                                bool is_sps = (nalu_type == 7);
                                bool is_pps = (nalu_type == 8);
                                bool is_idr = (nalu_type == 5);
                                bool is_slice = (nalu_type == 1);
                                
                                if (is_sps) {
                                    sps_sent[pipe->pipeid] = true;
                                    ALOGI("[RTP] Sending SPS (NAL type 7), size=%d", nalu_size);
                                }
                                if (is_pps) {
                                    pps_sent[pipe->pipeid] = true;
                                    ALOGI("[RTP] Sending PPS (NAL type 8), size=%d", nalu_size);
                                }
                                
                                // 判斷是否為最後一個 NAL（檢查是否還有更多起始碼）
                                bool is_last_nal = (nalu_end >= stream_size) || (is_idr || is_slice);
                                
                                if (frame_count_per_pipe[pipe->pipeid] % 30 == 0 || is_sps || is_pps || is_idr) {
                                    ALOGI("[RTP] Pushing parsed NAL #%d: type=%d, size=%d, marker=%d", 
                                          nalu_index+1, nalu_type, nalu_size, is_last_nal ? 1 : 0);
                                }
                                
                                // 發送 NAL 單元
                                int ret = rtp_pusher_push_nalu_no_startcode_with_marker(
                                    g_rtp_pushers[pipe->pipeid],
                                    nalu_data,
                                    nalu_size,
                                    stStream.stPack.u64PTS,
                                    is_last_nal ? 1 : 0);
                                
                                if (ret != 0 && (frame_count_per_pipe[pipe->pipeid] % 30 == 0 || is_sps || is_pps)) {
                                    ALOGE("[RTP] Failed to push parsed NAL #%d, type=%d, ret=%d", 
                                          nalu_index, nalu_type, ret);
                                }
                                
                                nalu_index++;
                            }
                            
                            offset = nalu_end;
                        }
                    }
                    
                    if (frame_count_per_pipe[pipe->pipeid] % 30 == 0) {
                        ALOGI("[RTP] Pushed frame #%d: total_size=%d, NALs=%d, PTS=%llu, TS=%u", 
                              frame_count_per_pipe[pipe->pipeid], stStream.stPack.u32Len, 
                              stStream.stPack.u32NaluNum > 0 ? stStream.stPack.u32NaluNum : 0,
                              stStream.stPack.u64PTS, g_rtp_pushers[pipe->pipeid]->rtp_timestamp);
                    }
                    }  // is_new_frame：僅新 PTS 時推送，避免重複幀
                    }  // 釋放本路 RTP 鎖
                } else if (pipe->pipeid >= 0 && pipe->pipeid < 64 && !g_rtp_pushers[pipe->pipeid]) {
                    static int warn_count[64] = {0};
                    if (warn_count[pipe->pipeid]++ < 5) {
                        ALOGE("[RTP] Cannot push: pipeid=%d, venc_chn=%d, pusher=%p (valid range: 0-63)", 
                              pipe->pipeid, pipe->m_venc_attr.n_venc_chn,
                              (pipe->pipeid >= 0 && pipe->pipeid < 64) ? g_rtp_pushers[pipe->pipeid] : NULL);
                        if (pipe->pipeid >= 0 && pipe->pipeid < 64) {
                            ALOGE("[RTP] g_rtp_pushers[%d] = %p", pipe->pipeid, g_rtp_pushers[pipe->pipeid]);
                        }
                    }
                }
            }
            break;
            default:
                break;
            }

            if (pipe->output_func)
            {
                pipeline_buffer_t buf;
                buf.pipeid = pipe->pipeid;
                buf.m_output_type = pipe->m_output_type;
                buf.n_width = 0;
                buf.n_height = 0;
                buf.n_size = stStream.stPack.u32Len;
                buf.n_stride = 0;
                buf.d_type = po_none;
                buf.p_vir = stStream.stPack.pu8Addr;
                buf.p_phy = stStream.stPack.ulPhyAddr;
                buf.p_pipe = pipe;
                pipe->output_func(&buf);
            }

            s32Ret = AX_VENC_ReleaseStream(pipe->m_venc_attr.n_venc_chn, &stStream);
            if (s32Ret)
            {
                ALOGE("VencChn %d: AX_VENC_ReleaseStream failed!s32Ret:0x%x\n", pipe->m_venc_attr.n_venc_chn, s32Ret);
                usleep(30 * 1000);
            }
        }
        else
        {
            // 錯誤代碼 0x80070222 通常是超時錯誤（緩衝區空），這是正常的
            // 只在非超時錯誤時記錄，並減少日誌頻率
            static int error_count[64] = {0};
            static int last_logged_count[64] = {0};
            int pid = (pipe->pipeid >= 0 && pipe->pipeid < 64) ? pipe->pipeid : 0;
            error_count[pid]++;
            consecutive_fail_count[pid]++;
            
            // 連續無成功取幀時打 log：每 ~10s 一次，以及 ~60s 一次，便於與 MediaMTX i/o timeout 對齊
            if (consecutive_fail_count[pid] > 0 && consecutive_fail_count[pid] % 1000 == 0) {
                int approx_sec = (int)(consecutive_fail_count[pid] * 10 / 1000);
                ALOGW("[VENC] pipeid %d (venc_chn %d): no frame for ~%ds (GetStream failed), possible: VDEC/IVPS no input or encoder error",
                      pipe->pipeid, pipe->m_venc_attr.n_venc_chn, approx_sec);
            }
            
            // 0x80070222 是超時錯誤，其他是真正的錯誤
            if (s32Ret != 0x80070222) {
                // 非超時錯誤，每100次記錄一次（per pipeid）
                if (error_count[pid] - last_logged_count[pid] >= 100) {
                    ALOGW("VencChn %d (pipeid %d): AX_VENC_GetStream failed!s32Ret:0x%x (total errors: %d)\n",
                          pipe->m_venc_attr.n_venc_chn, pipe->pipeid, s32Ret, error_count[pid]);
                    last_logged_count[pid] = error_count[pid];
                }
            }
            // 超時錯誤時使用較短的 sleep，非超時錯誤使用較長的 sleep
            usleep((s32Ret == 0x80070222) ? 10 * 1000 : 30 * 1000);
        }
    }

EXIT:
    ALOGN("[VENC] pipeid %d (venc_chn %d): getStream thread Exit! (n_loog_exit=1, no more RTP from this stream)\n",
          pipe->pipeid, pipe->m_venc_attr.n_venc_chn);
    return NULL;
}

AX_BOOL set_jpeg_param(pipeline_t *pipe)
{
    AX_S32 s32Ret = 0;
    AX_VENC_JPEG_PARAM_T stJpegParam;
    memset(&stJpegParam, 0, sizeof(stJpegParam));
    s32Ret = AX_VENC_GetJpegParam(pipe->m_venc_attr.n_venc_chn, &stJpegParam);
    if (AX_SUCCESS != s32Ret)
    {
        printf("AX_VENC_GetJpegParam:%d failed, error type 0x%x!\n", pipe->m_venc_attr.n_venc_chn, s32Ret);
        return AX_FALSE;
    }

    stJpegParam.u32Qfactor = 90;
    /* Use user set qtable. Qtable example */
    // if (gS32QTableEnable)
    // {
    //     memcpy(stJpegParam.u8YQt, QTableLuminance, sizeof(QTableLuminance));
    //     memcpy(stJpegParam.u8CbCrQt, QTableChrominance, sizeof(QTableChrominance));
    // }

    s32Ret = AX_VENC_SetJpegParam(pipe->m_venc_attr.n_venc_chn, &stJpegParam);
    if (AX_SUCCESS != s32Ret)
    {
        printf("AX_VENC_SetJpegParam:%d failed, error type 0x%x!\n", pipe->m_venc_attr.n_venc_chn, s32Ret);
        return AX_FALSE;
    }

    return AX_TRUE;
}

AX_BOOL set_rc_param(pipeline_t *pipe, AX_VENC_RC_MODE_E enRcMode)
{
    // #ifdef VIDEO_ENABLE_RC_DYNAMIC
    AX_S32 s32Ret = 0;
    AX_VENC_RC_PARAM_T stRcParam;

    s32Ret = AX_VENC_GetRcParam(pipe->m_venc_attr.n_venc_chn, &stRcParam);
    if (AX_SUCCESS != s32Ret)
    {
        printf("AX_VENC_GetRcParam:%d failed, error type 0x%x!\n", pipe->m_venc_attr.n_venc_chn, s32Ret);
        return AX_FALSE;
    }

    if (enRcMode == AX_VENC_RC_MODE_MJPEGCBR)
    {
        stRcParam.stMjpegCbr.u32BitRate = 4000;
        stRcParam.stMjpegCbr.u32MinQp = 20;
        stRcParam.stMjpegCbr.u32MaxQp = 30;
    }
    else if (enRcMode == AX_VENC_RC_MODE_MJPEGVBR)
    {
        stRcParam.stMjpegVbr.u32MaxBitRate = 4000;
        stRcParam.stMjpegVbr.u32MinQp = 20;
        stRcParam.stMjpegVbr.u32MaxQp = 30;
    }
    else if (enRcMode == AX_VENC_RC_MODE_MJPEGFIXQP)
    {
        stRcParam.stMjpegFixQp.s32FixedQp = 22;
    }

    s32Ret = AX_VENC_SetRcParam(pipe->m_venc_attr.n_venc_chn, &stRcParam);
    if (AX_SUCCESS != s32Ret)
    {
        printf("AX_VENC_SetRcParam:%d failed, error type 0x%x!\n", pipe->m_venc_attr.n_venc_chn, s32Ret);
        return AX_FALSE;
    }

    // #endif
    return AX_TRUE;
}

int _create_venc_chn(pipeline_t *pipe)
{
    if (pipe->m_venc_attr.n_venc_chn > MAX_VENC_CHN_COUNT)
    {
        ALOGE("venc_chn must lower than %d, got %d\n", MAX_VENC_CHN_COUNT, pipe->m_venc_attr.n_venc_chn);
        return -1;
    }
    typedef struct _stRCInfo
    {
        SAMPLE_VENC_RC_E eRCType;
        AX_U32 nMinQp;
        AX_U32 nMaxQp;
        AX_U32 nMinIQp;
        AX_U32 nMaxIQp;
        AX_S32 nIntraQpDelta;
    } RC_INFO_T;
    typedef struct _stVideoConfig
    {
        AX_PAYLOAD_TYPE_E ePayloadType;
        AX_U32 nGOP;
        AX_U32 nSrcFrameRate;
        AX_U32 nDstFrameRate;
        AX_U32 nStride;
        AX_S32 nInWidth;
        AX_S32 nInHeight;
        AX_S32 nOutWidth;
        AX_S32 nOutHeight;
        AX_S32 nOffsetCropX;
        AX_S32 nOffsetCropY;
        AX_S32 nOffsetCropW;
        AX_S32 nOffsetCropH;
        AX_IMG_FORMAT_E eImgFormat;
        RC_INFO_T stRCInfo;
        AX_S32 nBitrate;
    } VIDEO_CONFIG_T;
    AX_VENC_CHN_ATTR_T stVencChnAttr;
    VIDEO_CONFIG_T config;
    memset(&config, 0, sizeof(VIDEO_CONFIG_T));
    // AX_S32 s32Ret = 0;

    // [32 路規格] MediaMTX 輸出：單路 1.2～2.0 Mbps、GOP 30（VBR；CBR 在部分 AX 版本會導致 AX_VENC_CreateChn 0x8007020a）
    const bool is_mediamtx = (pipe->m_output_type == po_mediamtx_h264 || pipe->m_output_type == po_mediamtx_h265);
    if (is_mediamtx) {
        config.stRCInfo.eRCType = SAMPLE_RC_VBR;  // 維持 VBR，避免 AX_VENC_CreateChn 失敗 0x8007020a
        config.nGOP = 30;
        AX_S32 raw_kbps = (AX_S32)((int64_t)pipe->m_ivps_attr.n_ivps_width * pipe->m_ivps_attr.n_ivps_height * 5 / 1024);
        const AX_S32 cap_kbps = 2000;       // 32 路：單路上限 2 Mbps，總碼率 ≤ 64 Mbps
        const AX_S32 min_mediamtx_kbps = 1200;
        if (raw_kbps > cap_kbps) raw_kbps = cap_kbps;
        if (raw_kbps < min_mediamtx_kbps) raw_kbps = min_mediamtx_kbps;
        config.nBitrate = raw_kbps;
        ALOGN("[VENC] MediaMTX output (32ch target): bitrate=%d kbps, GOP=%d (pipeid=%d)", config.nBitrate, config.nGOP, pipe->pipeid);
    } else {
        config.stRCInfo.eRCType = SAMPLE_RC_VBR;
        config.nGOP = 15;
        AX_S32 raw_kbps = (AX_S32)((int64_t)pipe->m_ivps_attr.n_ivps_width * pipe->m_ivps_attr.n_ivps_height * 5 / 1024);
        const AX_S32 min_kbps = 4000;
        const AX_S32 max_kbps = 160000;
        if (raw_kbps < min_kbps) raw_kbps = min_kbps;
        if (raw_kbps > max_kbps) raw_kbps = max_kbps;
        config.nBitrate = raw_kbps;
    }
    config.stRCInfo.nMinQp = 10;
    config.stRCInfo.nMaxQp = 45;  // 降低最大 QP，減少高動態時的塊狀感（原 51 易起格子）
    config.stRCInfo.nMinIQp = 10;
    config.stRCInfo.nMaxIQp = 45;
    config.stRCInfo.nIntraQpDelta = -2;
    config.nOffsetCropX = 0;
    config.nOffsetCropY = 0;
    config.nOffsetCropW = 0;
    config.nOffsetCropH = 0;
    switch (pipe->m_output_type)
    {
    case po_venc_h264:
    case po_rtsp_h264:
    case po_mediamtx_h264:
        config.ePayloadType = PT_H264;
        break;
    case po_venc_h265:
    case po_rtsp_h265:
    case po_mediamtx_h265:
        config.ePayloadType = PT_H265;
        break;
    case po_venc_mjpg:
        config.ePayloadType = PT_MJPEG;
        break;
    default:
        // ALOGE("pipeline_output_e=%d,should not init venc");
        return -1;
    }

    config.nInWidth = pipe->m_ivps_attr.n_ivps_width;
    config.nInHeight = pipe->m_ivps_attr.n_ivps_height;
    config.nStride = config.nInWidth;

    switch (pipe->m_ivps_attr.n_ivps_rotate)
    {
    case AX_IVPS_ROTATION_90:
    case AX_IVPS_ROTATION_270:
        config.nInWidth = pipe->m_ivps_attr.n_ivps_height;
        config.nInHeight = pipe->m_ivps_attr.n_ivps_width;
        config.nStride = config.nInWidth;
        break;

    default:
        break;
    }

    config.nSrcFrameRate = pipe->m_ivps_attr.n_ivps_fps;
    config.nDstFrameRate = pipe->m_ivps_attr.n_ivps_fps;

    memset(&stVencChnAttr, 0, sizeof(AX_VENC_CHN_ATTR_T));

    stVencChnAttr.stVencAttr.u32MaxPicWidth = config.nInWidth;
    stVencChnAttr.stVencAttr.u32MaxPicHeight = config.nInHeight;

    stVencChnAttr.stVencAttr.u32PicWidthSrc = config.nInWidth;   /*the picture width*/
    stVencChnAttr.stVencAttr.u32PicHeightSrc = config.nInHeight; /*the picture height*/
    stVencChnAttr.stVencAttr.stCropCfg.bEnable = AX_FALSE;
    // stVencChnAttr.stVencAttr.u32CropOffsetX = config.nOffsetCropX;
    // stVencChnAttr.stVencAttr.u32CropOffsetY = config.nOffsetCropY;
    // stVencChnAttr.stVencAttr.u32CropWidth = config.nOffsetCropW;
    // stVencChnAttr.stVencAttr.u32CropHeight = config.nOffsetCropH;
    // stVencChnAttr.stVencAttr.u32VideoRange = 1; /* 0: Narrow Range(NR), Y[16,235], Cb/Cr[16,240]; 1: Full Range(FR), Y/Cb/Cr[0,255] */

    // ALOGN("VencChn %d:w:%d, h:%d, s:%d, Crop:(%d, %d, %d, %d) rcType:%d, payload:%d", gVencChnMapping[VencChn], stVencChnAttr.stVencAttr.u32PicWidthSrc, stVencChnAttr.stVencAttr.u32PicHeightSrc, config.nStride, stVencChnAttr.stVencAttr.u32CropOffsetX, stVencChnAttr.stVencAttr.u32CropOffsetY, stVencChnAttr.stVencAttr.u32CropWidth, stVencChnAttr.stVencAttr.u32CropHeight, config.stRCInfo.eRCType, config.ePayloadType);

    stVencChnAttr.stVencAttr.u32BufSize = config.nStride * config.nInHeight * 3 / 2; /*stream buffer size*/
    // stVencChnAttr.stVencAttr.u32MbLinesPerSlice = 0;                                 /*get stream mode is slice mode or frame mode?*/
    stVencChnAttr.stVencAttr.enLinkMode = AX_VENC_LINK_MODE;
    /* GOP Setting */
    stVencChnAttr.stGopAttr.enGopMode = AX_VENC_GOPMODE_NORMALP;
    // 降低 FIFO 以減少 RTP/WebRTC 端到端延遲（原 5+5 約 333ms@30fps）
    stVencChnAttr.stVencAttr.u8InFifoDepth = 1;
    stVencChnAttr.stVencAttr.u8OutFifoDepth = 2;

    stVencChnAttr.stVencAttr.enType = config.ePayloadType;

#if 0                                                                          // V1.27
    stVencChnAttr.stRcAttr.uFrameRate.tFrmRateCtrl.nSrcFrameRate = AX_FRAME_RATE(config.nSrcFrameRate);  /* input frame rate */
    stVencChnAttr.stRcAttr.uFrameRate.tFrmRateCtrl.nDstFrameRate = AX_FRAME_RATE(config.nDstFrameRate); /* target frame rate */
#else                                                                          // V1.40
    stVencChnAttr.stRcAttr.stFrameRate.fSrcFrameRate = (config.nSrcFrameRate); /* input frame rate */
    stVencChnAttr.stRcAttr.stFrameRate.fDstFrameRate = (config.nDstFrameRate); /* target frame rate */
#endif

    switch (stVencChnAttr.stVencAttr.enType)
    {
    case PT_H265:
    {
        stVencChnAttr.stVencAttr.enProfile = AX_VENC_HEVC_MAIN_PROFILE;
        stVencChnAttr.stVencAttr.enLevel = AX_VENC_HEVC_LEVEL_6;
        stVencChnAttr.stVencAttr.enTier = AX_VENC_HEVC_MAIN_TIER;

        if (config.stRCInfo.eRCType == SAMPLE_RC_CBR)
        {
            AX_VENC_H265_CBR_T stH265Cbr;
            memset(&stH265Cbr, 0, sizeof(stH265Cbr));
            stVencChnAttr.stRcAttr.enRcMode = AX_VENC_RC_MODE_H265CBR;
            stVencChnAttr.stRcAttr.s32FirstFrameStartQp = -1;
            stH265Cbr.u32Gop = config.nGOP;
            // stH265Cbr.u32SrcFrameRate = config.nSrcFrameRate;  /* input frame rate */
            // stH265Cbr.fr32DstFrameRate = config.nDstFrameRate; /* target frame rate */
            stH265Cbr.u32BitRate = config.nBitrate;
            stH265Cbr.u32MinQp = config.stRCInfo.nMinQp;
            stH265Cbr.u32MaxQp = config.stRCInfo.nMaxQp;
            stH265Cbr.u32MinIQp = config.stRCInfo.nMinIQp;
            stH265Cbr.u32MaxIQp = config.stRCInfo.nMaxIQp;
            stH265Cbr.s32IntraQpDelta = config.stRCInfo.nIntraQpDelta;
            memcpy(&stVencChnAttr.stRcAttr.stH265Cbr, &stH265Cbr, sizeof(AX_VENC_H265_CBR_T));
        }
        else if (config.stRCInfo.eRCType == SAMPLE_RC_VBR)
        {
            AX_VENC_H265_VBR_T stH265Vbr;
            memset(&stH265Vbr, 0, sizeof(stH265Vbr));
            stVencChnAttr.stRcAttr.enRcMode = AX_VENC_RC_MODE_H265VBR;
            stVencChnAttr.stRcAttr.s32FirstFrameStartQp = -1;
            stH265Vbr.u32Gop = config.nGOP;
            // stH265Vbr.u32SrcFrameRate = config.nSrcFrameRate;
            // stH265Vbr.fr32DstFrameRate = config.nDstFrameRate;
            stH265Vbr.u32MaxBitRate = config.nBitrate;
            stH265Vbr.u32MinQp = config.stRCInfo.nMinQp;
            stH265Vbr.u32MaxQp = config.stRCInfo.nMaxQp;
            stH265Vbr.u32MinIQp = config.stRCInfo.nMinIQp;
            stH265Vbr.u32MaxIQp = config.stRCInfo.nMaxIQp;
            stH265Vbr.s32IntraQpDelta = config.stRCInfo.nIntraQpDelta;
            memcpy(&stVencChnAttr.stRcAttr.stH265Vbr, &stH265Vbr, sizeof(AX_VENC_H265_VBR_T));
        }
        else if (config.stRCInfo.eRCType == SAMPLE_RC_FIXQP)
        {
            AX_VENC_H265_FIXQP_T stH265FixQp;
            memset(&stH265FixQp, 0, sizeof(stH265FixQp));
            stVencChnAttr.stRcAttr.enRcMode = AX_VENC_RC_MODE_H265FIXQP;
            stH265FixQp.u32Gop = config.nGOP;
            // stH265FixQp.u32SrcFrameRate = config.nSrcFrameRate;
            // stH265FixQp.fr32DstFrameRate = config.nDstFrameRate;
            stH265FixQp.u32IQp = 25;
            stH265FixQp.u32PQp = 30;
            stH265FixQp.u32BQp = 32;
            memcpy(&stVencChnAttr.stRcAttr.stH265FixQp, &stH265FixQp, sizeof(AX_VENC_H265_FIXQP_T));
        }
        break;
    }
    case PT_H264:
    {
        stVencChnAttr.stVencAttr.enProfile = AX_VENC_H264_MAIN_PROFILE;
        stVencChnAttr.stVencAttr.enLevel = AX_VENC_H264_LEVEL_5_2;

        if (config.stRCInfo.eRCType == SAMPLE_RC_CBR)
        {
            AX_VENC_H264_CBR_T stH264Cbr;
            memset(&stH264Cbr, 0, sizeof(stH264Cbr));
            stVencChnAttr.stRcAttr.enRcMode = AX_VENC_RC_MODE_H264CBR;
            stVencChnAttr.stRcAttr.s32FirstFrameStartQp = -1;
            stH264Cbr.u32Gop = config.nGOP;
            // stH264Cbr.u32SrcFrameRate = config.nSrcFrameRate;  /* input frame rate */
            // stH264Cbr.fr32DstFrameRate = config.nDstFrameRate; /* target frame rate */
            stH264Cbr.u32BitRate = config.nBitrate;
            stH264Cbr.u32MinQp = config.stRCInfo.nMinQp;
            stH264Cbr.u32MaxQp = config.stRCInfo.nMaxQp;
            stH264Cbr.u32MinIQp = config.stRCInfo.nMinIQp;
            stH264Cbr.u32MaxIQp = config.stRCInfo.nMaxIQp;
            stH264Cbr.s32IntraQpDelta = config.stRCInfo.nIntraQpDelta;
            memcpy(&stVencChnAttr.stRcAttr.stH264Cbr, &stH264Cbr, sizeof(AX_VENC_H264_CBR_T));
        }
        else if (config.stRCInfo.eRCType == SAMPLE_RC_VBR)
        {
            AX_VENC_H264_VBR_T stH264Vbr;
            memset(&stH264Vbr, 0, sizeof(stH264Vbr));
            stVencChnAttr.stRcAttr.enRcMode = AX_VENC_RC_MODE_H264VBR;
            stVencChnAttr.stRcAttr.s32FirstFrameStartQp = -1;
            stH264Vbr.u32Gop = config.nGOP;
            // stH264Vbr.u32SrcFrameRate = config.nSrcFrameRate;
            // stH264Vbr.fr32DstFrameRate = config.nDstFrameRate;
            stH264Vbr.u32MaxBitRate = config.nBitrate;
            stH264Vbr.u32MinQp = config.stRCInfo.nMinQp;
            stH264Vbr.u32MaxQp = config.stRCInfo.nMaxQp;
            stH264Vbr.u32MinIQp = config.stRCInfo.nMinIQp;
            stH264Vbr.u32MaxIQp = config.stRCInfo.nMaxIQp;
            stH264Vbr.s32IntraQpDelta = config.stRCInfo.nIntraQpDelta;
            memcpy(&stVencChnAttr.stRcAttr.stH264Vbr, &stH264Vbr, sizeof(AX_VENC_H264_VBR_T));
        }
        else if (config.stRCInfo.eRCType == SAMPLE_RC_FIXQP)
        {
            AX_VENC_H264_FIXQP_T stH264FixQp;
            memset(&stH264FixQp, 0, sizeof(stH264FixQp));
            stVencChnAttr.stRcAttr.enRcMode = AX_VENC_RC_MODE_H264FIXQP;
            stH264FixQp.u32Gop = config.nGOP;
            // stH264FixQp.u32SrcFrameRate = config.nSrcFrameRate;
            // stH264FixQp.fr32DstFrameRate = config.nDstFrameRate;
            stH264FixQp.u32IQp = 25;
            stH264FixQp.u32PQp = 30;
            stH264FixQp.u32BQp = 32;
            memcpy(&stVencChnAttr.stRcAttr.stH264FixQp, &stH264FixQp, sizeof(AX_VENC_H264_FIXQP_T));
        }
        break;
    }
    case PT_MJPEG:
    {
        if (config.stRCInfo.eRCType == SAMPLE_RC_CBR)
        {
            AX_VENC_MJPEG_CBR_T stMjpegCbrAttr;
            memset(&stMjpegCbrAttr, 0, sizeof(stMjpegCbrAttr));
            stVencChnAttr.stRcAttr.enRcMode = AX_VENC_RC_MODE_MJPEGCBR;
            // stVencChnAttr.stRcAttr.s32FirstFrameStartQp = -1;

            stMjpegCbrAttr.u32StatTime = 1;
            // stMjpegCbrAttr.u32SrcFrameRate = config.nSrcFrameRate;
            // stMjpegCbrAttr.fr32DstFrameRate = config.nDstFrameRate;
            stMjpegCbrAttr.u32BitRate = 4000;
            stMjpegCbrAttr.u32MinQp = 20;
            stMjpegCbrAttr.u32MaxQp = 30;
            memcpy(&stVencChnAttr.stRcAttr.stMjpegCbr, &stMjpegCbrAttr, sizeof(AX_VENC_MJPEG_CBR_T));
        }
        else if (config.stRCInfo.eRCType == SAMPLE_RC_VBR)
        {
            AX_VENC_MJPEG_VBR_T stMjpegVbrAttr;
            memset(&stMjpegVbrAttr, 0, sizeof(stMjpegVbrAttr));
            stVencChnAttr.stRcAttr.enRcMode = AX_VENC_RC_MODE_MJPEGVBR;
            // stVencChnAttr.stRcAttr.s32FirstFrameStartQp = -1;
            stMjpegVbrAttr.u32StatTime = 1;
            // stMjpegVbrAttr.u32SrcFrameRate = config.nSrcFrameRate;
            // stMjpegVbrAttr.fr32DstFrameRate = config.nDstFrameRate;
            stMjpegVbrAttr.u32MaxBitRate = 4000;
            stMjpegVbrAttr.u32MinQp = 20;
            stMjpegVbrAttr.u32MaxQp = 30;
            memcpy(&stVencChnAttr.stRcAttr.stMjpegVbr, &stMjpegVbrAttr, sizeof(AX_VENC_MJPEG_VBR_T));
        }
        else if (config.stRCInfo.eRCType == SAMPLE_RC_FIXQP)
        {
            AX_VENC_MJPEG_FIXQP_T stMjpegFixQpAttr;
            memset(&stMjpegFixQpAttr, 0, sizeof(stMjpegFixQpAttr));
            stVencChnAttr.stRcAttr.enRcMode = AX_VENC_RC_MODE_MJPEGFIXQP;

            // stMjpegFixQpAttr.u32SrcFrameRate = config.nSrcFrameRate;
            // stMjpegFixQpAttr.fr32DstFrameRate = config.nDstFrameRate;
            stMjpegFixQpAttr.s32FixedQp = 22;
            memcpy(&stVencChnAttr.stRcAttr.stMjpegFixQp, &stMjpegFixQpAttr, sizeof(AX_VENC_MJPEG_FIXQP_T));
        }
        break;
    }
    default:
        ALOGE("VencChn %d:Payload type unrecognized.", pipe->m_venc_attr.n_venc_chn);
        return -1;
    }

    ALOGI("[VENC] Creating VENC channel %d for pipeid %d, width=%d, height=%d", 
          pipe->m_venc_attr.n_venc_chn, pipe->pipeid, config.nOutWidth, config.nOutHeight);
    AX_S32 ret = AX_VENC_CreateChn(pipe->m_venc_attr.n_venc_chn, &stVencChnAttr);
    if (AX_SUCCESS != ret)
    {
        ALOGE("[VENC] VencChn %d: AX_VENC_CreateChn failed for pipeid %d, s32Ret:0x%x", 
              pipe->m_venc_attr.n_venc_chn, pipe->pipeid, ret);
        return -1;
    }
    ALOGI("[VENC] VENC channel %d created successfully for pipeid %d", 
          pipe->m_venc_attr.n_venc_chn, pipe->pipeid);

    if (pipe->m_output_type == po_venc_mjpg)
    {
        set_rc_param(pipe, stVencChnAttr.stRcAttr.enRcMode);
        set_jpeg_param(pipe);
    }

    if (0 != pthread_create(&pipe->m_venc_attr.tid, NULL, _venc_get_frame_thread, pipe))
    {
        return -1;
    }

    return 0;
}

int _destore_venc_grp(pipeline_t *pipe)
{
    AX_S32 s32Ret = 0;
    if (pipe->m_venc_attr.tid)
    {
        pthread_join(pipe->m_venc_attr.tid, NULL);
    }

    s32Ret = AX_VENC_StopRecvFrame(pipe->m_venc_attr.n_venc_chn);
    if (0 != s32Ret)
    {
        ALOGE("VencChn %d:AX_VENC_StopRecvFrame failed,s32Ret:0x%x.\n", pipe->m_venc_attr.n_venc_chn, s32Ret);
        return s32Ret;
    }

    s32Ret = AX_VENC_DestroyChn(pipe->m_venc_attr.n_venc_chn);
    if (0 != s32Ret)
    {
        ALOGE("VencChn %d:AX_VENC_DestroyChn failed,s32Ret:0x%x.\n", pipe->m_venc_attr.n_venc_chn, s32Ret);
        return s32Ret;
    }
    return 0;
}
