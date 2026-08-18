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
#include "../../utilities/net_utils.h"
#include <errno.h>
#include <string.h>
#include <mutex>

#include "ax_ivps_api.h"
#include "ax_vdec_api.h"
#include "ax_venc_api.h"

#include "algorithm"
#include "vector"
#include "string"
#include "string.h"
#include "map"
#include "unistd.h"

typedef struct
{

    std::map<int, pipeline_t *> pipeid_pipe;

    bool b_maix3_init = false;
    bool b_hdmi_init = false;

    std::vector<int> ivps_grp;
    std::vector<int> vdec_grp;
    std::vector<int> venc_chn;
} pipeline_internal_handle_t;

static pipeline_internal_handle_t pipeline_handle;
/** 保護 pipeid_pipe 的並發存取：create/destory_pipeline（主/配置線程）與 user_input（demux 線程） */
static std::mutex g_pipeline_handle_mutex;

template <typename T>
bool contain(std::vector<T> &v, T &t)
{
    auto item = std::find(v.begin(), v.end(), t);
    if (item != v.end())
    {
        return true;
    }
    return false;
}

template <typename KT, typename VT>
bool contain(std::map<KT, VT> &v, KT &t)
{
    auto item = v.find(t);
    if (item != v.end())
    {
        return true;
    }
    return false;
}

template <typename T>
bool erase(std::vector<T> &v, T &t)
{
    auto item = std::find(v.begin(), v.end(), t);
    if (item != v.end())
    {
        v.erase(item);
        return true;
    }
    return false;
}

template <typename KT, typename VT>
bool erase(std::map<KT, VT> &v, KT &t)
{
    auto item = v.find(t);
    if (item != v.end())
    {
        v.erase(item);
        return true;
    }
    return false;
}

// RTP 推送器管理（在 venc 文件中定義）
extern rtp_pusher_t* g_rtp_pushers[64];

int _create_vo(char *pStr, pipeline_t *pipe);
void _destory_vo();
int _create_vo_hdmi(pipeline_t *pipe);
int _destory_vo_hdmi(pipeline_t *pipe);
int _create_ivps_grp(pipeline_t *pipe);
int _destore_ivps_grp(pipeline_t *pipe);
int _create_venc_chn(pipeline_t *pipe);
int _destore_venc_grp(pipeline_t *pipe);

int _create_vdec_grp(pipeline_t *pipe);
int _destore_vdec_grp(pipeline_t *pipe);
int _create_jvdec_grp(pipeline_t *pipe);
int _destore_jvdec_grp(pipeline_t *pipe);

int create_pipeline(pipeline_t *pipe)
{
    if (!pipe)
    {
        ALOGE("invalid pipeline_t ptr");
        return -1;
    }

    if (pipe->enable == 0)
    {
        ALOGE("PIPE-%d doesn`t enable", pipe->pipeid);
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(g_pipeline_handle_mutex);
        if (contain(pipeline_handle.pipeid_pipe, pipe->pipeid))
        {
            ALOGE("PIPE-%d has been create", pipe->pipeid);
            return -1;
        }
        pipeline_handle.pipeid_pipe[pipe->pipeid] = pipe;
    }

    switch (pipe->m_input_type)
    {
    case pi_user:
    {
        if (pipeline_handle.ivps_grp.size() == 0)
        {
            int s32Ret = AX_IVPS_Init();
            if (0 != s32Ret)
            {
                ALOGE("AX_IVPS_Init failed,s32Ret:0x%x\n", s32Ret);
                return s32Ret;
            }
        }
        if (contain(pipeline_handle.ivps_grp, pipe->m_ivps_attr.n_ivps_grp))
        {
            ALOGE("IVPS-%d has been create", pipe->m_ivps_attr.n_ivps_grp);
            return -1;
        }
        pipeline_handle.ivps_grp.push_back(pipe->m_ivps_attr.n_ivps_grp);
        int s32Ret = _create_ivps_grp(pipe);
        if (AX_SUCCESS != s32Ret)
        {
            ALOGE("_create_ivps_grp failed,s32Ret:0x%x\n", s32Ret);
            return -1;
        }
    }
    break;
    case pi_vin:
    {
        if (pipe->n_vin_pipe > MAX_VIN_PIPE_COUNT)
        {
            ALOGE("vin_pipe must lower than %d, got %d\n", MAX_VIN_PIPE_COUNT, pipe->n_vin_pipe);
            return -1;
        }
        if (pipe->n_vin_chn > MAX_VIN_CHN_COUNT)
        {
            ALOGE("vin_chn must lower than %d, got %d\n", MAX_VIN_CHN_COUNT, pipe->n_vin_chn);
            return -1;
        }

        if (pipeline_handle.ivps_grp.size() == 0)
        {
            int s32Ret = AX_IVPS_Init();
            if (0 != s32Ret)
            {
                ALOGE("AX_IVPS_Init failed,s32Ret:0x%x\n", s32Ret);
                return s32Ret;
            }
        }
        if (contain(pipeline_handle.ivps_grp, pipe->m_ivps_attr.n_ivps_grp))
        {
            ALOGE("IVPS-%d has been create", pipe->m_ivps_attr.n_ivps_grp);
            return -1;
        }
        pipeline_handle.ivps_grp.push_back(pipe->m_ivps_attr.n_ivps_grp);

        int s32Ret = _create_ivps_grp(pipe);
        if (AX_SUCCESS != s32Ret)
        {
            ALOGE("_create_ivps_grp failed,s32Ret:0x%x\n", s32Ret);
            return -1;
        }

        AX_MOD_INFO_T srcMod, dstMod;
        srcMod.enModId = AX_ID_VIN;
        srcMod.s32GrpId = pipe->n_vin_pipe;
        srcMod.s32ChnId = pipe->n_vin_chn;

        dstMod.enModId = AX_ID_IVPS;
        dstMod.s32GrpId = pipe->m_ivps_attr.n_ivps_grp;
        dstMod.s32ChnId = 0;
        AX_SYS_Link(&srcMod, &dstMod);
    }
    break;
    case pi_vdec_h264:
    case pi_vdec_jpeg:
    {
        if (pipeline_handle.ivps_grp.size() == 0)
        {
            int s32Ret = AX_IVPS_Init();
            if (0 != s32Ret)
            {
                ALOGE("AX_IVPS_Init failed,s32Ret:0x%x\n", s32Ret);
                return s32Ret;
            }
        }
        if (contain(pipeline_handle.ivps_grp, pipe->m_ivps_attr.n_ivps_grp))
        {
            ALOGE("IVPS-%d has been create", pipe->m_ivps_attr.n_ivps_grp);
            return -1;
        }
        pipeline_handle.ivps_grp.push_back(pipe->m_ivps_attr.n_ivps_grp);

        int s32Ret = _create_ivps_grp(pipe);
        if (AX_SUCCESS != s32Ret)
        {
            ALOGE("_create_ivps_grp failed,s32Ret:0x%x\n", s32Ret);
            return -1;
        }

        if (pipeline_handle.vdec_grp.size() == 0)
        {
            AX_VDEC_MOD_ATTR_T stModAttr = {0};
            stModAttr.enDecModule = AX_ENABLE_BOTH_VDEC_JDEC;
            stModAttr.u32MaxGroupCount = AX_VDEC_MAX_GRP_NUM;
            int s32Ret = AX_VDEC_Init(&stModAttr);
            if (0 != s32Ret)
            {
                ALOGE("AX_VDEC_Init failed,s32Ret:0x%x\n", s32Ret);
                return s32Ret;
            }
        }

        if (!contain(pipeline_handle.vdec_grp, pipe->m_vdec_attr.n_vdec_grp))
        {
            int s32Ret = _create_vdec_grp(pipe);
            if (AX_SUCCESS != s32Ret)
            {
                ALOGE("_create_vdec_grp failed,s32Ret:0x%x pipe->m_vdec_attr.n_vdec_grp=%d\n", s32Ret, pipe->m_vdec_attr.n_vdec_grp);
                return -1;
            }
            pipeline_handle.vdec_grp.push_back(pipe->m_vdec_attr.n_vdec_grp);
        }
#if VDEC_LINK_MODE
        AX_MOD_INFO_T srcMod, dstMod;
        srcMod.enModId = AX_ID_VDEC;
        srcMod.s32GrpId = pipe->m_vdec_attr.n_vdec_grp;
        srcMod.s32ChnId = 0;

        dstMod.enModId = AX_ID_IVPS;
        dstMod.s32GrpId = pipe->m_ivps_attr.n_ivps_grp;
        dstMod.s32ChnId = 0;
        AX_SYS_Link(&srcMod, &dstMod);
#endif
    }
    break;
    default:
        break;
    }

    switch (pipe->m_output_type)
    {
    case po_venc_mjpg:
    case po_venc_h264:
    case po_venc_h265:
    case po_rtsp_h264:
    case po_rtsp_h265:
    case po_mediamtx_h264:
    case po_mediamtx_h265:
    {
        if (pipeline_handle.venc_chn.size() == 0)
        {
            AX_VENC_MOD_ATTR_T stModAttr;
            memset(&stModAttr, 0, sizeof(AX_VENC_MOD_ATTR_T));
            stModAttr.enVencType = AX_VENC_VIDEO_ENCODER;
            stModAttr.stModThdAttr.u32TotalThreadNum = 1;
            stModAttr.stModThdAttr.bExplicitSched = AX_FALSE;

            int s32Ret = AX_VENC_Init(&stModAttr);
            if (AX_SUCCESS != s32Ret)
            {
                ALOGE("AX_VENC_Init failed, s32Ret:0x%x", s32Ret);
                return s32Ret;
            }
        }

        if (contain(pipeline_handle.venc_chn, pipe->m_venc_attr.n_venc_chn))
        {
            ALOGE("VENC-%d has been create (pipeid=%d), skipping creation", pipe->m_venc_attr.n_venc_chn, pipe->pipeid);
            // 注意：如果 VENC channel 已經存在，可能是因為之前的 pipeline 沒有正確清理
            // 但我們仍然需要初始化 RTP pusher，所以不返回錯誤，繼續執行
            // return -1;
        } else {
            pipeline_handle.venc_chn.push_back(pipe->m_venc_attr.n_venc_chn);
            ALOGI("[Pipeline] VENC channel %d registered for pipeid %d", pipe->m_venc_attr.n_venc_chn, pipe->pipeid);
        }

        AX_MOD_INFO_T srcMod, dstMod;
        srcMod.enModId = AX_ID_IVPS;
        srcMod.s32GrpId = pipe->m_ivps_attr.n_ivps_grp;
        srcMod.s32ChnId = 0;

        dstMod.enModId = AX_ID_VENC;
        dstMod.s32GrpId = 0;
        dstMod.s32ChnId = pipe->m_venc_attr.n_venc_chn;
        AX_SYS_Link(&srcMod, &dstMod);
        // }

        // pipeline_handle.b_init_venc++;

        ALOGI("[Pipeline] Creating VENC channel %d for pipeid %d, end_point=%s", 
              pipe->m_venc_attr.n_venc_chn, pipe->pipeid, pipe->m_venc_attr.end_point);
        int s32Ret = _create_venc_chn(pipe);
        if (AX_SUCCESS != s32Ret)
        {
            ALOGE("[Pipeline] _create_venc_chn failed for pipeid %d, venc_chn=%d, s32Ret:0x%x", 
                  pipe->pipeid, pipe->m_venc_attr.n_venc_chn, s32Ret);
            return -1;
        }
        
        ALOGI("[Pipeline] VENC channel %d created successfully for pipeid %d, output_type=%d, end_point=%s", 
              pipe->m_venc_attr.n_venc_chn, pipe->pipeid, pipe->m_output_type, pipe->m_venc_attr.end_point);

        // 初始化 RTP 推送器（直接推送 NAL 單元到 MediaMTX）
        ALOGI("[Pipeline] Checking if output_type matches MediaMTX: %d == %d or %d?", 
              pipe->m_output_type, po_mediamtx_h264, po_mediamtx_h265);
        if (pipe->m_output_type == po_mediamtx_h264 || pipe->m_output_type == po_mediamtx_h265)
        {
            ALOGI("[RTP] Initializing RTP pusher for pipeid %d, output_type=%d", pipe->pipeid, pipe->m_output_type);
            if (pipe->pipeid >= 0 && pipe->pipeid < 64) {
                // end_point 格式：IP:PORT 或 IP（預設端口 8000）
                // 例如: "192.168.1.100:8000" 或 "192.168.1.100"
                std::string endpoint = pipe->m_venc_attr.end_point;
                if (endpoint.empty()) {
                    endpoint = "127.0.0.1:8000";
                    ALOGI("[RTP] Using default endpoint: %s", endpoint.c_str());
                } else {
                    ALOGI("[RTP] Using configured endpoint: %s", endpoint.c_str());
                }
                
                // 解析 IP 和端口
                std::string ip = "127.0.0.1";
                uint16_t port = 8000;  // MediaMTX 預設 RTP 端口
                
                size_t colon_pos = endpoint.find(':');
                if (colon_pos != std::string::npos) {
                    ip = endpoint.substr(0, colon_pos);
                    port = (uint16_t)atoi(endpoint.substr(colon_pos + 1).c_str());
                } else {
                    ip = endpoint;
                }
                
                ALOGI("[RTP] Parsed: IP=%s, Port=%d", ip.c_str(), port);
                
                rtp_pusher_t* pusher = (rtp_pusher_t*)malloc(sizeof(rtp_pusher_t));
                if (pusher) {
                    int is_h265 = (pipe->m_output_type == po_mediamtx_h265) ? 1 : 0;
                    ALOGI("[RTP] Calling rtp_pusher_init...");
                    int ret = rtp_pusher_init(pusher, ip.c_str(), port, is_h265);
                    if (ret == 0) {
                        g_rtp_pushers[pipe->pipeid] = pusher;
                        ALOGI("[RTP] Pusher initialized successfully for pipeid %d, %s -> %s:%d", 
                              pipe->pipeid, is_h265 ? "H.265" : "H.264", ip.c_str(), port);
                    } else {
                        ALOGE("[RTP] Failed to initialize pusher for pipeid %d, ret=%d (errno=%d: %s)", 
                              pipe->pipeid, ret, errno, strerror(errno));
                        free(pusher);
                    }
                } else {
                    ALOGE("[RTP] Failed to allocate memory for pusher");
                }
            } else {
                ALOGE("[RTP] Invalid pipeid %d (must be 0-63)", pipe->pipeid);
            }
        } else {
            ALOGI("[RTP] Output type %d is not MediaMTX type, skipping RTP pusher init", pipe->m_output_type);
        }
    }

    break;
    case po_vo_sipeed_maix3_screen:
    {
        if (!pipeline_handle.b_maix3_init)
        {
            AX_MOD_INFO_T srcMod, dstMod;
            srcMod.enModId = AX_ID_IVPS;
            srcMod.s32GrpId = pipe->m_ivps_attr.n_ivps_grp;
            srcMod.s32ChnId = 0;
            dstMod.enModId = AX_ID_VO;
            dstMod.s32GrpId = 0;
            dstMod.s32ChnId = 0;
            AX_SYS_Link(&srcMod, &dstMod);

            int s32Ret = _create_vo((char *)"dsi0@480x854@60", pipe);
            if (AX_SUCCESS != s32Ret)
            {
                ALOGE("VoInit failed,s32Ret:0x%x\n", s32Ret);
                return -1;
            }
            pipeline_handle.b_maix3_init = true;
        }
        else
        {
            ALOGE("screen has been init");
        }
    }
    break;
    case po_vo_hdmi:
    {
        if (!pipeline_handle.b_hdmi_init)
        {
            int ret = _create_vo_hdmi(pipe);
            if (ret != 0)
            {
                ALOGE("_create_vo_hdmi failed %d", ret);
                return -1;
            }

            pipeline_handle.b_hdmi_init = true;
        }
        // AX_MOD_INFO_T srcMod, dstMod;
        // srcMod.enModId = AX_ID_IVPS;
        // srcMod.s32GrpId = pipe->m_ivps_attr.n_ivps_grp;
        // srcMod.s32ChnId = 0;
        // dstMod.enModId = AX_ID_VO;
        // dstMod.s32GrpId = 0;
        // dstMod.s32ChnId = pipe->m_vo_attr.hdmi.n_chn;
        // AX_SYS_Link(&srcMod, &dstMod);
    }
    break;
    default:
        break;
    }
    return 0;
}

int destory_pipeline(pipeline_t *pipe)
{
    if (!pipe)
    {
        ALOGE("invalid pipeline_t ptr");
        return -1;
    }

    if (!pipe->enable)
    {
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(g_pipeline_handle_mutex);
        if (!contain(pipeline_handle.pipeid_pipe, pipe->pipeid))
        {
            return -1;
        }
        pipe->n_loog_exit = 1;
        erase(pipeline_handle.pipeid_pipe, pipe->pipeid);
    }

    switch (pipe->m_output_type)
    {
    case po_venc_mjpg:
    case po_venc_h264:
    case po_venc_h265:
    case po_rtsp_h264:
    case po_rtsp_h265:
    case po_mediamtx_h264:
    case po_mediamtx_h265:
    {
        AX_MOD_INFO_T srcMod, dstMod;
        srcMod.enModId = AX_ID_IVPS;
        srcMod.s32GrpId = pipe->m_ivps_attr.n_ivps_grp;
        srcMod.s32ChnId = 0;

        dstMod.enModId = AX_ID_VENC;
        dstMod.s32GrpId = 0;
        dstMod.s32ChnId = pipe->m_venc_attr.n_venc_chn;
        AX_SYS_UnLink(&srcMod, &dstMod);

        if (contain(pipeline_handle.venc_chn, pipe->m_venc_attr.n_venc_chn))
        {
            // _destore_venc_grp 內會 pthread_join VENC 線程，確保不再使用 g_rtp_pushers 後再於下方釋放
            _destore_venc_grp(pipe);
            erase(pipeline_handle.venc_chn, pipe->m_venc_attr.n_venc_chn);
        }

        if (pipeline_handle.venc_chn.size() == 0)
        {
            ALOGN("AX_VENC_Deinit");
            AX_VENC_Deinit();
        }

        // 清理 RTP 推送器（必須在 _destore_venc_grp 之後，避免 VENC 線程仍在使用時釋放）
        if (pipe->m_output_type == po_mediamtx_h264 || pipe->m_output_type == po_mediamtx_h265)
        {
            if (pipe->pipeid >= 0 && pipe->pipeid < 64 && g_rtp_pushers[pipe->pipeid]) {
                rtp_pusher_deinit(g_rtp_pushers[pipe->pipeid]);
                free(g_rtp_pushers[pipe->pipeid]);
                g_rtp_pushers[pipe->pipeid] = NULL;
                ALOGI("[RTP] Pusher deinitialized for pipeid %d", pipe->pipeid);
            }
        }
    }

    break;
    case po_vo_sipeed_maix3_screen:
    {
        AX_MOD_INFO_T srcMod, dstMod;
        srcMod.enModId = AX_ID_IVPS;
        srcMod.s32GrpId = pipe->m_ivps_attr.n_ivps_grp;
        srcMod.s32ChnId = 0;
        dstMod.enModId = AX_ID_VO;
        dstMod.s32GrpId = 0;
        dstMod.s32ChnId = 0;
        AX_SYS_UnLink(&srcMod, &dstMod);
        if (pipeline_handle.b_maix3_init)
        {
            _destory_vo();
            pipeline_handle.b_maix3_init = false;
        }
    }
    break;
    case po_vo_hdmi:
    {
        if (pipeline_handle.b_hdmi_init)
        {
            _destory_vo_hdmi(pipe);
            pipeline_handle.b_hdmi_init = false;
        }
        // AX_MOD_INFO_T srcMod, dstMod;
        // srcMod.enModId = AX_ID_IVPS;
        // srcMod.s32GrpId = pipe->m_ivps_attr.n_ivps_grp;
        // srcMod.s32ChnId = 0;
        // dstMod.enModId = AX_ID_VO;
        // dstMod.s32GrpId = 0;
        // dstMod.s32ChnId = pipe->m_vo_attr.hdmi.n_chn;
        // AX_SYS_UnLink(&srcMod, &dstMod);
    }
    break;
    default:
        break;
    }

    switch (pipe->m_input_type)
    {
    case pi_user:
    {
        if (contain(pipeline_handle.ivps_grp, pipe->m_ivps_attr.n_ivps_grp))
        {
            _destore_ivps_grp(pipe);
            erase(pipeline_handle.ivps_grp, pipe->m_ivps_attr.n_ivps_grp);
        }

        if (pipeline_handle.ivps_grp.size() == 0)
        {
            ALOGN("AX_IVPS_Deinit");
            AX_IVPS_Deinit();
        }
    }
    case pi_vin:
    {
        AX_MOD_INFO_T srcMod, dstMod;
        srcMod.enModId = AX_ID_VIN;
        srcMod.s32GrpId = pipe->n_vin_pipe;
        srcMod.s32ChnId = pipe->n_vin_chn;

        dstMod.enModId = AX_ID_IVPS;
        dstMod.s32GrpId = pipe->m_ivps_attr.n_ivps_grp;
        dstMod.s32ChnId = 0;
        AX_SYS_UnLink(&srcMod, &dstMod);

        if (contain(pipeline_handle.ivps_grp, pipe->m_ivps_attr.n_ivps_grp))
        {
            _destore_ivps_grp(pipe);
            erase(pipeline_handle.ivps_grp, pipe->m_ivps_attr.n_ivps_grp);
        }

        if (pipeline_handle.ivps_grp.size() == 0)
        {
            ALOGN("AX_IVPS_Deinit");
            AX_IVPS_Deinit();
        }
    }
    break;
    case pi_vdec_h264:
    case pi_vdec_jpeg:
    {
        AX_MOD_INFO_T srcMod, dstMod;
        srcMod.enModId = AX_ID_VDEC;
        srcMod.s32GrpId = pipe->m_vdec_attr.n_vdec_grp;
        srcMod.s32ChnId = 0;

        dstMod.enModId = AX_ID_IVPS;
        dstMod.s32GrpId = pipe->m_ivps_attr.n_ivps_grp;
        dstMod.s32ChnId = 0;
        AX_SYS_UnLink(&srcMod, &dstMod);

        if (contain(pipeline_handle.vdec_grp, pipe->m_vdec_attr.n_vdec_grp))
        {
            if (pipe->m_input_type == pi_vdec_h264)
                _destore_vdec_grp(pipe);
            erase(pipeline_handle.vdec_grp, pipe->m_vdec_attr.n_vdec_grp);
        }
        if (pipeline_handle.vdec_grp.size() == 0)
        {
            ALOGN("AX_VDEC_DeInit");
            AX_VDEC_Deinit();
        }
        if (contain(pipeline_handle.ivps_grp, pipe->m_ivps_attr.n_ivps_grp))
        {
            _destore_ivps_grp(pipe);
            erase(pipeline_handle.ivps_grp, pipe->m_ivps_attr.n_ivps_grp);
        }

        if (pipeline_handle.ivps_grp.size() == 0)
        {
            ALOGN("AX_IVPS_Deinit");
            AX_IVPS_Deinit();
        }
    }
    break;
    default:
        break;
    }

    return 0;
}

int user_input(pipeline_t *pipe, int pipe_cnt, pipeline_buffer_t *buf)
{
    if (!pipe)
    {
        ALOGE("invalid pipeline_t ptr");
        return -1;
    }

    if (!buf)
    {
        ALOGE("invalid pipeline_buffer_t ptr");
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(g_pipeline_handle_mutex);
        if (!contain(pipeline_handle.pipeid_pipe, pipe->pipeid))
        {
            ALOGE("pipe-%d haven`t create", pipe->pipeid);
            return -1;
        }
    }

    switch (pipe->m_input_type)
    {
        // 不确定是否能用
    case pi_user:
    {
        AX_VIDEO_FRAME_INFO_T frameInfo = {0};

        AX_U32 uiPicSize = (buf->n_width * buf->n_height) * 3 / 2;
        AX_BLK blk_id = AX_POOL_GetBlock(0, uiPicSize, NULL);
        if (AX_INVALID_BLOCKID == blk_id)
        {
            printf("AX_POOL_GetBlock AX_POOL_GetBlockfailed! \n");
            return -1;
        }

        frameInfo.bEndOfStream = AX_FALSE;
        frameInfo.enModId = AX_ID_IVPS;
        frameInfo.stVFrame.u32BlkId[0] = blk_id;
        frameInfo.stVFrame.u32Width = buf->n_width;
        frameInfo.stVFrame.u32Height = buf->n_height;
        frameInfo.stVFrame.enImgFormat = AX_FORMAT_YUV420_SEMIPLANAR;
        frameInfo.stVFrame.enVscanFormat = AX_VSCAN_FORMAT_RASTER;
        frameInfo.stVFrame.stCompressInfo.enCompressMode = AX_COMPRESS_MODE_NONE;
        frameInfo.stVFrame.u64PhyAddr[0] = AX_POOL_Handle2PhysAddr(blk_id);
        frameInfo.stVFrame.u64VirAddr[0] = (AX_U64)AX_POOL_GetBlockVirAddr(blk_id);
        frameInfo.stVFrame.u32PicStride[0] = buf->n_width;
        frameInfo.stVFrame.u64PhyAddr[1] = frameInfo.stVFrame.u64PhyAddr[0] + frameInfo.stVFrame.u32PicStride[0] * frameInfo.stVFrame.u32Height;
        frameInfo.stVFrame.u64PhyAddr[2] = 0;
        frameInfo.stVFrame.u64VirAddr[1] = frameInfo.stVFrame.u64VirAddr[0] + frameInfo.stVFrame.u32PicStride[0] * frameInfo.stVFrame.u32Height;
        frameInfo.stVFrame.u64VirAddr[2] = 0;
        // frameInfo.u32PoolId = AX_POOL_Handle2PoolId(blk_id);

        memcpy((void *)frameInfo.stVFrame.u64VirAddr[0], buf->p_vir, uiPicSize);
        int ret;
        // [關鍵修復] 使用 thread_local 靜態變量緩存 vector，避免每幀都創建新的 vector
        static thread_local std::vector<int> tmp_;
        tmp_.clear();  // 清空但保留容量，避免重新分配

        for (int i = 0; i < pipe_cnt; i++)
        {
            if (!contain(tmp_, pipe[i].m_ivps_attr.n_ivps_grp))
            {
                ret = AX_IVPS_SendFrame(pipe[i].m_ivps_attr.n_ivps_grp, &frameInfo.stVFrame, 200);
                if (ret != 0)
                {
                    // Queue 滿了 (0x23) 或其他錯誤，丟棄這個 frame 以避免記憶體堆積
                    // 這裡不釋放 blk_id，因為會在後面統一釋放
                    if (ret == 0x23) {
                        static int drop_count[64] = {0};
                        int pid = (pipe[i].pipeid >= 0 && pipe[i].pipeid < 64) ? pipe[i].pipeid : 0;
                        if (++drop_count[pid] % 100 == 0) {
                            ALOGW("AX_IVPS_SendFrame queue full (pipeid %d), dropped %d frames", pipe[i].pipeid, drop_count[pid]);
                        }
                    } else {
                        ALOGE("AX_IVPS_SendFrame failed: 0x%x", ret);
                    }
                }
                tmp_.push_back(pipe[i].m_ivps_attr.n_ivps_grp);
            }
        }

        // AX_U32 PoolId = frameInfo.u32PoolId;

        AX_BLK BlkId = frameInfo.stVFrame.u32BlkId[0];
        ret = AX_POOL_ReleaseBlock(BlkId);
        if (ret != 0)
        {
            printf("AX_POOL_ReleaseBlock fail!Error Code:0x%X\n", ret);
            return -1;
        }
    }
    break;
    case pi_vdec_h264:
    {
        AX_VDEC_STREAM_T stream = {0};
        memset(&stream, 0, sizeof(AX_VDEC_STREAM_T));
        stream.u32StreamPackLen = buf->n_size;
        stream.pu8Addr = (unsigned char *)buf->p_vir;
        stream.u64PhyAddr = 0;
        // [殘影修復] RTSP/ demux 每次回調給的是一整幀（一個 access unit），應標記幀結束，否則 VDEC 會緩存等待更多數據導致幀邊界錯亂與殘影
        stream.bEndOfFrame = AX_TRUE;
        stream.bEndOfStream = AX_FALSE;
        // printf("0x%x\n", stream.pu8Addr);

        int ret;
        // [關鍵修復] 使用 thread_local 靜態變量緩存 vector，避免每幀都創建新的 vector
        static thread_local std::vector<int> tmp_;
        tmp_.clear();  // 清空但保留容量，避免重新分配

        for (int i = 0; i < pipe_cnt; i++)
        {
            if (!contain(tmp_, pipe[i].m_vdec_attr.n_vdec_grp))
            {
                // 超時 0：絕不阻塞 demux 回調，避免久播延遲累積至 3 秒以上；佇列滿則丟幀
                ret = AX_VDEC_SendStream(pipe[i].m_vdec_attr.n_vdec_grp, &stream, 0);
                if (ret != 0)
                {
                    /* 0x23 = AX_ERR_QUEUE_FULL，完整碼為 0x80080123；僅低字節比對以識別佇列滿 */
                    if ((ret & 0xFF) == 0x23) {
                        static int drop_count[64] = {0};
                        int pid = (pipe[i].pipeid >= 0 && pipe[i].pipeid < 64) ? pipe[i].pipeid : 0;
                        if (++drop_count[pid] % 100 == 0) {
                            ALOGW("AX_VDEC_SendStream queue full (pipeid %d), dropped %d frames", pipe[i].pipeid, drop_count[pid]);
                        }
                    } else {
                        ALOGE("AX_VDEC_SendStream 0x%x,data=%p len=%d", ret, (void*)stream.pu8Addr, stream.u32StreamPackLen);
                    }
                }
                tmp_.push_back(pipe[i].m_vdec_attr.n_vdec_grp);
            }
        }

        // int ret = AX_VDEC_SendStream(pipe->m_vdec_attr.n_vdec_grp, &stream, 200);
        // if (ret != 0)
        // {
        //     ALOGE("AX_VDEC_SendStream 0x%x", ret);
        //     return -1;
        // }
    }
    break;
    case pi_vdec_jpeg:
    {
        _create_jvdec_grp(pipe);
        AX_VDEC_STREAM_T stream = {0};
        int unsigned long long pts = 0;
        stream.u64PTS = pts++;
        stream.u32StreamPackLen = buf->n_size;
        stream.pu8Addr = (unsigned char *)buf->p_vir;
        stream.bEndOfFrame = buf->p_vir == NULL ? AX_TRUE : AX_FALSE;
        stream.bEndOfStream = buf->p_vir == NULL ? AX_TRUE : AX_FALSE;

        int ret;
        // [關鍵修復] 使用 thread_local 靜態變量緩存 vector，避免每幀都創建新的 vector
        static thread_local std::vector<int> tmp_;
        tmp_.clear();  // 清空但保留容量，避免重新分配

        for (int i = 0; i < pipe_cnt; i++)
        {
            if (!contain(tmp_, pipe[i].m_vdec_attr.n_vdec_grp))
            {
                ret = AX_VDEC_SendStream(pipe[i].m_vdec_attr.n_vdec_grp, &stream, -1);
                if (ret != 0)
                {
                    ALOGE("AX_VDEC_SendStream 0x%x,data=0x%x len=%d", ret, (unsigned long long int)stream.pu8Addr, stream.u32StreamPackLen);
                }
                tmp_.push_back(pipe[i].m_vdec_attr.n_vdec_grp);
            }
        }
        // int ret = AX_VDEC_SendStream(pipe->m_vdec_attr.n_vdec_grp, &stream, -1);
        // if (ret != 0)
        // {
        //     ALOGE("AX_VDEC_SendStream 0x%x", ret);
        //     return -1;
        // }
        _destore_jvdec_grp(pipe);
    }
    break;
    default:
        break;
    }

    return 0;
}