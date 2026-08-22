#include "ai_interface.h"
#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include "c_api.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdlib>

#include "opencv2/opencv.hpp"
#include "letterbox_utils.hpp"

/* ================== config ================== */
#define CLASS_NUM 2
#define REG_MAX 16
#define CONF_THRESH 0.45f
#define NMS_THRESH 0.45f

static float read_env_float(const char* key, float def) {
    const char* v = std::getenv(key);
    if (!v || !*v) return def;
    return static_cast<float>(atof(v));
}

static const char* CLASS_NAMES[] = {
    "calling",
    "smoking"
};

/* ================== utils ================== */
static inline float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

static std::vector<char> read_model_file(const char* filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        printf("[Error] Cannot open model file: %s\n", filename);
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        printf("[Error] Failed to read model file: %s\n", filename);
        return {};
    }
    return buffer;
}

/* ================== model ================== */
class BehaviorModel : public IAIModel {
public:
    int Init(const char* model_path) override {
        printf("[Behavior] Loading model: %s\n", model_path);
        confThresh_ = read_env_float("BEHAVIOR_CONF_THRESH", read_env_float("MODEL_CONF_THRESH", CONF_THRESH));
        nmsThresh_ = read_env_float("BEHAVIOR_NMS_THRESH", read_env_float("MODEL_NMS_THRESH", NMS_THRESH));
        printf("[Behavior] thresholds: conf=%.3f nms=%.3f\n", confThresh_, nmsThresh_);

        auto model_buffer = read_model_file(model_path);
        if (model_buffer.empty()) return -1;

        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;

        int ret = AX_ENGINE_CreateHandle(
            &m_handle,
            model_buffer.data(),
            model_buffer.size()
        );
        if (ret != 0) {
            printf("[Error] CreateHandle failed: 0x%x\n", ret);
            return -1;
        }

        AX_ENGINE_GetIOInfo(m_handle, &m_io_info);

        m_io_data.pInputs  = new AX_ENGINE_IO_BUFFER_T[m_io_info->nInputSize];
        m_io_data.nInputSize = m_io_info->nInputSize;
        m_io_data.pOutputs = new AX_ENGINE_IO_BUFFER_T[m_io_info->nOutputSize];
        m_io_data.nOutputSize = m_io_info->nOutputSize;

        /* -------- input -------- */
        for (unsigned int i = 0; i < m_io_info->nInputSize; ++i) {
            memset(&m_io_data.pInputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));

            AX_U32 size = 640 * 640 * 3;
            AX_U64 phy = 0;
            AX_VOID* vir = NULL;

            AX_SYS_MemAlloc(&phy, &vir, size, 128, (const AX_S8*)"ax_input_bgr");

            m_io_data.pInputs[i].nSize     = size;
            m_io_data.pInputs[i].phyAddr  = phy;
            m_io_data.pInputs[i].pVirAddr = vir;

            m_io_data.pInputs[i].pStride = new AX_S32[4];
            m_io_data.pInputs[i].pStride[0] = 640 * 3;
        }

        /* -------- output -------- */
        for (unsigned int i = 0; i < m_io_info->nOutputSize; ++i) {
            memset(&m_io_data.pOutputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));

            auto& info = m_io_info->pOutputs[i];

            AX_U64 phy = 0;
            AX_VOID* vir = NULL;
            AX_SYS_MemAlloc(&phy, &vir, info.nSize, 128, (const AX_S8*)"ax_output");

            m_io_data.pOutputs[i].nSize     = info.nSize;
            m_io_data.pOutputs[i].phyAddr  = phy;
            m_io_data.pOutputs[i].pVirAddr = vir;
            m_io_data.pOutputs[i].pStride  = new AX_S32[4];
        }

        m_input_w = 640;
        m_input_h = 640;
        return 0;
    }

    void GetInputSize(int* w, int* h) override {
        *w = m_input_w;
        *h = m_input_h;
    }

    int Inference(const AX_VIDEO_FRAME_T* pFrame, AI_RESULT_T* pResult) override {
        if (!m_handle) return -1;

        cv::Mat nv12(
            pFrame->u32Height * 3 / 2,
            pFrame->u32Width,
            CV_8UC1,
            (void*)pFrame->u64VirAddr[0],
            pFrame->u32PicStride[0]
        );

        cv::Mat bgr;
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
        cv::Mat input_bgr;
        ai_letterbox::LetterboxInfo lb_info = ai_letterbox::letterbox(bgr, input_bgr, m_input_w, m_input_h);

        memcpy(m_io_data.pInputs[0].pVirAddr, input_bgr.data, m_input_w * m_input_h * 3);
        AX_SYS_MflushCache(
            m_io_data.pInputs[0].phyAddr,
            m_io_data.pInputs[0].pVirAddr,
            m_io_data.pInputs[0].nSize
        );

        if (AX_ENGINE_RunSync(m_handle, &m_io_data) != 0)
            return -1;

        std::vector<Object> proposals;
        generate_proposals(proposals);

        std::sort(proposals.begin(), proposals.end(),
                  [](const Object& a, const Object& b) { return a.prob > b.prob; });

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked);

        const int src_w = (int)pFrame->u32Width;
        const int src_h = (int)pFrame->u32Height;

        pResult->nObjSize = std::min((int)picked.size(), MAX_DETECT_OBJ_NUM);
        for (int i = 0; i < pResult->nObjSize; i++) {
            Object obj = proposals[picked[i]];
            ai_letterbox::scale_bbox_to_original(obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h, lb_info);
            pResult->objects[i].x = obj.bbox.x / (float)src_w;
            pResult->objects[i].y = obj.bbox.y / (float)src_h;
            pResult->objects[i].w = obj.bbox.w / (float)src_w;
            pResult->objects[i].h = obj.bbox.h / (float)src_h;
            pResult->objects[i].score = obj.prob;
            pResult->objects[i].class_id = obj.label;
            snprintf(pResult->objects[i].label, 32, "%s", CLASS_NAMES[obj.label]);
        }
        return 0;
    }

    int Deinit() override {
        if (!m_handle) return 0;

        for (unsigned int i = 0; i < m_io_data.nInputSize; ++i) {
            AX_SYS_MemFree(m_io_data.pInputs[i].phyAddr,
                           m_io_data.pInputs[i].pVirAddr);
            delete[] m_io_data.pInputs[i].pStride;
        }
        for (unsigned int i = 0; i < m_io_data.nOutputSize; ++i) {
            AX_SYS_MemFree(m_io_data.pOutputs[i].phyAddr,
                           m_io_data.pOutputs[i].pVirAddr);
            delete[] m_io_data.pOutputs[i].pStride;
        }

        delete[] m_io_data.pInputs;
        delete[] m_io_data.pOutputs;

        AX_ENGINE_DestroyHandle(m_handle);
        m_handle = nullptr;
        return 0;
    }

private:
    struct Rect { float x, y, w, h; };
    struct Object { Rect bbox; int label; float prob; };

    AX_ENGINE_HANDLE m_handle = nullptr;
    AX_ENGINE_IO_INFO_T* m_io_info = nullptr;
    AX_ENGINE_IO_T m_io_data = {0};
    int m_input_w = 640;
    int m_input_h = 640;
    float confThresh_ = CONF_THRESH;
    float nmsThresh_ = NMS_THRESH;

    void generate_proposals(std::vector<Object>& proposals) {
        for (int head = 0; head < 3; head++) {
            float* cls = (float*)m_io_data.pOutputs[head].pVirAddr;
            float* box = (float*)m_io_data.pOutputs[head + 3].pVirAddr;

            int H = m_io_info->pOutputs[head].pShape[1];
            int W = m_io_info->pOutputs[head].pShape[2];
            int stride = 640 / H;

            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    float* cls_ptr = cls + (y * W + x) * CLASS_NUM;
                    float* box_ptr = box + (y * W + x) * 4 * REG_MAX;

                    int label = 0;
                    float max_logit = cls_ptr[0];
                    for (int c = 1; c < CLASS_NUM; c++) {
                        if (cls_ptr[c] > max_logit) {
                            max_logit = cls_ptr[c];
                            label = c;
                        }
                    }
                    float score = sigmoid(max_logit);
                    if (score < confThresh_) continue;

                    float d[4];
                    for (int k = 0; k < 4; k++) {
                        float sum = 0, acc = 0;
                        for (int r = 0; r < REG_MAX; r++) {
                            float e = expf(box_ptr[k * REG_MAX + r]);
                            sum += e;
                            acc += e * r;
                        }
                        d[k] = acc / sum * stride;
                    }

                    float cx = (x + 0.5f) * stride;
                    float cy = (y + 0.5f) * stride;

                    proposals.push_back({
                        {cx - d[0], cy - d[1], d[0] + d[2], d[1] + d[3]},
                        label,
                        score
                    });
                }
            }
        }
    }

    void nms_sorted_bboxes(const std::vector<Object>& objs,
                           std::vector<int>& picked) {
        picked.clear();
        for (int i = 0; i < objs.size(); i++) {
            bool keep = true;
            for (int j : picked) {
                float inter_x1 = std::max(objs[i].bbox.x, objs[j].bbox.x);
                float inter_y1 = std::max(objs[i].bbox.y, objs[j].bbox.y);
                float inter_x2 = std::min(objs[i].bbox.x + objs[i].bbox.w,
                                          objs[j].bbox.x + objs[j].bbox.w);
                float inter_y2 = std::min(objs[i].bbox.y + objs[i].bbox.h,
                                          objs[j].bbox.y + objs[j].bbox.h);
                float inter = std::max(0.f, inter_x2 - inter_x1) *
                              std::max(0.f, inter_y2 - inter_y1);
                float uni = objs[i].bbox.w * objs[i].bbox.h +
                            objs[j].bbox.w * objs[j].bbox.h - inter;
                if (inter / uni > nmsThresh_) { keep = false; break; }
            }
            if (keep) picked.push_back(i);
        }
    }
};

/* ================== factory ================== */
extern "C" {
IAIModel* CreateAIModel() { return new BehaviorModel(); }
void DestroyAIModel(IAIModel* p) { delete p; }
}
