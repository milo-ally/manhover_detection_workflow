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
#include <map>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "letterbox_utils.hpp"

#define MODEL_CLASS_NUM 1
#define REG_MAX 16
#define DEFAULT_CONF_THRESH 0.35f
#define DEFAULT_NMS_THRESH 0.45f

static const char* CLASS_NAMES[] = {"face"};

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
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

class FaceDetectionModel : public IAIModel {
public:
    int Init(const char* model_path) override {
        printf("[FaceDetection] Loading model: %s\n", model_path);
        std::vector<char> model_buffer = read_model_file(model_path);
        if (model_buffer.empty()) return -1;

        int ret = AX_ENGINE_CreateHandle(&m_handle, model_buffer.data(), model_buffer.size());
        if (ret != 0) {
            printf("[Error] CreateHandle failed: 0x%x\n", ret);
            return -1;
        }

        AX_ENGINE_GetIOInfo(m_handle, &m_io_info);

        m_io_data.pInputs = new AX_ENGINE_IO_BUFFER_T[m_io_info->nInputSize];
        m_io_data.nInputSize = m_io_info->nInputSize;
        m_io_data.pOutputs = new AX_ENGINE_IO_BUFFER_T[m_io_info->nOutputSize];
        m_io_data.nOutputSize = m_io_info->nOutputSize;

        for (unsigned int i = 0; i < m_io_info->nInputSize; ++i) {
            memset(&m_io_data.pInputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));
            AX_U32 size = 640 * 640 * 3;
            AX_U64 phy = 0;
            AX_VOID* vir = NULL;
            AX_SYS_MemAlloc(&phy, &vir, size, 128, (const AX_S8*)"ax_face_input_rgb");
            m_io_data.pInputs[i].nSize = size;
            m_io_data.pInputs[i].phyAddr = phy;
            m_io_data.pInputs[i].pVirAddr = vir;
            m_io_data.pInputs[i].pStride = new AX_S32[4];
            m_io_data.pInputs[i].pStride[0] = 1920;
        }

        for (unsigned int i = 0; i < m_io_info->nOutputSize; ++i) {
            memset(&m_io_data.pOutputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));
            auto& info = m_io_info->pOutputs[i];
            m_io_data.pOutputs[i].nSize = info.nSize;
            AX_U64 phy = 0;
            AX_VOID* vir = NULL;
            AX_SYS_MemAlloc(&phy, &vir, info.nSize, 128, (const AX_S8*)"ax_face_output");
            m_io_data.pOutputs[i].phyAddr = phy;
            m_io_data.pOutputs[i].pVirAddr = vir;
            m_io_data.pOutputs[i].pStride = new AX_S32[4];
        }

        m_input_w = 640;
        m_input_h = 640;
        confThresh_ = DEFAULT_CONF_THRESH;
        nmsThresh_ = DEFAULT_NMS_THRESH;
        if (const char* c = std::getenv("FACE_DET_CONF")) {
            confThresh_ = static_cast<float>(atof(c));
        }
        if (const char* n = std::getenv("FACE_DET_NMS")) {
            nmsThresh_ = static_cast<float>(atof(n));
        }
        if (confThresh_ < 0.01f) confThresh_ = 0.01f;
        if (confThresh_ > 0.99f) confThresh_ = 0.99f;
        if (nmsThresh_ < 0.01f) nmsThresh_ = 0.01f;
        if (nmsThresh_ > 0.99f) nmsThresh_ = 0.99f;
        printf("[FaceDetection] thresholds: conf=%.3f, nms=%.3f\n", confThresh_, nmsThresh_);
        return 0;
    }

    void GetInputSize(int* w, int* h) override { *w = m_input_w; *h = m_input_h; }

    int Inference(const AX_VIDEO_FRAME_T* pFrame, AI_RESULT_T* pResult) override {
        if (!m_handle) return -1;

        cv::Mat nv12_mat(pFrame->u32Height * 3 / 2, pFrame->u32Width, CV_8UC1, (void*)pFrame->u64VirAddr[0], pFrame->u32PicStride[0]);
        cv::Mat rgb_mat;
        cv::cvtColor(nv12_mat, rgb_mat, cv::COLOR_YUV2RGB_NV12);

        cv::Mat input_rgb;
        ai_letterbox::LetterboxInfo lb_info = ai_letterbox::letterbox(rgb_mat, input_rgb, m_input_w, m_input_h);

        memcpy(m_io_data.pInputs[0].pVirAddr, input_rgb.data, m_input_w * m_input_h * 3);
        AX_SYS_MflushCache(m_io_data.pInputs[0].phyAddr, m_io_data.pInputs[0].pVirAddr, m_io_data.pInputs[0].nSize);

        int ret = AX_ENGINE_RunSync(m_handle, &m_io_data);
        if (ret != 0) return -1;

        std::vector<Object> proposals;
        generate_proposals(proposals);
        qsort_descent_inplace(proposals);
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nmsThresh_);

        const int src_w = (int)pFrame->u32Width;
        const int src_h = (int)pFrame->u32Height;

        pResult->nObjSize = std::min((int)picked.size(), MAX_DETECT_OBJ_NUM);
        float maxScore = 0.0f;
        for (int i = 0; i < pResult->nObjSize; i++) {
            Object prop = proposals[picked[i]];
            if (prop.prob > maxScore) maxScore = prop.prob;
            ai_letterbox::scale_bbox_to_original(prop.bbox.x, prop.bbox.y, prop.bbox.w, prop.bbox.h, lb_info);
            pResult->objects[i].x = prop.bbox.x / (float)src_w;
            pResult->objects[i].y = prop.bbox.y / (float)src_h;
            pResult->objects[i].w = prop.bbox.w / (float)src_w;
            pResult->objects[i].h = prop.bbox.h / (float)src_h;
            pResult->objects[i].score = prop.prob;
            pResult->objects[i].class_id = 0;
            snprintf(pResult->objects[i].label, 32, "%s", CLASS_NAMES[0]);
        }

        static unsigned int s_frame = 0;
        if ((++s_frame % 60) == 0) {
            printf("[FaceDetection] frame=%u proposals=%zu picked=%zu final=%u maxScore=%.3f conf=%.3f nms=%.3f\n",
                   s_frame, proposals.size(), picked.size(), pResult->nObjSize, maxScore, confThresh_, nmsThresh_);
        }
        return 0;
    }

    int Deinit() override {
        if (m_handle) {
            if (m_io_data.pInputs) {
                for (unsigned int i = 0; i < m_io_data.nInputSize; ++i) {
                    AX_SYS_MemFree(m_io_data.pInputs[i].phyAddr, m_io_data.pInputs[i].pVirAddr);
                    delete[] m_io_data.pInputs[i].pStride;
                }
                delete[] m_io_data.pInputs;
            }
            if (m_io_data.pOutputs) {
                for (unsigned int i = 0; i < m_io_data.nOutputSize; ++i) {
                    AX_SYS_MemFree(m_io_data.pOutputs[i].phyAddr, m_io_data.pOutputs[i].pVirAddr);
                    delete[] m_io_data.pOutputs[i].pStride;
                }
                delete[] m_io_data.pOutputs;
            }
            AX_ENGINE_DestroyHandle(m_handle);
            m_handle = nullptr;
        }
        return 0;
    }

private:
    AX_ENGINE_HANDLE m_handle = nullptr;
    AX_ENGINE_IO_INFO_T* m_io_info = nullptr;
    AX_ENGINE_IO_T m_io_data = {0};
    int m_input_w = 640;
    int m_input_h = 640;
    float confThresh_ = DEFAULT_CONF_THRESH;
    float nmsThresh_ = DEFAULT_NMS_THRESH;

    struct Rect { float x, y, w, h; };
    struct Object { Rect bbox; int label; float prob; };
    struct HeadData {
        float* box;
        float* cls;
        bool is_combined;
        bool is_nchw;
        int H;
        int W;
    };

    void generate_proposals(std::vector<Object>& proposals) {
        std::map<int, HeadData> heads;
        for (unsigned int i = 0; i < m_io_info->nOutputSize; ++i) {
            auto& output = m_io_info->pOutputs[i];
            float* data = (float*)m_io_data.pOutputs[i].pVirAddr;
            if (!data) continue;

            int C_nhwc = output.pShape[3];
            int H_nhwc = output.pShape[1];
            int W_nhwc = output.pShape[2];
            int C_nchw = output.pShape[1];
            int H_nchw = output.pShape[2];
            int W_nchw = output.pShape[3];

            bool is_nhwc = (C_nhwc == 4 * REG_MAX + MODEL_CLASS_NUM) ||
                           (C_nhwc == 4 * REG_MAX) ||
                           (C_nhwc == MODEL_CLASS_NUM);
            bool is_nchw = (C_nchw == 4 * REG_MAX + MODEL_CLASS_NUM) ||
                           (C_nchw == 4 * REG_MAX) ||
                           (C_nchw == MODEL_CLASS_NUM);
            if (!is_nhwc && !is_nchw) continue;

            bool use_nchw = (!is_nhwc && is_nchw);
            int H = use_nchw ? H_nchw : H_nhwc;
            int W = use_nchw ? W_nchw : W_nhwc;
            int C = use_nchw ? C_nchw : C_nhwc;
            if (H <= 0 || W <= 0) continue;

            if (heads.find(H) == heads.end()) heads[H] = {nullptr, nullptr, false, use_nchw, H, W};
            heads[H].is_nchw = use_nchw;
            heads[H].W = W;

            if (C == 4 * REG_MAX + MODEL_CLASS_NUM) {
                heads[H].box = data;
                heads[H].cls = data;
                heads[H].is_combined = true;
            } else if (C == 4 * REG_MAX) {
                heads[H].box = data;
            } else if (C == MODEL_CLASS_NUM) {
                heads[H].cls = data;
            }
        }

        for (auto& pair : heads) {
            int H = pair.first;
            HeadData& head = pair.second;
            if (!head.box || !head.cls) continue;

            int stride = 640 / H;
            int W = head.W;
            int C_combined = 4 * REG_MAX + MODEL_CLASS_NUM;
            int C_box = 4 * REG_MAX;

            auto at_nhwc = [&](const float* base, int h, int w, int c, int C) -> float {
                return base[(h * W + w) * C + c];
            };
            auto at_nchw = [&](const float* base, int h, int w, int c, int C) -> float {
                (void)C;
                return base[(c * H + h) * W + w];
            };

            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    float cls_raw = 0.0f;
                    if (head.is_combined) {
                        cls_raw = head.is_nchw ? at_nchw(head.cls, h, w, C_box, C_combined)
                                               : at_nhwc(head.cls, h, w, C_box, C_combined);
                    } else {
                        cls_raw = head.is_nchw ? at_nchw(head.cls, h, w, 0, MODEL_CLASS_NUM)
                                               : at_nhwc(head.cls, h, w, 0, MODEL_CLASS_NUM);
                    }

                    float score = sigmoid(cls_raw);
                    if (score < confThresh_) continue;

                    float pred_ltrb[4];
                    for (int k = 0; k < 4; k++) {
                        float exp_sum = 0.0f;
                        float weighted_sum = 0.0f;
                        for (int r = 0; r < REG_MAX; r++) {
                            int c = k * REG_MAX + r;
                            float raw = 0.0f;
                            if (head.is_combined) {
                                raw = head.is_nchw ? at_nchw(head.box, h, w, c, C_combined)
                                                   : at_nhwc(head.box, h, w, c, C_combined);
                            } else {
                                raw = head.is_nchw ? at_nchw(head.box, h, w, c, C_box)
                                                   : at_nhwc(head.box, h, w, c, C_box);
                            }
                            float e = expf(raw);
                            exp_sum += e;
                            weighted_sum += e * r;
                        }
                        pred_ltrb[k] = weighted_sum / exp_sum;
                    }

                    float cx = (w + 0.5f) * stride;
                    float cy = (h + 0.5f) * stride;
                    float x1 = cx - pred_ltrb[0] * stride;
                    float y1 = cy - pred_ltrb[1] * stride;
                    float x2 = cx + pred_ltrb[2] * stride;
                    float y2 = cy + pred_ltrb[3] * stride;

                    Object obj;
                    obj.bbox = {x1, y1, x2 - x1, y2 - y1};
                    obj.label = 0;
                    obj.prob = score;
                    proposals.push_back(obj);
                }
            }
        }
    }

    void qsort_descent_inplace(std::vector<Object>& objects) {
        std::sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
    }

    void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold) {
        picked.clear();
        const int n = objects.size();
        std::vector<float> areas(n);
        for (int i = 0; i < n; i++) areas[i] = objects[i].bbox.w * objects[i].bbox.h;
        for (int i = 0; i < n; i++) {
            const Object& a = objects[i];
            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++) {
                const Object& b = objects[picked[j]];
                float inter_x1 = std::max(a.bbox.x, b.bbox.x);
                float inter_y1 = std::max(a.bbox.y, b.bbox.y);
                float inter_x2 = std::min(a.bbox.x + a.bbox.w, b.bbox.x + b.bbox.w);
                float inter_y2 = std::min(a.bbox.y + a.bbox.h, b.bbox.y + b.bbox.h);
                float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                if (union_area > 0.f && inter_area / union_area > nms_threshold) { keep = 0; break; }
            }
            if (keep) picked.push_back(i);
        }
    }
};

extern "C" {
    IAIModel* CreateAIModel() { return new FaceDetectionModel(); }
    void DestroyAIModel(IAIModel* p) { delete p; }
}
