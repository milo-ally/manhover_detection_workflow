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

#define CLASS_NUM 2

#define REG_MAX 16
#define CONF_THRESH 0.25f
#define NMS_THRESH 0.45f

static float read_env_float(const char* key, float def) {
    const char* v = std::getenv(key);
    if (!v || !*v) return def;
    return static_cast<float>(atof(v));
}

static const char* CLASS_NAMES[] = {
    "smoke",      // Class 0
    "fire"        // Class 1
};

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

class Yolo11SmokeFireModel : public IAIModel {
public:
    int Init(const char* model_path) override {
        printf("[SmokeFire] Loading model: %s\n", model_path);
        confThresh_ = read_env_float("SMOKE_FIRE_CONF_THRESH", read_env_float("MODEL_CONF_THRESH", CONF_THRESH));
        nmsThresh_ = read_env_float("SMOKE_FIRE_NMS_THRESH", read_env_float("MODEL_NMS_THRESH", NMS_THRESH));
        printf("[SmokeFire] thresholds: conf=%.3f nms=%.3f\n", confThresh_, nmsThresh_);
        std::vector<char> model_buffer = read_model_file(model_path);
        if (model_buffer.empty()) return -1;

        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE; 

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
            AX_SYS_MemAlloc(&phy, &vir, size, 128, (const AX_S8*)"ax_input_rgb");
            
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
            AX_SYS_MemAlloc(&phy, &vir, info.nSize, 128, (const AX_S8*)"ax_output");
            
            m_io_data.pOutputs[i].phyAddr = phy;
            m_io_data.pOutputs[i].pVirAddr = vir;
            m_io_data.pOutputs[i].pStride = new AX_S32[4];
        }

        if (m_io_info->nInputSize > 0) {
             m_input_w = 640;
             m_input_h = 640;
        }
        return 0;
    }

    void GetInputSize(int* w, int* h) override { *w = m_input_w; *h = m_input_h; }

    int Inference(const AX_VIDEO_FRAME_T* pFrame, AI_RESULT_T* pResult) override {
        if (!m_handle) return -1;

        // NV12 -> RGB 轉碼
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
        for (int i = 0; i < pResult->nObjSize; i++) {
            Object prop = proposals[picked[i]];
            ai_letterbox::scale_bbox_to_original(prop.bbox.x, prop.bbox.y, prop.bbox.w, prop.bbox.h, lb_info);
            pResult->objects[i].x = prop.bbox.x / (float)src_w;
            pResult->objects[i].y = prop.bbox.y / (float)src_h;
            pResult->objects[i].w = prop.bbox.w / (float)src_w;
            pResult->objects[i].h = prop.bbox.h / (float)src_h;
            pResult->objects[i].score = prop.prob;
            pResult->objects[i].class_id = prop.label;
            
            // 根據 label ID 填入正確名稱
            if (prop.label >= 0 && prop.label < CLASS_NUM) {
                snprintf(pResult->objects[i].label, 32, "%s", CLASS_NAMES[prop.label]);
            } else {
                snprintf(pResult->objects[i].label, 32, "unknown");
            }
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
    float confThresh_ = CONF_THRESH;
    float nmsThresh_ = NMS_THRESH;

    struct Rect { float x, y, w, h; };
    struct Object { Rect bbox; int label; float prob; };
    struct HeadData { float* box; float* cls; bool is_combined; };

    void generate_proposals(std::vector<Object>& proposals) {
        std::map<int, HeadData> heads;

        // 1. Group outputs by Grid Size (H)
        for (unsigned int i = 0; i < m_io_info->nOutputSize; ++i) {
            auto& output = m_io_info->pOutputs[i];
            float* data = (float*)m_io_data.pOutputs[i].pVirAddr;
            if (!data) continue;

            int H = output.pShape[1];
            int C = output.pShape[3];

            if (heads.find(H) == heads.end()) {
                heads[H] = {nullptr, nullptr, false};
            }

            if (C == 4 * REG_MAX + CLASS_NUM) { // Combined Head (66)
                heads[H].box = data;
                heads[H].cls = data; // Same pointer, needs offset
                heads[H].is_combined = true;
            } else if (C == 4 * REG_MAX) { // Box Head (64)
                heads[H].box = data;
            } else if (C == CLASS_NUM) { // Cls Head (2)
                heads[H].cls = data;
            }
        }

        // 2. Process each head
        for (auto& pair : heads) {
            int H = pair.first;
            HeadData& head = pair.second;
            
            if (!head.box || !head.cls) continue; // Incomplete head

            int stride = 640 / H;
            int W = H; // Assuming square
            int C_combined = 4 * REG_MAX + CLASS_NUM;
            int C_box = 4 * REG_MAX;
            int C_cls = CLASS_NUM;

            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    float* box_ptr = nullptr;
                    float* cls_ptr = nullptr;

                    if (head.is_combined) {
                        float* ptr = head.box + (h * W + w) * C_combined;
                        box_ptr = ptr;
                        cls_ptr = ptr + C_box;
                    } else {
                        box_ptr = head.box + (h * W + w) * C_box;
                        cls_ptr = head.cls + (h * W + w) * C_cls;
                    }

                    // Find max class
                    int max_id = 0;
                    float max_prob = -1000.0f;
                    for (int c = 0; c < CLASS_NUM; c++) {
                        if (cls_ptr[c] > max_prob) {
                            max_prob = cls_ptr[c];
                            max_id = c;
                        }
                    }

                    float score = sigmoid(max_prob);
                    if (score < confThresh_) continue;

                    // Decode box (DFL)
                    float pred_ltrb[4];
                    for (int k = 0; k < 4; k++) {
                        float exp_sum = 0.0f; 
                        float weighted_sum = 0.0f;
                        const float* dfl = box_ptr + k * REG_MAX;
                        for (int r = 0; r < REG_MAX; r++) {
                            float e = expf(dfl[r]);
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
                    obj.label = max_id;
                    obj.prob = score;
                    proposals.push_back(obj);
                }
            }
        }
    }

    void qsort_descent_inplace(std::vector<Object>& faceobjects) {
        std::sort(faceobjects.begin(), faceobjects.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
    }

    void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold) {
        picked.clear();
        const int n = faceobjects.size();
        std::vector<float> areas(n);
        for (int i = 0; i < n; i++) areas[i] = faceobjects[i].bbox.w * faceobjects[i].bbox.h;
        for (int i = 0; i < n; i++) {
            const Object& a = faceobjects[i];
            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++) {
                const Object& b = faceobjects[picked[j]];
                float inter_x1 = std::max(a.bbox.x, b.bbox.x);
                float inter_y1 = std::max(a.bbox.y, b.bbox.y);
                float inter_x2 = std::min(a.bbox.x + a.bbox.w, b.bbox.x + b.bbox.w);
                float inter_y2 = std::min(a.bbox.y + a.bbox.h, b.bbox.y + b.bbox.h);
                float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                if (inter_area / union_area > nms_threshold) { keep = 0; break; }
            }
            if (keep) picked.push_back(i);
        }
    }
};

extern "C" {
    IAIModel* CreateAIModel() { return new Yolo11SmokeFireModel(); }
    void DestroyAIModel(IAIModel* p) { delete p; }
}
