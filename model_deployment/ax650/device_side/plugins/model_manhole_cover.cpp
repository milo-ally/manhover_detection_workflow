#include "ai_interface.h"
#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include "letterbox_utils.hpp"

#include "opencv2/opencv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define MANHOLE_CLASS_NUM 5
#define MANHOLE_INPUT_SIZE 640
#define MANHOLE_ANCHOR_NUM 8400
#define CONF_THRESH 0.25f
#define NMS_THRESH 0.45f

// Manhole-cover five-class model.
// This model is different from the existing YOLO DFL-head plugins:
// output0 is [1, 9, 8400] = cx/cy/w/h + 5 class scores.
static const char* CLASS_NAMES[MANHOLE_CLASS_NUM] = {
    "good", "broke", "lose", "uncovered", "circle"
};

static float read_env_float(const char* key, float def) {
    const char* v = std::getenv(key);
    if (!v || !*v) return def;
    return static_cast<float>(atof(v));
}

static std::vector<char> read_model_file(const char* filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        printf("[ManholeCover][Error] Cannot open model file: %s\n", filename);
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        printf("[ManholeCover][Error] Failed to read model file: %s\n", filename);
        return {};
    }
    return buffer;
}

class ManholeCoverModel : public IAIModel {
public:
    int Init(const char* model_path) override {
        printf("[ManholeCover] Loading model: %s\n", model_path);
        confThresh_ = read_env_float("MANHOLE_CONF_THRESH", read_env_float("MODEL_CONF_THRESH", CONF_THRESH));
        nmsThresh_ = read_env_float("MANHOLE_NMS_THRESH", read_env_float("MODEL_NMS_THRESH", NMS_THRESH));
        printf("[ManholeCover] thresholds: conf=%.3f nms=%.3f\n", confThresh_, nmsThresh_);

        std::vector<char> model_buffer = read_model_file(model_path);
        if (model_buffer.empty()) return -1;

        int ret = AX_ENGINE_CreateHandle(&m_handle, model_buffer.data(), model_buffer.size());
        if (ret != 0) {
            printf("[ManholeCover][Error] CreateHandle failed: 0x%x\n", ret);
            return -1;
        }

        AX_ENGINE_GetIOInfo(m_handle, &m_io_info);
        if (!m_io_info || m_io_info->nInputSize == 0 || m_io_info->nOutputSize == 0) {
            printf("[ManholeCover][Error] Invalid model IO info\n");
            return -1;
        }

        m_io_data.pInputs = new AX_ENGINE_IO_BUFFER_T[m_io_info->nInputSize];
        m_io_data.nInputSize = m_io_info->nInputSize;
        m_io_data.pOutputs = new AX_ENGINE_IO_BUFFER_T[m_io_info->nOutputSize];
        m_io_data.nOutputSize = m_io_info->nOutputSize;

        for (unsigned int i = 0; i < m_io_info->nInputSize; ++i) {
            memset(&m_io_data.pInputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));
            AX_U32 size = MANHOLE_INPUT_SIZE * MANHOLE_INPUT_SIZE * 3;
            AX_U64 phy = 0;
            AX_VOID* vir = nullptr;
            if (AX_SYS_MemAlloc(&phy, &vir, size, 128, (const AX_S8*)"manhole_input_rgb") != 0) {
                printf("[ManholeCover][Error] AX_SYS_MemAlloc input failed\n");
                return -1;
            }
            m_io_data.pInputs[i].nSize = size;
            m_io_data.pInputs[i].phyAddr = phy;
            m_io_data.pInputs[i].pVirAddr = vir;
            m_io_data.pInputs[i].pStride = new AX_S32[4];
            memset(m_io_data.pInputs[i].pStride, 0, sizeof(AX_S32) * 4);
            m_io_data.pInputs[i].pStride[0] = MANHOLE_INPUT_SIZE * 3;
        }

        for (unsigned int i = 0; i < m_io_info->nOutputSize; ++i) {
            memset(&m_io_data.pOutputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));
            auto& info = m_io_info->pOutputs[i];
            AX_U64 phy = 0;
            AX_VOID* vir = nullptr;
            if (AX_SYS_MemAlloc(&phy, &vir, info.nSize, 128, (const AX_S8*)"manhole_output") != 0) {
                printf("[ManholeCover][Error] AX_SYS_MemAlloc output failed\n");
                return -1;
            }
            m_io_data.pOutputs[i].nSize = info.nSize;
            m_io_data.pOutputs[i].phyAddr = phy;
            m_io_data.pOutputs[i].pVirAddr = vir;
            m_io_data.pOutputs[i].pStride = new AX_S32[4];
            memset(m_io_data.pOutputs[i].pStride, 0, sizeof(AX_S32) * 4);
        }

        return 0;
    }

    void GetInputSize(int* w, int* h) override {
        if (w) *w = MANHOLE_INPUT_SIZE;
        if (h) *h = MANHOLE_INPUT_SIZE;
    }

    int Inference(const AX_VIDEO_FRAME_T* pFrame, AI_RESULT_T* pResult) override {
        fprintf(stderr, "/// ============================ DEBUG ================================\n");
        fprintf(stderr, "[ManholeCover] Inference ENTER: handle=%p frame=%p result=%p\n",
                static_cast<void*>(m_handle), static_cast<const void*>(pFrame),
                static_cast<void*>(pResult));
        fflush(stderr);

        if (!m_handle || !pFrame || !pResult) {
            fprintf(stderr, "[ManholeCover] Inference ABORT: invalid handle/frame/result\n");
            fprintf(stderr, "/// ============================ DEBUG ================================\n");
            fflush(stderr);
            return -1;
        }
        memset(pResult, 0, sizeof(AI_RESULT_T));

        fprintf(stderr, "[ManholeCover] input frame: width=%u height=%u stride=%u size=%u\n",
                static_cast<unsigned int>(pFrame->u32Width),
                static_cast<unsigned int>(pFrame->u32Height),
                static_cast<unsigned int>(pFrame->u32PicStride[0]),
                static_cast<unsigned int>(pFrame->u32FrameSize));
        fflush(stderr);

        // Board-side AI frames enter plugins as NV12; conversion and letterbox must
        // match model_convert/pulsar2_sim and model_val exactly.
        cv::Mat nv12_mat(pFrame->u32Height * 3 / 2, pFrame->u32Width, CV_8UC1,
                         (void*)pFrame->u64VirAddr[0], pFrame->u32PicStride[0]);
        cv::Mat rgb_mat;
        cv::cvtColor(nv12_mat, rgb_mat, cv::COLOR_YUV2RGB_NV12);

        cv::Mat input_rgb;
        ai_letterbox::LetterboxInfo lb_info =
            ai_letterbox::letterbox(rgb_mat, input_rgb, MANHOLE_INPUT_SIZE, MANHOLE_INPUT_SIZE);

        memcpy(m_io_data.pInputs[0].pVirAddr, input_rgb.data, MANHOLE_INPUT_SIZE * MANHOLE_INPUT_SIZE * 3);
        AX_SYS_MflushCache(m_io_data.pInputs[0].phyAddr,
                           m_io_data.pInputs[0].pVirAddr,
                           m_io_data.pInputs[0].nSize);

        int ret = AX_ENGINE_RunSync(m_handle, &m_io_data);
        if (ret != 0) {
            fprintf(stderr, "[ManholeCover][Error] AX_ENGINE_RunSync failed: 0x%x\n", ret);
            fprintf(stderr, "/// ============================ DEBUG ================================\n");
            fflush(stderr);
            return -1;
        }

        std::vector<Object> proposals;
        generate_proposals(proposals);
        sort_descent(proposals);

        std::vector<int> picked;
        nms_by_class(proposals, picked, nmsThresh_);

        const int src_w = static_cast<int>(pFrame->u32Width);
        const int src_h = static_cast<int>(pFrame->u32Height);
        pResult->nObjSize = std::min(static_cast<int>(picked.size()), MAX_DETECT_OBJ_NUM);
        for (AX_U32 i = 0; i < pResult->nObjSize; ++i) {
            Object obj = proposals[picked[i]];
            ai_letterbox::scale_bbox_to_original(obj.x, obj.y, obj.w, obj.h, lb_info);
            obj.x = std::max(0.0f, std::min(obj.x, static_cast<float>(src_w)));
            obj.y = std::max(0.0f, std::min(obj.y, static_cast<float>(src_h)));
            obj.w = std::max(0.0f, std::min(obj.w, static_cast<float>(src_w) - obj.x));
            obj.h = std::max(0.0f, std::min(obj.h, static_cast<float>(src_h) - obj.y));

            AI_OBJ_T& out = pResult->objects[i];
            out.x = obj.x / static_cast<float>(src_w);
            out.y = obj.y / static_cast<float>(src_h);
            out.w = obj.w / static_cast<float>(src_w);
            out.h = obj.h / static_cast<float>(src_h);
            out.class_id = obj.label;
            out.score = obj.score;
            snprintf(out.label, sizeof(out.label), "%s", CLASS_NAMES[obj.label]);
        }

        fprintf(stderr, "[ManholeCover] AI_RESULT_T: nObjSize=%u\n",
                static_cast<unsigned int>(pResult->nObjSize));
        for (AX_U32 i = 0; i < pResult->nObjSize; ++i) {
            const AI_OBJ_T& obj = pResult->objects[i];
            fprintf(stderr, "[ManholeCover] object[%u]: class_id=%d label=%s score=%.6f "
                    "x=%.6f y=%.6f w=%.6f h=%.6f\n",
                    static_cast<unsigned int>(i), static_cast<int>(obj.class_id),
                    obj.label, obj.score, obj.x, obj.y, obj.w, obj.h);
        }
        fprintf(stderr, "/// ============================ DEBUG ================================\n");
        fflush(stderr);

        return 0;
    }

    int Deinit() override {
        if (m_io_data.pInputs) {
            for (unsigned int i = 0; i < m_io_data.nInputSize; ++i) {
                if (m_io_data.pInputs[i].pVirAddr) {
                    AX_SYS_MemFree(m_io_data.pInputs[i].phyAddr, m_io_data.pInputs[i].pVirAddr);
                }
                delete[] m_io_data.pInputs[i].pStride;
            }
            delete[] m_io_data.pInputs;
            m_io_data.pInputs = nullptr;
        }
        if (m_io_data.pOutputs) {
            for (unsigned int i = 0; i < m_io_data.nOutputSize; ++i) {
                if (m_io_data.pOutputs[i].pVirAddr) {
                    AX_SYS_MemFree(m_io_data.pOutputs[i].phyAddr, m_io_data.pOutputs[i].pVirAddr);
                }
                delete[] m_io_data.pOutputs[i].pStride;
            }
            delete[] m_io_data.pOutputs;
            m_io_data.pOutputs = nullptr;
        }
        if (m_handle) {
            AX_ENGINE_DestroyHandle(m_handle);
            m_handle = nullptr;
        }
        return 0;
    }

private:
    struct Object {
        float x;
        float y;
        float w;
        float h;
        int label;
        float score;
    };

    AX_ENGINE_HANDLE m_handle = nullptr;
    AX_ENGINE_IO_INFO_T* m_io_info = nullptr;
    AX_ENGINE_IO_T m_io_data = {0};
    float confThresh_ = CONF_THRESH;
    float nmsThresh_ = NMS_THRESH;

    float* output0() {
        // Prefer the tensor with exact output0 byte size. If AX Engine exposes only
        // one output, fall back to output[0] to keep deployment tolerant of names.
        const size_t expected = static_cast<size_t>(MANHOLE_CLASS_NUM + 4) * MANHOLE_ANCHOR_NUM * sizeof(float);
        for (unsigned int i = 0; i < m_io_data.nOutputSize; ++i) {
            if (m_io_data.pOutputs[i].pVirAddr && m_io_data.pOutputs[i].nSize == expected) {
                return static_cast<float*>(m_io_data.pOutputs[i].pVirAddr);
            }
        }
        return m_io_data.nOutputSize > 0 ? static_cast<float*>(m_io_data.pOutputs[0].pVirAddr) : nullptr;
    }

    void generate_proposals(std::vector<Object>& proposals) {
        float* out = output0();
        if (!out) return;
        constexpr int channels = MANHOLE_CLASS_NUM + 4;
        // Layout follows validation_common.py: [channels, anchors].
        // Channels 0..3 are cx/cy/w/h in letterboxed 640x640 pixels;
        // channels 4..8 are direct class scores, not DFL logits.
        for (int i = 0; i < MANHOLE_ANCHOR_NUM; ++i) {
            float best_score = out[(4 + 0) * MANHOLE_ANCHOR_NUM + i];
            int best_label = 0;
            for (int c = 1; c < MANHOLE_CLASS_NUM; ++c) {
                float score = out[(4 + c) * MANHOLE_ANCHOR_NUM + i];
                if (score > best_score) {
                    best_score = score;
                    best_label = c;
                }
            }
            if (!std::isfinite(best_score) || best_score < confThresh_) continue;

            float cx = out[0 * MANHOLE_ANCHOR_NUM + i];
            float cy = out[1 * MANHOLE_ANCHOR_NUM + i];
            float w = out[2 * MANHOLE_ANCHOR_NUM + i];
            float h = out[3 * MANHOLE_ANCHOR_NUM + i];
            if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h)) continue;

            Object obj;
            obj.x = cx - w * 0.5f;
            obj.y = cy - h * 0.5f;
            obj.w = w;
            obj.h = h;
            obj.label = best_label;
            obj.score = best_score;
            proposals.push_back(obj);
        }
        (void)channels;
    }

    static void sort_descent(std::vector<Object>& objects) {
        std::sort(objects.begin(), objects.end(),
                  [](const Object& a, const Object& b) { return a.score > b.score; });
    }

    static float iou(const Object& a, const Object& b) {
        float x1 = std::max(a.x, b.x);
        float y1 = std::max(a.y, b.y);
        float x2 = std::min(a.x + a.w, b.x + b.w);
        float y2 = std::min(a.y + a.h, b.y + b.h);
        float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float area_a = std::max(0.0f, a.w) * std::max(0.0f, a.h);
        float area_b = std::max(0.0f, b.w) * std::max(0.0f, b.h);
        return inter / (area_a + area_b - inter + 1e-7f);
    }

    static void nms_by_class(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold) {
        picked.clear();
        for (int i = 0; i < static_cast<int>(objects.size()); ++i) {
            bool keep = true;
            for (int j : picked) {
                if (objects[i].label == objects[j].label && iou(objects[i], objects[j]) > nms_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) picked.push_back(i);
        }
    }
};

extern "C" {
    IAIModel* CreateAIModel() { return new ManholeCoverModel(); }
    void DestroyAIModel(IAIModel* p) { delete p; }
}
