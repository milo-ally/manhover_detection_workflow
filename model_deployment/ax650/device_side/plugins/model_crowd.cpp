/**
 * 人員聚集插件：共用人員偵測模型，先做人員偵測 + ByteTrack 追蹤，再以 DBSCAN 分群，
 * 僅輸出 group bbox 與成員數（不輸出單人框）。
 */
#include "ai_interface.h"
#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include "BYTETracker.h"
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <deque>
#include <queue>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "letterbox_utils.hpp"

#define MODEL_CLASS_NUM 80
#define TARGET_CLASS_ID 0
#define REG_MAX 16
#define CONF_THRESH 0.25f
#define NMS_THRESH 0.45f

static float read_env_float(const char* key, float def) {
    const char* v = std::getenv(key);
    if (!v || !*v) return def;
    return static_cast<float>(atof(v));
}

// DBSCAN 聚集參數（歸一化座標 0~1）- 放寬版，便於確認有框
#define GROUP_EPS_NORM 0.18f      // 相鄰距離閾值（約 18% 圖寬，放寬以利成群）
#define GROUP_MIN_SAMPLES 2       // 至少 2 人才算一群（除錯時可改 1 會出現單人框）
#define GROUP_EXPAND_RATIO 0.1f   // 群框外擴比例
#define CROWD_LOG_INTERVAL 10     // 每 N 幀打一次 log，便於確認有無偵測/追蹤/群

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static std::vector<char> read_model_file(const char* filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) return {};
    return buffer;
}

// 簡易 DBSCAN：points 為歸一化 (x,y) 中心點，傳回每個點的 cluster label（-1 為噪聲）
static void dbscan(const std::vector<std::pair<float, float>>& points,
                   float eps, int minSamples,
                   std::vector<int>& labels) {
    const int n = (int)points.size();
    labels.assign(n, -1);
    if (n == 0 || eps <= 0.f) return;

    auto dist2 = [&](int i, int j) {
        float dx = points[i].first - points[j].first;
        float dy = points[i].second - points[j].second;
        return dx * dx + dy * dy;
    };

    std::vector<std::vector<int>> neighbors(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && dist2(i, j) <= eps * eps)
                neighbors[i].push_back(j);
        }
    }

    int clusterId = 0;
    std::vector<bool> visited(n, false);
    for (int i = 0; i < n; i++) {
        if (visited[i]) continue;
        visited[i] = true;
        if ((int)neighbors[i].size() < minSamples) continue;

        labels[i] = clusterId;
        std::deque<int> q(neighbors[i].begin(), neighbors[i].end());
        while (!q.empty()) {
            int p = q.front();
            q.pop_front();
            if (labels[p] >= 0) continue;
            labels[p] = clusterId;
            if (!visited[p]) {
                visited[p] = true;
                if ((int)neighbors[p].size() >= minSamples) {
                    for (int k : neighbors[p])
                        if (labels[k] < 0) q.push_back(k);
                }
            }
        }
        clusterId++;
    }
}

// 從追蹤結果建群組：tracks 為 (nx, ny, nw, nh) 歸一化 bbox，傳回 group_bbox (x,y,w,h norm) + members
struct GroupInfo {
    float x, y, w, h;
    int members;
};
static void build_groups(const std::vector<std::vector<float>>& trackNormBoxes,
                         std::vector<GroupInfo>& groups) {
    groups.clear();
    if (trackNormBoxes.empty()) return;

    std::vector<std::pair<float, float>> centers;
    for (const auto& b : trackNormBoxes) {
        if (b.size() >= 4)
            centers.push_back({ b[0] + b[2] * 0.5f, b[1] + b[3] * 0.5f });
    }
    if (centers.size() < (size_t)GROUP_MIN_SAMPLES) return;

    std::vector<int> labels;
    dbscan(centers, GROUP_EPS_NORM, GROUP_MIN_SAMPLES, labels);

    std::map<int, std::vector<size_t>> clusterToIdx;
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] >= 0)
            clusterToIdx[labels[i]].push_back(i);
    }

    for (const auto& kv : clusterToIdx) {
        const auto& indices = kv.second;
        float x1 = 1.f, y1 = 1.f, x2 = 0.f, y2 = 0.f;
        for (size_t idx : indices) {
            const auto& b = trackNormBoxes[idx];
            float bx1 = b[0], by1 = b[1], bx2 = b[0] + b[2], by2 = b[1] + b[3];
            if (bx1 < x1) x1 = bx1;
            if (by1 < y1) y1 = by1;
            if (bx2 > x2) x2 = bx2;
            if (by2 > y2) y2 = by2;
        }
        float w = x2 - x1;
        float h = y2 - y1;
        if (w <= 0.f || h <= 0.f) continue;
        float pad = GROUP_EXPAND_RATIO;
        x1 = std::max(0.f, x1 - w * pad);
        y1 = std::max(0.f, y1 - h * pad);
        x2 = std::min(1.f, x2 + w * pad);
        y2 = std::min(1.f, y2 + h * pad);
        w = x2 - x1;
        h = y2 - y1;

        GroupInfo g;
        g.x = x1;
        g.y = y1;
        g.w = w;
        g.h = h;
        g.members = (int)indices.size();
        groups.push_back(g);
    }
}

class Yolo11CrowdModel : public IAIModel {
public:
    int Init(const char* model_path) override {
        printf("[Crowd] Loading model (same as human): %s\n", model_path);
        confThresh_ = read_env_float("CROWD_CONF_THRESH", read_env_float("MODEL_CONF_THRESH", CONF_THRESH));
        nmsThresh_ = read_env_float("CROWD_NMS_THRESH", read_env_float("MODEL_NMS_THRESH", NMS_THRESH));
        printf("[Crowd] thresholds: conf=%.3f nms=%.3f\n", confThresh_, nmsThresh_);
        std::vector<char> model_buffer = read_model_file(model_path);
        if (model_buffer.empty()) {
            // 若無 yolo11_human_crowd.axmodel，則共用人員偵測模型文件
            const char* fallback = "../models/yolo11_human_detection.axmodel";
            printf("[Crowd] Fallback to: %s\n", fallback);
            model_buffer = read_model_file(fallback);
            if (model_buffer.empty()) return -1;
        }

        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;

        int ret = AX_ENGINE_CreateHandle(&m_handle, model_buffer.data(), model_buffer.size());
        if (ret != 0) {
            printf("[Crowd] CreateHandle failed: 0x%x\n", ret);
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

        m_input_w = 640;
        m_input_h = 640;
        m_tracker.reset(new BYTETracker(30, 25));
        printf("[Crowd] human_detection.axmodel 已載入，推理與聚集插件就緒\n");
        return 0;
    }

    void GetInputSize(int* w, int* h) override { *w = m_input_w; *h = m_input_h; }

    int Inference(const AX_VIDEO_FRAME_T* pFrame, AI_RESULT_T* pResult) override {
        if (!m_handle || !pResult) return -1;

        cv::Mat nv12_mat(pFrame->u32Height * 3 / 2, pFrame->u32Width, CV_8UC1,
                         (void*)pFrame->u64VirAddr[0], pFrame->u32PicStride[0]);
        cv::Mat rgb_mat;
        cv::cvtColor(nv12_mat, rgb_mat, cv::COLOR_YUV2RGB_NV12);
        cv::Mat input_rgb;
        ai_letterbox::LetterboxInfo lb_info = ai_letterbox::letterbox(rgb_mat, input_rgb, m_input_w, m_input_h);

        memcpy(m_io_data.pInputs[0].pVirAddr, input_rgb.data, m_input_w * m_input_h * 3);
        AX_SYS_MflushCache(m_io_data.pInputs[0].phyAddr, m_io_data.pInputs[0].pVirAddr, m_io_data.pInputs[0].nSize);

        int ret = AX_ENGINE_RunSync(m_handle, &m_io_data);
        if (ret != 0) {
            printf("[Crowd] AX_ENGINE_RunSync failed: %d\n", ret);
            return -1;
        }

        std::vector<Object> proposals;
        generate_proposals(proposals);
        qsort_descent_inplace(proposals);
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nmsThresh_);

        const int src_w = (int)pFrame->u32Width;
        const int src_h = (int)pFrame->u32Height;

        const int numDet = std::min((int)picked.size(), TRACK_OBJETCS_MAX_SIZE);
        track_object_t trackInput[TRACK_OBJETCS_MAX_SIZE];
        memset(trackInput, 0, sizeof(trackInput));
        for (int i = 0; i < numDet; i++) {
            Object prop = proposals[picked[i]];
            ai_letterbox::scale_bbox_to_original(prop.bbox.x, prop.bbox.y, prop.bbox.w, prop.bbox.h, lb_info);
            trackInput[i].label = 0;
            trackInput[i].prob = prop.prob;
            trackInput[i].rect.x = prop.bbox.x;
            trackInput[i].rect.y = prop.bbox.y;
            trackInput[i].rect.width = prop.bbox.w;
            trackInput[i].rect.height = prop.bbox.h;
        }

        std::vector<STrack> tracks = m_tracker->update(trackInput, numDet);

        size_t trackedCount = 0;
        std::vector<std::vector<float>> trackNormBoxes;
        for (const STrack& t : tracks) {
            if (t.state != TrackState::Tracked) continue;
            trackedCount++;
            if (t.tlwh.size() < 4) continue;
            float x = t.tlwh[0] / (float)src_w;
            float y = t.tlwh[1] / (float)src_h;
            float w = t.tlwh[2] / (float)src_w;
            float h = t.tlwh[3] / (float)src_h;
            trackNormBoxes.push_back({ x, y, w, h });
        }

        std::vector<GroupInfo> groups;
        build_groups(trackNormBoxes, groups);

        // 首次運行 log，確認回調有進入插件
        static int s_firstRun = 1;
        if (s_firstRun) {
            s_firstRun = 0;
            printf("[Crowd] First inference run: det=%d picked=%d tracked=%zu groups=%zu\n",
                   (int)proposals.size(), (int)picked.size(), trackedCount, groups.size());
        }
        // 週期性 log：確認偵測/追蹤/群數；若 tracked>=2 但 groups=0 多半是 DBSCAN 太嚴
        static int s_logCounter = 0;
        if (++s_logCounter >= CROWD_LOG_INTERVAL) {
            s_logCounter = 0;
            printf("[Crowd] det=%d picked=%d tracks=%zu tracked=%zu groups=%zu (eps=%.2f min=%d)\n",
                   (int)proposals.size(), (int)picked.size(), tracks.size(), trackedCount, groups.size(),
                   (double)GROUP_EPS_NORM, (int)GROUP_MIN_SAMPLES);
            if (trackedCount >= 2 && groups.empty())
                printf("[Crowd] tracked>=2 but groups=0 -> DBSCAN too strict? try larger eps or min_samples=1 for debug\n");
        }

        pResult->nObjSize = std::min((AX_U32)groups.size(), (AX_U32)MAX_DETECT_OBJ_NUM);
        for (AX_U32 i = 0; i < pResult->nObjSize; i++) {
            const GroupInfo& g = groups[i];
            pResult->objects[i].x = g.x;
            pResult->objects[i].y = g.y;
            pResult->objects[i].w = g.w;
            pResult->objects[i].h = g.h;
            pResult->objects[i].score = 1.0f;
            pResult->objects[i].class_id = g.members;
            pResult->objects[i].track_id = 0;
            snprintf(pResult->objects[i].label, sizeof(pResult->objects[i].label), "G (%d)", g.members);
            memset(pResult->objects[i].keypoints, 0, sizeof(pResult->objects[i].keypoints));
            pResult->objects[i].nKeypoints = 0;
        }
        return 0;
    }

    int Deinit() override {
        m_tracker.reset();
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
    AX_ENGINE_IO_T m_io_data = { 0 };
    int m_input_w = 640;
    int m_input_h = 640;
    float confThresh_ = CONF_THRESH;
    float nmsThresh_ = NMS_THRESH;
    std::unique_ptr<BYTETracker> m_tracker;

    struct Rect { float x, y, w, h; };
    struct Object { Rect bbox; int label; float prob; };
    struct HeadData { float* box; float* cls; bool is_combined; };

    void generate_proposals(std::vector<Object>& proposals) {
        std::map<int, HeadData> heads;
        for (unsigned int i = 0; i < m_io_info->nOutputSize; ++i) {
            auto& output = m_io_info->pOutputs[i];
            float* data = (float*)m_io_data.pOutputs[i].pVirAddr;
            if (!data) continue;
            int H = output.pShape[1];
            int C = output.pShape[3];
            if (heads.find(H) == heads.end()) heads[H] = { nullptr, nullptr, false };
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
            int W = H;
            int C_combined = 4 * REG_MAX + MODEL_CLASS_NUM;
            int C_box = 4 * REG_MAX;
            int C_cls = MODEL_CLASS_NUM;

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
                    int max_id = 0;
                    float max_prob = -1000.0f;
                    for (int c = 0; c < MODEL_CLASS_NUM; c++) {
                        if (cls_ptr[c] > max_prob) { max_prob = cls_ptr[c]; max_id = c; }
                    }
                    if (max_id != TARGET_CLASS_ID) continue;
                    float score = sigmoid(max_prob);
                    if (score < confThresh_) continue;

                    float pred_ltrb[4];
                    for (int k = 0; k < 4; k++) {
                        float exp_sum = 0.0f, weighted_sum = 0.0f;
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
                    obj.bbox = { x1, y1, x2 - x1, y2 - y1 };
                    obj.label = 0;
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
        const int n = (int)faceobjects.size();
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
                float inter_area = std::max(0.f, inter_x2 - inter_x1) * std::max(0.f, inter_y2 - inter_y1);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                if (union_area > 0 && inter_area / union_area > nms_threshold) { keep = 0; break; }
            }
            if (keep) picked.push_back(i);
        }
    }
};

extern "C" {
    IAIModel* CreateAIModel() { return new Yolo11CrowdModel(); }
    void DestroyAIModel(IAIModel* p) { delete p; }
}
