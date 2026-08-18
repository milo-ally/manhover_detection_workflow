#include "ai_interface.h"
#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <deque>
#include <map>
#include <set>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "letterbox_utils.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- 模型與算法參數配置 ---
#define REG_MAX 16
#define CONF_THRESH 0.25f
#define NMS_THRESH 0.45f
#define NUM_KEYPOINTS 17 

// 關鍵點索引
enum KeypointIndex {
    KP_NOSE = 0, KP_L_EYE = 1, KP_R_EYE = 2, KP_L_EAR = 3, KP_R_EAR = 4,
    KP_L_SHOULDER = 5, KP_R_SHOULDER = 6, KP_L_ELBOW = 7, KP_R_ELBOW = 8,
    KP_L_WRIST = 9, KP_R_WRIST = 10, KP_L_HIP = 11, KP_R_HIP = 12,
    KP_L_KNEE = 13, KP_R_KNEE = 14, KP_L_ANKLE = 15, KP_R_ANKLE = 16
};

// 狀態定義
enum class State {
    STANDING,   // 站立
    SITTING,    // 坐下
    FALL_RISK,  // 跌倒預警
    FALLEN,     // 確認跌倒
    UNKNOWN
};

// 參數配置
struct FallDetectionConfig {
    float smooth_alpha = 0.3f;          // 平滑係數
    float fall_speed_threshold = 0.05f; // 跌倒速度閾值
    float fall_angle_trigger = 45.0f;   // 跌倒角度觸發 (小於此值視為倒下)
    float safety_vertical_angle = 60.0f; // 安全垂直角度 (大於此值視為直立)
    
    // 大腿垂直展開度閾值 (Knee.y - Hip.y) / Ref_Height
    // 原本 0.18 -> 降為 0.10。
    // 適應俯視角攝像頭：即使透視導致大腿看起來很短，只要有 10% 的垂直落差，仍判定為站立。
    float thigh_stand_threshold = 0.10f; 
    
    // 坐姿 BBox 寬高比閾值 (Width / Height)
    // 原本 0.55 -> 放寬至 0.65。
    // 只有當 BBox 變得非常寬 (大於 0.65) 時，才傾向於認為是坐下，防止揹包/厚衣服導致誤判。
    float aspect_ratio_sit_threshold = 0.65f;

    float ground_threshold_ratio = 0.4f; // 距地高度閾值 (髖部離地 < 40% 身高)
    
    int trigger_frames = 4;   // 跌倒觸發幀數
    int confirm_frames = 10;  // 跌倒確認幀數
    int recovery_frames = 15; // 站立恢復幀數
};

static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static float read_env_float(const char* key, float def) {
    const char* v = std::getenv(key);
    if (!v || !*v) return def;
    return static_cast<float>(atof(v));
}

static float compute_iou(const float* box1, const float* box2) {
    float x1 = std::max(box1[0], box2[0]);
    float y1 = std::max(box1[1], box2[1]);
    float x2 = std::min(box1[0] + box1[2], box2[0] + box2[2]);
    float y2 = std::min(box1[1] + box1[3], box2[1] + box2[3]);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = box1[2]*box1[3] + box2[2]*box2[3] - inter;
    return (union_area > 0) ? (inter / union_area) : 0.0f;
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

// --- 核心類 ---
class Yolo11PoseFallModel : public IAIModel {
public:
    int Init(const char* model_path) override {
        printf("[StrictFall] Loading model: %s\n", model_path);
        confThresh_ = read_env_float("FALL_CONF_THRESH", read_env_float("MODEL_CONF_THRESH", CONF_THRESH));
        nmsThresh_ = read_env_float("FALL_NMS_THRESH", read_env_float("MODEL_NMS_THRESH", NMS_THRESH));
        printf("[StrictFall] thresholds: conf=%.3f nms=%.3f\n", confThresh_, nmsThresh_);
        std::vector<char> model_buffer = read_model_file(model_path);
        if (model_buffer.empty()) return -1;

        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE; 

        if (AX_ENGINE_CreateHandle(&m_handle, model_buffer.data(), model_buffer.size()) != 0) return -1;
        AX_ENGINE_GetIOInfo(m_handle, &m_io_info);
        
        m_io_data.pInputs = new AX_ENGINE_IO_BUFFER_T[m_io_info->nInputSize];
        m_io_data.nInputSize = m_io_info->nInputSize;
        m_io_data.pOutputs = new AX_ENGINE_IO_BUFFER_T[m_io_info->nOutputSize];
        m_io_data.nOutputSize = m_io_info->nOutputSize;

        int ret = 0;
        for (unsigned int i = 0; i < m_io_info->nInputSize; ++i) {
            memset(&m_io_data.pInputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));
            AX_U32 size = 640 * 640 * 3; 
            ret = AX_SYS_MemAlloc(&m_io_data.pInputs[i].phyAddr, &m_io_data.pInputs[i].pVirAddr, size, 128, (const AX_S8*)"ax_input");
            if (ret != 0) goto init_error;
            m_io_data.pInputs[i].nSize = size;
            m_io_data.pInputs[i].pStride = new AX_S32[4];
            m_io_data.pInputs[i].pStride[0] = 640 * 3; 
        }

        for (unsigned int i = 0; i < m_io_info->nOutputSize; ++i) {
            memset(&m_io_data.pOutputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));
            auto& info = m_io_info->pOutputs[i];
            ret = AX_SYS_MemAlloc(&m_io_data.pOutputs[i].phyAddr, &m_io_data.pOutputs[i].pVirAddr, info.nSize, 128, (const AX_S8*)"ax_output");
            if (ret != 0) goto init_error;
            m_io_data.pOutputs[i].nSize = info.nSize;
            m_io_data.pOutputs[i].pStride = new AX_S32[4];
        }
        
        m_input_w = 640;
        m_input_h = 640;
        return 0;

    init_error:
        Deinit();
        return -1;
    }

    void GetInputSize(int* w, int* h) override { *w = m_input_w; *h = m_input_h; }

    int Inference(const AX_VIDEO_FRAME_T* pFrame, AI_RESULT_T* pResult) override {
        if (!m_handle) return -1;

        cv::Mat nv12_mat(pFrame->u32Height * 3 / 2, pFrame->u32Width, CV_8UC1, (void*)pFrame->u64VirAddr[0]);
        cv::Mat bgr_mat;
        cv::cvtColor(nv12_mat, bgr_mat, cv::COLOR_YUV2BGR_NV12);
        
        cv::Mat input_img;
        ai_letterbox::LetterboxInfo lb_info = ai_letterbox::letterbox(bgr_mat, input_img, m_input_w, m_input_h);

        memcpy(m_io_data.pInputs[0].pVirAddr, input_img.data, m_input_w * m_input_h * 3);
        AX_SYS_MflushCache(m_io_data.pInputs[0].phyAddr, m_io_data.pInputs[0].pVirAddr, m_io_data.pInputs[0].nSize);

        if (AX_ENGINE_RunSync(m_handle, &m_io_data) != 0) return -1;

        std::vector<PoseObject> proposals;
        generate_proposals(proposals); 
        
        for(auto& p : proposals) scale_coords(p, lb_info);
        qsort_descent_inplace(proposals);
        
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nmsThresh_);

        pResult->nObjSize = 0;
        matched_trackers_this_frame_.clear();

        for (int i = 0; i < (int)picked.size() && pResult->nObjSize < MAX_DETECT_OBJ_NUM; i++) {
            const auto& prop = proposals[picked[i]];
            // 放寬最小尺寸限制，避免漏檢遠處的人
            if (prop.bbox.w * prop.bbox.h < 300) continue; 

            PersonTracker* tracker = getOrCreateTracker(prop);
            
            update_tracker_features(tracker, prop);
            update_tracker_state(tracker);

            AI_OBJ_T& out_obj = pResult->objects[pResult->nObjSize++];
            fill_result(out_obj, prop, tracker, bgr_mat.cols, bgr_mat.rows);
        }
        
        cleanupLostTrackers();
        return 0;
    }

    int Deinit() override {
        if (m_handle) {
            for(unsigned i=0; i<m_io_data.nInputSize; i++) {
                if(m_io_data.pInputs[i].pVirAddr) 
                    AX_SYS_MemFree(m_io_data.pInputs[i].phyAddr, m_io_data.pInputs[i].pVirAddr);
                if(m_io_data.pInputs[i].pStride)
                    delete[] m_io_data.pInputs[i].pStride;
            }
            delete[] m_io_data.pInputs;
            
            for(unsigned i=0; i<m_io_data.nOutputSize; i++) {
                if(m_io_data.pOutputs[i].pVirAddr)
                    AX_SYS_MemFree(m_io_data.pOutputs[i].phyAddr, m_io_data.pOutputs[i].pVirAddr);
                if(m_io_data.pOutputs[i].pStride)
                    delete[] m_io_data.pOutputs[i].pStride;
            }
            delete[] m_io_data.pOutputs;
            AX_ENGINE_DestroyHandle(m_handle);
            m_handle = nullptr;
        }
        return 0;
    }

private:
    struct Rect { float x, y, w, h; };
    struct Keypoint { float x, y; float conf; };
    struct PoseObject { 
        Rect bbox; 
        int label; 
        float prob;
        std::vector<Keypoint> keypoints;
    };
    
    struct PersonTracker {
        int id;
        State state = State::STANDING;
        int state_counter = 0;
        int lost_frames = 0;
        bool initialized = false;
        
        cv::Point2f sm_neck = {0,0};
        cv::Point2f sm_hip = {0,0};
        cv::Point2f sm_knee = {0,0};
        cv::Point2f sm_ankle = {0,0};
        
        float max_height_history = 0.0f; 
        float spine_angle = 90.0f;       
        float thigh_openness = 1.0f;     // 大腿垂直展開度
        float vertical_speed = 0.0f;     
        float width_height_ratio = 0.0f; // BBox 寬高比 (W/H)
        float hip_dist_to_ground = 0.0f;

        bool is_legs_occluded = false;   // 腿部遮擋標記

        // 狀態積分器，用於防抖
        // > 0 傾向於 Sitting, < 0 傾向於 Standing
        int sit_stand_score = 0; 

        std::deque<float> neck_y_hist;
        cv::Point2f last_center;
        Rect last_bbox;

        PersonTracker() { neck_y_hist.resize(5, 0.0f); }
    };

    AX_ENGINE_HANDLE m_handle = nullptr;
    AX_ENGINE_IO_INFO_T* m_io_info = nullptr;
    AX_ENGINE_IO_T m_io_data = {0};
    int m_input_w = 640;
    int m_input_h = 640;
    float confThresh_ = CONF_THRESH;
    float nmsThresh_ = NMS_THRESH;
    FallDetectionConfig m_config;

    std::map<int, PersonTracker> trackers_;
    int next_tracker_id_ = 0;
    std::set<PersonTracker*> matched_trackers_this_frame_;

    // --- 輔助函數 ---

    cv::Point2f get_stable_point(const std::vector<Keypoint>& kps, int idx_l, int idx_r, 
                                const Rect& bbox, float ratio_y, cv::Point2f last_pos, bool& is_valid) {
        float conf_l = kps[idx_l].conf;
        float conf_r = kps[idx_r].conf;
        is_valid = true;

        if (conf_l > 0.3f && conf_r > 0.3f) {
            return cv::Point2f((kps[idx_l].x + kps[idx_r].x) / 2.0f, (kps[idx_l].y + kps[idx_r].y) / 2.0f);
        }
        else if (conf_l > 0.3f) return cv::Point2f(kps[idx_l].x, kps[idx_l].y);
        else if (conf_r > 0.3f) return cv::Point2f(kps[idx_r].x, kps[idx_r].y);
        
        if (last_pos.x > 0.1f) {
            is_valid = false;
            return last_pos; 
        }

        is_valid = false;
        return cv::Point2f(bbox.x + bbox.w / 2.0f, bbox.y + bbox.h * ratio_y);
    }

    // --- 特徵提取 ---
    void update_tracker_features(PersonTracker* t, const PoseObject& obj) {
        bool valid_neck, valid_hip, valid_knee, valid_ankle;
        
        cv::Point2f cur_neck = get_stable_point(obj.keypoints, KP_L_SHOULDER, KP_R_SHOULDER, obj.bbox, 0.15f, t->sm_neck, valid_neck);
        cv::Point2f cur_hip  = get_stable_point(obj.keypoints, KP_L_HIP, KP_R_HIP, obj.bbox, 0.5f, t->sm_hip, valid_hip);
        cv::Point2f cur_knee = get_stable_point(obj.keypoints, KP_L_KNEE, KP_R_KNEE, obj.bbox, 0.75f, t->sm_knee, valid_knee);
        cv::Point2f cur_ankle= get_stable_point(obj.keypoints, KP_L_ANKLE, KP_R_ANKLE, obj.bbox, 0.95f, t->sm_ankle, valid_ankle);

        // 判定腿部是否被遮擋
        // 如果膝蓋和腳踝置信度都很低，標記為遮擋
        float conf_knees = std::max(obj.keypoints[KP_L_KNEE].conf, obj.keypoints[KP_R_KNEE].conf);
        float conf_ankles = std::max(obj.keypoints[KP_L_ANKLE].conf, obj.keypoints[KP_R_ANKLE].conf);
        t->is_legs_occluded = (conf_knees < 0.25f && conf_ankles < 0.25f);

        float alpha = m_config.smooth_alpha;
        if (!t->initialized) {
            t->sm_neck = cur_neck; t->sm_hip = cur_hip; t->sm_knee = cur_knee; t->sm_ankle = cur_ankle;
            t->max_height_history = obj.bbox.h;
            t->initialized = true;
        } else {
            // 平滑更新
            float a_base = alpha;
            // 遮擋時不要更新腿部，防止飄移
            float a_leg = t->is_legs_occluded ? 0.0f : alpha;
            
            t->sm_neck = t->sm_neck * (1.0f - a_base) + cur_neck * a_base;
            t->sm_hip  = t->sm_hip  * (1.0f - a_base) + cur_hip  * a_base;
            t->sm_knee = t->sm_knee * (1.0f - a_leg) + cur_knee * a_leg;
            t->sm_ankle = t->sm_ankle * (1.0f - a_leg) + cur_ankle * a_leg;
        }

        if (obj.bbox.h > t->max_height_history) t->max_height_history = obj.bbox.h;
        else t->max_height_history *= 0.998f; 
        float ref_height = std::max(t->max_height_history, 20.0f);

        // 1. 脊柱角度
        float dx = std::abs(t->sm_neck.x - t->sm_hip.x);
        float dy = t->sm_hip.y - t->sm_neck.y; 
        if (dy < 0) dy = 0; 
        t->spine_angle = atan2(dy, dx) * 180.0f / M_PI;

        // 2. 大腿垂直展開度 (Thigh Openness)
        // 站立時：膝蓋在髖部下方很遠處 -> 值大
        // 坐姿時：膝蓋在髖部同高度(側面) 或 短縮(正面) -> 值小
        if (t->is_legs_occluded) {
            t->thigh_openness = -1.0f; // 無效值
        } else {
            float thigh_dy = t->sm_knee.y - t->sm_hip.y;
            t->thigh_openness = thigh_dy / ref_height;
        }

        // 3. 距地高度
        if (t->is_legs_occluded) {
            t->hip_dist_to_ground = ref_height; // 遮擋時假定很高
        } else {
            float ground_y = std::max(t->sm_ankle.y, obj.bbox.y + obj.bbox.h); 
            t->hip_dist_to_ground = ground_y - t->sm_hip.y;
        }

        // 4. 速度
        if (valid_neck && t->neck_y_hist.size() >= 3) {
            float prev_y = t->neck_y_hist.front();
            t->vertical_speed = (t->sm_neck.y - prev_y) / ref_height;
        } else {
            t->vertical_speed *= 0.5f;
        }
        
        t->neck_y_hist.push_back(t->sm_neck.y);
        if (t->neck_y_hist.size() > 5) t->neck_y_hist.pop_front();
        
        t->width_height_ratio = obj.bbox.w / obj.bbox.h;
        t->last_bbox = obj.bbox;
    }

    // --- 狀態機 ---
    void update_tracker_state(PersonTracker* t) {
        if (!t->initialized) return;

        // --- 1. 計算基礎特徵標誌位 ---
        bool is_safe_vertical = (t->spine_angle > m_config.safety_vertical_angle);
        bool is_lying_angle = (t->spine_angle < m_config.fall_angle_trigger);
        bool is_fast_drop = (t->vertical_speed > m_config.fall_speed_threshold);
        bool is_near_ground = (t->hip_dist_to_ground < t->max_height_history * m_config.ground_threshold_ratio);
        
        // --- 2. 坐姿/站姿 智能判斷邏輯 ---

        // A. 腿部特徵 (最準確，但在遮擋或俯視時受限)
        // 站立：展開度大於閾值 (0.10)
        bool legs_say_stand = (!t->is_legs_occluded && t->thigh_openness > m_config.thigh_stand_threshold);
        // 坐下：展開度顯著小於閾值 (0.08)，形成死區防止抖動
        bool legs_say_sit   = (!t->is_legs_occluded && t->thigh_openness < m_config.thigh_stand_threshold * 0.8f);

        // B. 形狀特徵 (備用，用於遮擋時)
        // 站立：瘦高 (Ratio < 0.65)
        bool shape_say_stand = (t->width_height_ratio < m_config.aspect_ratio_sit_threshold);
        // 坐下：寬扁 (Ratio > 0.71)
        bool shape_say_sit   = (t->width_height_ratio > m_config.aspect_ratio_sit_threshold * 1.1f);

        // C. 強站立保護 (Strong Standing Guard)
        // 解決俯視角誤判的核心：
        // 只要脊柱夠直 (>60度)，且膝蓋確實位於髖部下方一點點 (>0.05)，且沒有完全被遮擋
        // -> 強制視為站立 (忽略大腿看起來很短的視覺誤差)
        bool strong_stand_guard = (is_safe_vertical && t->thigh_openness > 0.05f && !t->is_legs_occluded);

        // --- 3. 計算當前幀積分 (Score) ---
        // Score > 0 傾向坐下 (Sitting)
        // Score < 0 傾向站立 (Standing)
        int frame_score = 0;

        if (strong_stand_guard) {
            frame_score = -5; // 強烈判定為站立
        } 
        else if (!t->is_legs_occluded) {
            // == 腿部可見 ==
            if (legs_say_sit) {
                // 腿確實縮起來了 -> 坐下
                frame_score += 2;
            } else if (legs_say_stand) {
                // 腿很長 -> 站立
                frame_score -= 2;
            } else {
                // 腿部長度尷尬 (0.08 ~ 0.10 之間)
                // 此時看身體是否直立，如果直立，傾向於站
                if (is_safe_vertical) frame_score -= 1;
            }
        } 
        else {
            // == 腿部被遮擋 (櫃檯、桌子) ==
            // 依賴 BBox 形狀
            if (shape_say_sit) frame_score += 1;
            else if (shape_say_stand) frame_score -= 1;
            
            // 遮擋情況下，如果上半身很直，默認傾向於站立 (如櫃檯後的服務員)
            // 除非形狀特別寬 (上面 shape_say_sit 會抵消這個 -1，變成 0)
            if (is_safe_vertical) frame_score -= 1;
        }

        // --- 4. 更新積分器與基礎姿態 ---
        t->sit_stand_score += frame_score;
        // 限制積分範圍 (防抖緩衝區)
        if (t->sit_stand_score > 10) t->sit_stand_score = 10;
        if (t->sit_stand_score < -10) t->sit_stand_score = -10;

        // 決定 Base Pose (穩定的姿態)
        State base_pose = State::STANDING;
        if (t->sit_stand_score > 2) base_pose = State::SITTING;
        else if (t->sit_stand_score < -2) base_pose = State::STANDING;
        else {
            // 積分在 -2 ~ 2 之間 (模糊區)，保持上一幀的非跌倒狀態
            if (t->state == State::SITTING) base_pose = State::SITTING;
            else base_pose = State::STANDING;
        }

        // --- 5. 狀態機轉移 ---
        State next = t->state;

        switch (t->state) {
            case State::STANDING:
            case State::SITTING:
            case State::UNKNOWN:
                // 優先級 1: 跌倒檢測
                if (is_fast_drop) {
                    if (is_safe_vertical) {
                        // 速度快但身體直 -> 快速坐下/蹲下
                        next = State::SITTING;
                        t->sit_stand_score = 5; // 快速將積分推向坐姿
                    } else if (frame_score > 0) {
                         // 速度快 + 幾何特徵像坐 -> 坐下
                         next = State::SITTING;
                    } else {
                         // 速度快 + 身體歪 + 不像坐 -> 跌倒風險
                         next = State::FALL_RISK;
                         t->state_counter = 0;
                    }
                }
                // 優先級 2: 靜態倒地 (必須同時滿足：角度平 + 離地近 + 寬BBox)
                // 半身遮擋時 is_near_ground 為 false (保護)，不會誤觸發
                else if (is_lying_angle && is_near_ground && t->width_height_ratio > 1.0f) {
                    t->state_counter++;
                    if (t->state_counter > m_config.trigger_frames * 2) next = State::FALLEN;
                }
                // 優先級 3: 正常姿態更新 (Sit <-> Stand)
                else {
                    next = base_pose;
                    t->state_counter = 0;
                }
                break;

            case State::FALL_RISK:
                // 風險期觀察
                if (is_lying_angle || (t->width_height_ratio > 1.2f && is_near_ground)) {
                    t->state_counter++;
                    if (t->state_counter >= m_config.trigger_frames) {
                        next = State::FALLEN;
                    }
                } else if (t->vertical_speed < 0.005f) { 
                    // 速度停止了
                    if (is_safe_vertical || base_pose == State::SITTING || !is_near_ground) {
                        // 恢復正常
                        next = base_pose; 
                    } else {
                        // 停下來了還是歪的 -> 確認跌倒
                        next = State::FALLEN;
                    }
                }
                break;

            case State::FALLEN:
                // 恢復邏輯
                // 條件：(身體變直 + 高度恢復) 或者 (半身遮擋且身體變直)
                bool height_recovered = (t->last_bbox.h > t->max_height_history * 0.75f);
                if (t->is_legs_occluded && is_safe_vertical) height_recovered = true; 

                if (is_safe_vertical && !is_lying_angle && height_recovered) {
                    t->state_counter++;
                    if (t->state_counter > m_config.recovery_frames) { 
                        next = State::STANDING;
                        t->state_counter = 0;
                        t->sit_stand_score = -5; // 恢復後重置為站立傾向
                    }
                } else {
                    // 軟重置計數器 (Decay)，容忍少量抖動
                    t->state_counter = std::max(0, t->state_counter - 1); 
                }
                break;
        }

        t->state = next;
    }

    void fill_result(AI_OBJ_T& out_obj, const PoseObject& prop, PersonTracker* tracker, int src_w, int src_h) {
        out_obj.x = prop.bbox.x / src_w;
        out_obj.y = prop.bbox.y / src_h;
        out_obj.w = prop.bbox.w / src_w;
        out_obj.h = prop.bbox.h / src_h;
        out_obj.score = prop.prob;
        out_obj.track_id = tracker->id;
        
        const char* label = "unknown";
        int class_id = 0;
        
        switch(tracker->state) {
            case State::STANDING: label = "standing"; break;
            case State::SITTING:  label = "sitting"; break;
            case State::FALL_RISK: label = "falling?"; class_id = 1; break;
            case State::FALLEN:    label = "FALLEN!"; class_id = 2; break;
            default: break;
        }
        
        snprintf(out_obj.label, 32, "%s", label);
        out_obj.class_id = class_id;

        out_obj.nKeypoints = std::min((int)prop.keypoints.size(), NUM_KEYPOINTS);
        for(int k=0; k<out_obj.nKeypoints; k++) {
            out_obj.keypoints[k].x = prop.keypoints[k].x / src_w;
            out_obj.keypoints[k].y = prop.keypoints[k].y / src_h;
            out_obj.keypoints[k].conf = prop.keypoints[k].conf;
        }
    }

    void scale_coords(PoseObject& obj, const ai_letterbox::LetterboxInfo& lb) {
        ai_letterbox::scale_bbox_to_original(obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h, lb);
        auto restore = [&](float& x, float& y) {
            x = (x - (float)lb.pad_w) / lb.scale;
            y = (y - (float)lb.pad_h) / lb.scale;
        };
        for (auto& kp : obj.keypoints) restore(kp.x, kp.y);
    }

    PersonTracker* getOrCreateTracker(const PoseObject& obj) {
        PersonTracker* best = nullptr;
        float max_iou = 0.0f;
        float current_box[4] = {obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h};

        for (auto& pair : trackers_) {
            if (matched_trackers_this_frame_.count(&pair.second)) continue;
            
            float prev_box[4] = {pair.second.last_bbox.x, pair.second.last_bbox.y, 
                                 pair.second.last_bbox.w, pair.second.last_bbox.h};
            
            float iou = compute_iou(current_box, prev_box);
            if (iou > 0.3f && iou > max_iou) { 
                max_iou = iou;
                best = &pair.second;
            }
        }

        if (!best) {
            cv::Point2f center(obj.bbox.x + obj.bbox.w/2, obj.bbox.y + obj.bbox.h/2);
            float min_dist = 1e9;
            for (auto& pair : trackers_) {
                if (matched_trackers_this_frame_.count(&pair.second)) continue;
                float dist = cv::norm(center - pair.second.last_center);
                float limit = std::max(obj.bbox.w, obj.bbox.h) * 1.5f; 
                if (dist < limit && dist < min_dist) {
                    min_dist = dist;
                    best = &pair.second;
                }
            }
        }

        if (!best) {
            int id = next_tracker_id_++;
            trackers_[id].id = id;
            best = &trackers_[id];
        }
        
        matched_trackers_this_frame_.insert(best);
        best->last_center = cv::Point2f(obj.bbox.x + obj.bbox.w/2, obj.bbox.y + obj.bbox.h/2);
        best->lost_frames = 0;
        return best;
    }

    void cleanupLostTrackers() {
        for (auto it = trackers_.begin(); it != trackers_.end();) {
            if (matched_trackers_this_frame_.find(&it->second) == matched_trackers_this_frame_.end()) {
                it->second.lost_frames++;
                if (it->second.lost_frames > 30) it = trackers_.erase(it);
                else ++it; 
            } else ++it;
        }
    }

    void generate_proposals(std::vector<PoseObject>& proposals) {
        float* output_ptr[6] = {nullptr};
        for (unsigned int i = 0; i < m_io_info->nOutputSize && i < 6; ++i) {
            output_ptr[i] = (float*)m_io_data.pOutputs[i].pVirAddr;
        }
        bool has_separate_kps = (m_io_info->nOutputSize >= 6 && output_ptr[3] != nullptr);

        for (int i = 0; i < 3; ++i) {
            if (!output_ptr[i]) continue;
            auto& output_info = m_io_info->pOutputs[i];
            int stride = (1 << i) * 8; 
            int H = output_info.pShape[1];
            int W = output_info.pShape[2];
            int C = output_info.pShape[3];
            float* kps_output_ptr = (has_separate_kps) ? output_ptr[i+3] : nullptr;
            int kps_C = (has_separate_kps) ? m_io_info->pOutputs[i+3].pShape[3] : 0;

            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    float* feat_ptr = output_ptr[i] + (h * W + w) * C;
                    int score_idx = -1;
                    int kps_start = -1;
                    float pred_ltrb[4];
                    
                    if (C == 65) score_idx = 64; 
                    else if (C == 56) { score_idx = 55; kps_start = 4; } 
                    else if (C >= 116) { score_idx = 115; kps_start = 64; } 

                    if (score_idx == -1) continue;
                    
                    if (C != 56) {
                        std::vector<float> dis_after_sm(REG_MAX, 0.f);
                        for (int k = 0; k < 4; k++) {
                            const float* src = feat_ptr + k * REG_MAX;
                            float max_val = -1e9f; for(int r=0;r<REG_MAX;r++) if(src[r]>max_val) max_val=src[r];
                            float sum = 0; for(int r=0; r<REG_MAX; r++) { dis_after_sm[r] = expf(src[r]-max_val); sum+=dis_after_sm[r]; }
                            float w_sum = 0; for(int r=0; r<REG_MAX; r++) { w_sum += (dis_after_sm[r]/sum)*r; }
                            pred_ltrb[k] = w_sum * stride;
                        }
                    } else {
                         pred_ltrb[0] = feat_ptr[0] * stride;
                         pred_ltrb[1] = feat_ptr[1] * stride;
                         pred_ltrb[2] = feat_ptr[2] * stride;
                         pred_ltrb[3] = feat_ptr[3] * stride;
                    }

                    float box_prob = sigmoid(feat_ptr[score_idx]);
                    if (box_prob < confThresh_) continue;

                    float pb_cx = (w + 0.5f) * stride;
                    float pb_cy = (h + 0.5f) * stride;
                    float x0 = pb_cx - pred_ltrb[0];
                    float y0 = pb_cy - pred_ltrb[1];
                    float x1 = pb_cx + pred_ltrb[2];
                    float y1 = pb_cy + pred_ltrb[3];
                    
                    x0 = std::max(std::min(x0, (float)(m_input_w - 1)), 0.f);
                    y0 = std::max(std::min(y0, (float)(m_input_h - 1)), 0.f);
                    x1 = std::max(std::min(x1, (float)(m_input_w - 1)), 0.f);
                    y1 = std::max(std::min(y1, (float)(m_input_h - 1)), 0.f);

                    PoseObject obj;
                    obj.bbox = {x0, y0, x1 - x0, y1 - y0};
                    obj.prob = box_prob;
                    obj.keypoints.resize(NUM_KEYPOINTS);

                    float* k_ptr = nullptr;
                    if (kps_output_ptr) k_ptr = kps_output_ptr + (h * W + w) * kps_C;
                    else if (kps_start >= 0) k_ptr = feat_ptr + kps_start;

                    if (k_ptr) {
                        for (int k = 0; k < NUM_KEYPOINTS; k++) {
                            float kx = (k_ptr[k*3] * 2.0f + w) * stride;
                            float ky = (k_ptr[k*3+1] * 2.0f + h) * stride;
                            float conf = sigmoid(k_ptr[k*3+2]);
                            kx = std::max(std::min(kx, (float)(m_input_w - 1)), 0.f);
                            ky = std::max(std::min(ky, (float)(m_input_h - 1)), 0.f);
                            obj.keypoints[k] = {kx, ky, conf};
                        }
                    }
                    proposals.push_back(obj);
                }
            }
        }
    }

    void qsort_descent_inplace(std::vector<PoseObject>& objs) {
        std::sort(objs.begin(), objs.end(), [](const PoseObject& a, const PoseObject& b) { return a.prob > b.prob; });
    }

    void nms_sorted_bboxes(const std::vector<PoseObject>& objs, std::vector<int>& picked, float nms_thresh) {
        picked.clear();
        const int n = objs.size();
        std::vector<float> areas(n);
        for (int i = 0; i < n; i++) areas[i] = objs[i].bbox.w * objs[i].bbox.h;
        for (int i = 0; i < n; i++) {
            const PoseObject& a = objs[i];
            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++) {
                const PoseObject& b = objs[picked[j]];
                float inter_x1 = std::max(a.bbox.x, b.bbox.x);
                float inter_y1 = std::max(a.bbox.y, b.bbox.y);
                float inter_x2 = std::min(a.bbox.x + a.bbox.w, b.bbox.x + b.bbox.w);
                float inter_y2 = std::min(a.bbox.y + a.bbox.h, b.bbox.y + b.bbox.h);
                float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                if (inter_area / union_area > nms_thresh) { keep = 0; break; }
            }
            if (keep) picked.push_back(i);
        }
    }
};

extern "C" {
    IAIModel* CreateAIModel() { return new Yolo11PoseFallModel(); }
    void DestroyAIModel(IAIModel* p) { delete p; }
}