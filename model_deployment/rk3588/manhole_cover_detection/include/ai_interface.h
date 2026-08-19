#ifndef _AI_INTERFACE_H_
#define _AI_INTERFACE_H_

// RK3588 版插件 ABI。与 AX650 的 include/ai_interface.h 同构，但去掉了
// AX 平台类型依赖（AX_F32/AX_VIDEO_FRAME_T 等），使用平台无关类型，
// 使插件源码可以在 AX650 / RK3588 两侧编译（推理后端各自实现）。

#include <cstdint>

// 定義最大檢測目標數 (防止記憶體溢出)
#define MAX_DETECT_OBJ_NUM 64

// 定義關鍵點（用於姿態檢測）
#define MAX_KEYPOINTS 17  // COCO 17個關鍵點

typedef struct {
    float x, y;        // 歸一化座標 (0~1)
    float conf;        // 置信度 (0~1)
} AI_KEYPOINT_T;

// 定義通用的檢測物件
typedef struct {
    float x, y, w, h;  // 歸一化座標 (0~1)
    int32_t class_id;  // 類別 ID
    float score;       // 置信度
    uint64_t track_id; // 追蹤 ID (如果開啟 ByteTrack)
    char label[32];    // 類別名稱
    // 關鍵點信息（用於姿態檢測，如果沒有關鍵點則全部為0）
    AI_KEYPOINT_T keypoints[MAX_KEYPOINTS];
    uint32_t nKeypoints;  // 實際關鍵點數量
} AI_OBJ_T;

// 定義檢測結果集合
typedef struct {
    uint32_t nObjSize;    // 實際檢測到的物件數量
    AI_OBJ_T objects[MAX_DETECT_OBJ_NUM]; // 固定大小陣列，避免 STL 跨邊界問題
} AI_RESULT_T;

// 平台無關視頻幀描述（替代 AX_VIDEO_FRAME_T）
typedef enum {
    AI_FRAME_FORMAT_BGR24 = 0,  // 連續 BGR 三通道（OpenCV Mat 佈局）
    AI_FRAME_FORMAT_NV12,       // YUV420 半平面
} AI_FRAME_FORMAT_E;

typedef struct {
    const void* data;   // 幀數據指針
    int width;          // 寬度（像素）
    int height;         // 高度（像素）
    int stride;         // 行步長（字節）；0 表示緊湊排列
    int format;         // AI_FRAME_FORMAT_E
} AI_FRAME_T;

// AI 模型抽象介面
class IAIModel {
public:
    virtual ~IAIModel() {}

    // 初始化
    virtual int Init(const char* model_path) = 0;

    // 獲取模型需要的輸入寬高 (用於配置預處理)
    virtual void GetInputSize(int* w, int* h) = 0;

    // 推論 (傳入平台無關幀，輸出結果)
    virtual int Inference(const AI_FRAME_T* pFrame, AI_RESULT_T* pResult) = 0;

    // 反初始化
    virtual int Deinit() = 0;
};

// 定義導出函數類型 (用於 dlsym)
typedef IAIModel* (*CreateAIModelFunc)();
typedef void (*DestroyAIModelFunc)(IAIModel*);

#endif
