#ifndef _AI_INTERFACE_H_
#define _AI_INTERFACE_H_

#include "ax_global_type.h"
// #include <vector> 
#include <string>

// 定義最大檢測目標數 (防止記憶體溢出)
#define MAX_DETECT_OBJ_NUM 64

// 定義關鍵點（用於姿態檢測）
#define MAX_KEYPOINTS 17  // COCO 17個關鍵點
typedef struct {
    AX_F32 x, y;        // 歸一化座標 (0~1)
    AX_F32 conf;        // 置信度 (0~1)
} AI_KEYPOINT_T;

// 定義通用的檢測物件
typedef struct {
    AX_F32 x, y, w, h;  // 歸一化座標 (0~1)
    AX_S32 class_id;    // 類別 ID
    AX_F32 score;       // 置信度
    AX_U64 track_id;    // 追蹤 ID (如果開啟 ByteTrack)
    char label[32];     // 類別名稱
    // 關鍵點信息（用於姿態檢測，如果沒有關鍵點則全部為0）
    AI_KEYPOINT_T keypoints[MAX_KEYPOINTS];
    AX_U32 nKeypoints;  // 實際關鍵點數量
} AI_OBJ_T;

// 定義檢測結果集合
typedef struct {
    AX_U32 nObjSize;    // 實際檢測到的物件數量
    AI_OBJ_T objects[MAX_DETECT_OBJ_NUM]; // 固定大小陣列，避免 STL 跨邊界問題
} AI_RESULT_T;

// AI 模型抽象介面
class IAIModel {
public:
    virtual ~IAIModel() {}
    
    // 初始化 
    virtual int Init(const char* model_path) = 0;
    
    // 獲取模型需要的輸入寬高 (用於配置 IVPS)
    virtual void GetInputSize(int* w, int* h) = 0;

    // 推論 (傳入 AX 視頻幀，輸出結果)
    virtual int Inference(const AX_VIDEO_FRAME_T* pFrame, AI_RESULT_T* pResult) = 0;

    // 反初始化
    virtual int Deinit() = 0;
};

// 定義導出函數類型 (用於 dlsym)
typedef IAIModel* (*CreateAIModelFunc)();
typedef void (*DestroyAIModelFunc)(IAIModel*);

#endif