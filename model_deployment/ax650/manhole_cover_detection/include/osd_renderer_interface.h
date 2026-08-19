#ifndef OSD_RENDERER_INTERFACE_H
#define OSD_RENDERER_INTERFACE_H

#include "ax_ivps_api.h"
#include "ai_interface.h"
#include <string>
#include <vector>
#include <memory> 

#ifndef AX_IVPS_REGION_MAX_DISP_NUM
#define AX_IVPS_REGION_MAX_DISP_NUM (32)
#endif

// OSD 渲染器抽象接口
// 每個模型可以實現自己的渲染邏輯（矩形框、骨架、文字等）
// 這樣添加新模型時不需要修改 video_stream_manager
class IOSDRenderer {
public:
    virtual ~IOSDRenderer() {}
    
    // 將 AI 檢測結果渲染到 OSD 顯示結構中
    // 參數：
    //   result: AI 檢測結果
    //   dstW, dstH: 目標分辨率（主碼流分辨率）
    //   tDisp: IVPS OSD 顯示結構（輸出，需要先初始化 tChnAttr）
    //   maxElements: 最大顯示元素數量（通常為 AX_IVPS_REGION_MAX_DISP_NUM）
    // 返回：
    //   實際使用的顯示元素數量（設置到 tDisp.nNum）
    virtual unsigned int render(const AI_RESULT_T* result, 
                                int dstW, int dstH,
                                AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                                unsigned int maxElements) = 0;
    
    // 獲取渲染器名稱（用於調試）
    virtual std::string getName() const = 0;
};

// 默認的簡單 OSD 渲染器（只繪製矩形框和文字）
// 適用於大多數檢測模型（如頭盔檢測、人體檢測等）
class DefaultOSDRenderer : public IOSDRenderer {
public:
    DefaultOSDRenderer() : bitmapPtrs_(), previousBatchSize_(0), lastBatchSize_(0) {}
    virtual ~DefaultOSDRenderer() {
        bitmapPtrs_.clear();
    }
    
    unsigned int render(const AI_RESULT_T* result, 
                       int dstW, int dstH,
                       AX_IVPS_RGN_DISP_GROUP_T* tDisp,
                       unsigned int maxElements) override;
    
    std::string getName() const override { return "DefaultOSDRenderer"; }

private:
    std::vector<std::unique_ptr<unsigned char[]>> bitmapPtrs_;
    // 延遲釋放：IVPS 會非同步使用 pBitmap，僅在「兩幀前」的批次可安全釋放，避免文字亂碼/洋紅線
    size_t previousBatchSize_;
    size_t lastBatchSize_;
};

#endif // OSD_RENDERER_INTERFACE_H

