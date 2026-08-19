#ifndef OSD_RENDERER_INTERFACE_H
#define OSD_RENDERER_INTERFACE_H

// RK3588 版 OSD 渲染器接口。与 AX650 的 include/osd_renderer_interface.h 同构：
// IOSDRenderer / DefaultOSDRenderer 类名与语义一致；仅输出目标不同——
// AX650 绘制到 IVPS OSD 显示结构（AX_IVPS_RGN_DISP_GROUP_T），
// RK3588 直接绘制到待编码的 BGR 帧（cv::Mat）。

#include "ai_interface.h"
#include <opencv2/core.hpp>
#include <string>
#include <memory>

// OSD 渲染器抽象接口
// 每個模型可以實現自己的渲染邏輯（矩形框、骨架、文字等）
// 這樣添加新模型時不需要修改 video_stream_manager
class IOSDRenderer {
public:
    virtual ~IOSDRenderer() {}

    // 將 AI 檢測結果渲染到輸出幀
    // 參數：
    //   result: AI 檢測結果
    //   dstW, dstH: 目標分辨率（輸出幀分辨率）
    //   frame: 待繪製的 BGR 輸出幀（繪製後直接送編碼器）
    //   maxElements: 最大顯示元素數量
    // 返回：
    //   實際繪製的目標數量
    virtual unsigned int render(const AI_RESULT_T* result,
                                int dstW, int dstH,
                                cv::Mat& frame,
                                unsigned int maxElements) = 0;

    // 獲取渲染器名稱（用於調試）
    virtual std::string getName() const = 0;
};

// 默認的簡單 OSD 渲染器（只繪製矩形框和文字）
// 適用於大多數檢測模型
class DefaultOSDRenderer : public IOSDRenderer {
public:
    DefaultOSDRenderer() = default;
    virtual ~DefaultOSDRenderer() = default;

    unsigned int render(const AI_RESULT_T* result,
                        int dstW, int dstH,
                        cv::Mat& frame,
                        unsigned int maxElements) override;

    std::string getName() const override { return "DefaultOSDRenderer"; }
};

#endif // OSD_RENDERER_INTERFACE_H
