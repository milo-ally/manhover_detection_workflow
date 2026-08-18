#ifndef OSD_RENDERER_H
#define OSD_RENDERER_H

#include <mutex>
#include "ax_ivps_api.h"
#include "../ai_interface.h"
#include "../../common/common_pipeline/common_pipeline.h"

#ifndef AX_IVPS_INVALID_REGION_HANDLE
#define AX_IVPS_INVALID_REGION_HANDLE ((IVPS_RGN_HANDLE)-1)
#endif
#ifndef AX_IVPS_REGION_MAX_DISP_NUM
#define AX_IVPS_REGION_MAX_DISP_NUM (32)
#endif

class OSDRenderer {
public:
    // 參考 ax650_o：使用 pipeline 創建的 OSD Region
    explicit OSDRenderer(pipeline_t* pipeline);
    ~OSDRenderer();
    
    bool init();
    void update(const AI_RESULT_T* result, int srcW, int srcH, int dstW, int dstH);
    void clear();
    
private:
    pipeline_t* pipeline_;
    int ivpsGroup_;
    std::mutex renderMutex_;
    bool regionInitialized_ = false;
    
    // 獲取 pipeline 創建的 OSD Region handle
    IVPS_RGN_HANDLE getRegionHandle() const;
};

#endif // OSD_RENDERER_H

