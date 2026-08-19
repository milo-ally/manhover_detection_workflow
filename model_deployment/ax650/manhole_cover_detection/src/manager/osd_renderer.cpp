#include "osd_renderer.h"
#include "../../utilities/sample_log.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <map>

// #region agent log
// Debug logging helper function
static void debug_log(const std::string& location, const std::string& message, const std::map<std::string, std::string>& data = {}) {
    std::ofstream log_file("c:\\Users\\ly0248\\Desktop\\20251231\\.cursor\\debug.log", std::ios::app);
    if (!log_file.is_open()) return;
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();
    
    log_file << "{\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()
             << ",\"location\":\"" << location << "\",\"message\":\"" << message << "\"";
    
    for (const auto& pair : data) {
        log_file << ",\"" << pair.first << "\":\"" << pair.second << "\"";
    }
    
    log_file << "}\n";
    log_file.close();
}
// #endregion

#include "../common/common_pipeline/common_pipeline.h"

// 构造函数，初始化 pipeline 指針
OSDRenderer::OSDRenderer(pipeline_t* pipeline) : pipeline_(pipeline) {
    if (pipeline_) {
        ivpsGroup_ = pipeline_->m_ivps_attr.n_ivps_grp;
    }
}

// 析构函数，释放区域资源
// 參考 ax650_o：OSD Region 由 pipeline 創建和銷毀，這裡不需要手動銷毀
OSDRenderer::~OSDRenderer() {
    std::lock_guard<std::mutex> lock(renderMutex_);
    if (regionInitialized_) {
        // IVPS_RGN_HANDLE handle = getRegionHandle();
        // if (handle != AX_IVPS_INVALID_REGION_HANDLE) {
        //     AX_S32 ret = AX_IVPS_RGN_Destroy(handle);
        //     if (ret != 0) {
        //         ALOGW("[OSDRenderer] AX_IVPS_RGN_Destroy failed: %d", ret);
        //     }
        // }
        regionInitialized_ = false;
    }
}

// 獲取 pipeline 創建的 OSD Region handle
IVPS_RGN_HANDLE OSDRenderer::getRegionHandle() const {
    if (!pipeline_ || pipeline_->m_ivps_attr.n_osd_rgn == 0) {
        return AX_IVPS_INVALID_REGION_HANDLE;
    }
    // 參考 ax650_o：使用 pipeline 創建的第一個 OSD Region
    return pipeline_->m_ivps_attr.n_osd_rgn_chn[0];
}

// 初始化OSD区域
// 參考 ax650_o：OSD Region 已經由 pipeline 創建，這裡只需要檢查是否有效
bool OSDRenderer::init() {
    std::lock_guard<std::mutex> lock(renderMutex_);
    if (!pipeline_) {
        ALOGE("[OSDRenderer] pipeline is null");
        return false;
    }

    IVPS_RGN_HANDLE handle = getRegionHandle();
    if (handle == AX_IVPS_INVALID_REGION_HANDLE) {
        ALOGE("[OSDRenderer] invalid region handle");
        return false;
    }

    regionInitialized_ = true;
    return true;
}

// 根据AI检测结果更新OSD显示
void OSDRenderer::update(const AI_RESULT_T* pResult, int srcW, int srcH, int dstW, int dstH) {
    if (!pResult) return;
    
    // 參考 ax650_o：使用 pipeline 創建的 OSD Region handle
    IVPS_RGN_HANDLE regionHandle = getRegionHandle();
    if (regionHandle == AX_IVPS_INVALID_REGION_HANDLE) return;
    
    std::lock_guard<std::mutex> lock(renderMutex_); // 加锁，保证线程安全
    
    AX_IVPS_RGN_DISP_GROUP_T tDisp = {0}; // 显示组结构体初始化（memset 後所有字段為 0）
    // 參考 ax_osd_drawer.hpp 的 get() 函數：正確設置 tChnAttr
    // 注意：對於 RECT 類型，tChnAttr 的某些字段可能不需要設置，但為了與 ax650_o 保持一致，我們設置所有字段
    tDisp.tChnAttr.nZindex = 0;           // 參考 ax_osd_drawer：使用 region index
    tDisp.tChnAttr.bSingleCanvas = AX_FALSE;
    tDisp.tChnAttr.nAlpha = 255;          // 透明度
    tDisp.tChnAttr.eFormat = AX_FORMAT_RGBA8888;  // 參考 ax_osd_drawer
    tDisp.tChnAttr.nBitColor.bColorInvEn = AX_FALSE;  // 參考 ax_osd_helper.hpp：對於 RECT 類型，不需要啟用 BitColor
    tDisp.tChnAttr.nBitColor.nColor = 0xFF0000;   // 參考 ax_osd_drawer
    tDisp.tChnAttr.nBitColor.nColorInv = 0xFF;    // 參考 ax_osd_drawer
    tDisp.tChnAttr.nBitColor.nColorInvThr = 0xA0A0A0;  // 參考 ax_osd_drawer
    // nColorKey 在 memset 後為 0，對於 RECT 類型可能不需要設置
    
    tDisp.nNum = pResult->nObjSize;       // 目标数量
    if (tDisp.nNum > AX_IVPS_REGION_MAX_DISP_NUM) tDisp.nNum = AX_IVPS_REGION_MAX_DISP_NUM;

    float scale_x = (float)dstW / srcW;
    float scale_y = (float)dstH / srcH;

    for (unsigned int i = 0; i < tDisp.nNum; ++i) {
        // 坐标变换，将归一化坐标转换为目标分辨率坐标
        int x = (int)(pResult->objects[i].x * dstW);
        int y = (int)(pResult->objects[i].y * dstH);
        int w = (int)(pResult->objects[i].w * dstW);
        int h = (int)(pResult->objects[i].h * dstH);
        
        // 边界检查和调整
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + w >= dstW) w = dstW - x - 2;
        if (y + h >= dstH) h = dstH - y - 2;
        if (w % 2 != 0) w--; if (h % 2 != 0) h--;
        
        if (w <= 4 || h <= 4) { tDisp.arrDisp[i].bShow = AX_FALSE; continue; }
        
        // 配置显示参数，绘制矩形框
        // 參考 ax_osd_drawer.hpp 的 add_rect 函數：只設置必要的字段
        tDisp.arrDisp[i].bShow = AX_TRUE;
        tDisp.arrDisp[i].eType = AX_IVPS_RGN_TYPE_RECT;
        tDisp.arrDisp[i].uDisp.tPolygon.tRect.nX = x;
        tDisp.arrDisp[i].uDisp.tPolygon.tRect.nY = y;
        tDisp.arrDisp[i].uDisp.tPolygon.tRect.nW = w;
        tDisp.arrDisp[i].uDisp.tPolygon.tRect.nH = h;
        tDisp.arrDisp[i].uDisp.tPolygon.nLineWidth = 4;      // 边框宽度
        tDisp.arrDisp[i].uDisp.tPolygon.nColor = 0x00FF00;   // 颜色（绿色，RGB格式：0xRRGGBB）
        tDisp.arrDisp[i].uDisp.tPolygon.nAlpha = 255;        // 不透明
        // 注意：參考 ax_osd_drawer，不設置 bSolid 和 nPointNum（這些字段在 memset 後為 0，對於 RECT 類型可能不需要）
    }
    
    // 更新区域显示
    // 參考 ax650_o：使用 pipeline 創建的 OSD Region handle
    int ret = AX_IVPS_RGN_Update(regionHandle, &tDisp);
    if (ret != 0) {
        static int error_count = 0;
        if (++error_count % 100 == 0) {
            ALOGE("OSDRenderer: AX_IVPS_RGN_Update failed, ret=0x%x, handle=%d", ret, regionHandle);
        }
    }
}

// 清除OSD显示
void OSDRenderer::clear() {
    AI_RESULT_T emptyResult = {0};
    emptyResult.nObjSize = 0;
    update(&emptyResult, 640, 640, 1920, 1080);
}

