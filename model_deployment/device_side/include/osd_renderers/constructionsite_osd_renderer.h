#ifndef WORKSAFETY_OSD_RENDERER_H
#define WORKSAFETY_OSD_RENDERER_H

#include "../osd_renderer_interface.h"
#include <vector>
#include <cstring>

// 作業安全檢測專用 OSD 渲染器
// 繪製：矩形框 + 文字標籤（根據類別設定顏色）
class ConstructionsiteOSDRenderer : public IOSDRenderer {
public:
	ConstructionsiteOSDRenderer() : bitmapPtrs_() {}
	virtual ~ConstructionsiteOSDRenderer() {
		bitmapPtrs_.clear();
	}

	unsigned int render(const AI_RESULT_T* result,
						int dstW, int dstH,
						AX_IVPS_RGN_DISP_GROUP_T* tDisp,
						unsigned int maxElements) override;

	std::string getName() const override { return "ConstructionsiteOSDRenderer"; }

private:
	unsigned int getColorForClass(const char* label) const;

	std::vector<std::unique_ptr<unsigned char[]>> bitmapPtrs_;
	size_t previousBatchSize_ = 0;
	size_t lastBatchSize_ = 0;
};

#endif // WORKSAFETY_OSD_RENDERER_H
