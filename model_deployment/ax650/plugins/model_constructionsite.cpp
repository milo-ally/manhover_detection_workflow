#include "ai_interface.h"
#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "letterbox_utils.hpp"

#define CLASS_NUM 8
#define REG_MAX 16
#define CONF_THRESH 0.45f
#define NMS_THRESH 0.45f

static float read_env_float(const char* key, float def) {
	const char* v = std::getenv(key);
	if (!v || !*v) return def;
	return static_cast<float>(atof(v));
}

static const char* CLASS_NAMES[] = {
	"person",
	"fire",
	"hat",
	"helmet",
	"vest",
	"safetyharness",
	"machiney",
	"smoke"
};

static inline float sigmoid(float x) {
	return 1.0f / (1.0f + expf(-x));
}

static std::vector<char> read_model_file(const char* filename) {
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		printf("[Error] Cannot open model file: %s\n", filename);
		return {};
	}
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);
	std::vector<char> buffer(size);
	if (!file.read(buffer.data(), size)) {
		printf("[Error] Failed to read model file: %s\n", filename);
		return {};
	}
	return buffer;
}

class Yolo11Constructionsite : public IAIModel {
public:
	int Init(const char* model_path) override {
		printf("[Yolo11Constructionsite] Loading model: %s\n", model_path);
		confThresh_ = read_env_float("CONSTRUCTION_CONF_THRESH", read_env_float("MODEL_CONF_THRESH", CONF_THRESH));
		nmsThresh_ = read_env_float("CONSTRUCTION_NMS_THRESH", read_env_float("MODEL_NMS_THRESH", NMS_THRESH));
		printf("[Constructionsite] thresholds: conf=%.3f nms=%.3f\n", confThresh_, nmsThresh_);
		std::vector<char> model_buffer = read_model_file(model_path);
		if (model_buffer.empty()) return -1;

		AX_ENGINE_NPU_ATTR_T npu_attr;
		memset(&npu_attr, 0, sizeof(npu_attr));
		npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;

		int ret = AX_ENGINE_CreateHandle(&m_handle, model_buffer.data(), model_buffer.size());
		if (ret != 0) {
			printf("[Error] CreateHandle failed: 0x%x\n", ret);
			return -1;
		}

		AX_ENGINE_GetIOInfo(m_handle, &m_io_info);

		m_io_data.pInputs = new AX_ENGINE_IO_BUFFER_T[m_io_info->nInputSize];
		m_io_data.nInputSize = m_io_info->nInputSize;
		m_io_data.pOutputs = new AX_ENGINE_IO_BUFFER_T[m_io_info->nOutputSize];
		m_io_data.nOutputSize = m_io_info->nOutputSize;

		for (unsigned int i = 0; i < m_io_info->nInputSize; ++i) {
			memset(&m_io_data.pInputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));
			auto& info = m_io_info->pInputs[i];
			AX_U32 size = info.nSize > 0 ? info.nSize : 640 * 640 * 3;
			AX_U64 phy = 0;
			AX_VOID* vir = NULL;
			AX_SYS_MemAlloc(&phy, &vir, size, 128, (const AX_S8*)"ax_input_bgr");

			m_io_data.pInputs[i].nSize = size;
			m_io_data.pInputs[i].phyAddr = phy;
			m_io_data.pInputs[i].pVirAddr = vir;

			m_io_data.pInputs[i].pStride = new AX_S32[4];
			memset(m_io_data.pInputs[i].pStride, 0, sizeof(AX_S32) * 4);
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
			memset(m_io_data.pOutputs[i].pStride, 0, sizeof(AX_S32) * 4);
		}

		if (m_io_info->nInputSize > 0) {
			auto& in0 = m_io_info->pInputs[0];
			int s1 = in0.pShape[1];
			int s2 = in0.pShape[2];
			int s3 = in0.pShape[3];
			if (s1 == 3) {
				m_input_h = s2;
				m_input_w = s3;
				m_input_nchw = true;
			} else if (s3 == 3) {
				m_input_h = s1;
				m_input_w = s2;
				m_input_nchw = false;
			} else {
				m_input_w = 640;
				m_input_h = 640;
				m_input_nchw = false;
			}
		}

		if (m_io_data.pInputs && m_io_info->nInputSize > 0) {
			int stride0 = m_input_nchw ? m_input_w : (m_input_w * 3);
			m_io_data.pInputs[0].pStride[0] = stride0;
		}
		return 0;
	}

	void GetInputSize(int* w, int* h) override { *w = m_input_w; *h = m_input_h; }

	int Inference(const AX_VIDEO_FRAME_T* pFrame, AI_RESULT_T* pResult) override {
		if (!m_handle || !pFrame || !pResult) return -1;

		cv::Mat nv12_mat(pFrame->u32Height * 3 / 2, pFrame->u32Width, CV_8UC1,
						 (void*)pFrame->u64VirAddr[0], pFrame->u32PicStride[0]);
		cv::Mat bgr_mat;
		cv::cvtColor(nv12_mat, bgr_mat, cv::COLOR_YUV2BGR_NV12);

		cv::Mat input_bgr;
		ai_letterbox::LetterboxInfo lb_info = ai_letterbox::letterbox(bgr_mat, input_bgr, m_input_w, m_input_h);

		cv::Mat input_rgb;
		cv::cvtColor(input_bgr, input_rgb, cv::COLOR_BGR2RGB);

		if (m_input_nchw) {
			pack_nchw(input_rgb, (unsigned char*)m_io_data.pInputs[0].pVirAddr, m_input_w, m_input_h);
		} else {
			memcpy(m_io_data.pInputs[0].pVirAddr, input_rgb.data, m_input_w * m_input_h * 3);
		}

		AX_SYS_MflushCache(m_io_data.pInputs[0].phyAddr, m_io_data.pInputs[0].pVirAddr, m_io_data.pInputs[0].nSize);

		int ret = AX_ENGINE_RunSync(m_handle, &m_io_data);
		if (ret != 0) return -1;

		std::vector<Object> proposals;
		generate_proposals(proposals);

		qsort_descent_inplace(proposals);
		std::vector<int> picked;
		nms_sorted_bboxes(proposals, picked, nmsThresh_);

		int src_w = pFrame->u32Width;
		int src_h = pFrame->u32Height;

		pResult->nObjSize = std::min((int)picked.size(), MAX_DETECT_OBJ_NUM);
		for (int i = 0; i < pResult->nObjSize; i++) {
			Object prop = proposals[picked[i]];
			ai_letterbox::scale_bbox_to_original(prop.bbox.x, prop.bbox.y, prop.bbox.w, prop.bbox.h, lb_info);

			prop.bbox.x = std::max(0.0f, std::min(prop.bbox.x, (float)src_w - 1));
			prop.bbox.y = std::max(0.0f, std::min(prop.bbox.y, (float)src_h - 1));
			prop.bbox.w = std::max(0.0f, std::min(prop.bbox.w, (float)src_w - prop.bbox.x));
			prop.bbox.h = std::max(0.0f, std::min(prop.bbox.h, (float)src_h - prop.bbox.y));

			pResult->objects[i].x = prop.bbox.x / src_w;
			pResult->objects[i].y = prop.bbox.y / src_h;
			pResult->objects[i].w = prop.bbox.w / src_w;
			pResult->objects[i].h = prop.bbox.h / src_h;
			pResult->objects[i].score = prop.prob;
			pResult->objects[i].class_id = prop.label;
			pResult->objects[i].track_id = 0;
			pResult->objects[i].nKeypoints = 0;

			if (prop.label >= 0 && prop.label < CLASS_NUM) {
				snprintf(pResult->objects[i].label, 32, "%s", CLASS_NAMES[prop.label]);
			} else {
				snprintf(pResult->objects[i].label, 32, "unknown");
			}
		}

		return 0;
	}

	int Deinit() override {
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
	AX_ENGINE_IO_T m_io_data = {0};
	int m_input_w = 640;
	int m_input_h = 640;
	float confThresh_ = CONF_THRESH;
	float nmsThresh_ = NMS_THRESH;
	bool m_input_nchw = false;

	struct Rect { float x, y, w, h; };
	struct Object { Rect bbox; int label; float prob; };

	void pack_nchw(const cv::Mat& img, unsigned char* dst, int w, int h) {
		const int channel_size = w * h;
		const unsigned char* src = img.data;
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				int idx = (y * w + x) * 3;
				dst[0 * channel_size + y * w + x] = src[idx + 0];
				dst[1 * channel_size + y * w + x] = src[idx + 1];
				dst[2 * channel_size + y * w + x] = src[idx + 2];
			}
		}
	}

	void generate_proposals(std::vector<Object>& proposals) {
		const int expected_c = REG_MAX * 4 + CLASS_NUM;

		for (unsigned int i = 0; i < m_io_info->nOutputSize; ++i) {
			auto& output = m_io_info->pOutputs[i];
			float* data = (float*)m_io_data.pOutputs[i].pVirAddr;
			if (!data) continue;

			int s1 = output.pShape[1];
			int s2 = output.pShape[2];
			int s3 = output.pShape[3];

			bool nhwc = (s3 == expected_c);
			bool nchw = (s1 == expected_c);
			int H = nhwc ? s1 : (nchw ? s2 : s1);
			int W = nhwc ? s2 : (nchw ? s3 : s2);
			int C = nhwc ? s3 : (nchw ? s1 : s3);

			int stride = m_input_w / H;
			std::vector<float> feat(C);

			for (int h = 0; h < H; h++) {
				for (int w = 0; w < W; w++) {
					const float* ptr = nullptr;
					if (nhwc) {
						ptr = data + (h * W + w) * C;
					} else {
						for (int c = 0; c < C; ++c) {
							feat[c] = data[(c * H + h) * W + w];
						}
						ptr = feat.data();
					}

					const float* cls_ptr = ptr + 4 * REG_MAX;

					int max_id = 0;
					float max_prob = -10000.0f;
					for (int c = 0; c < CLASS_NUM; c++) {
						if (cls_ptr[c] > max_prob) {
							max_prob = cls_ptr[c];
							max_id = c;
						}
					}

					float score = sigmoid(max_prob);
					if (score < confThresh_) continue;

					float pred_ltrb[4];
					for (int k = 0; k < 4; k++) {
						const float* dfl_ptr = ptr + k * REG_MAX;
						float max_val = dfl_ptr[0];
						for (int r = 1; r < REG_MAX; r++) {
							if (dfl_ptr[r] > max_val) max_val = dfl_ptr[r];
						}
						float exp_sum = 0.0f;
						float weighted_sum = 0.0f;
						for (int r = 0; r < REG_MAX; r++) {
							float e = expf(dfl_ptr[r] - max_val);
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
					obj.bbox = {x1, y1, x2 - x1, y2 - y1};
					obj.label = max_id;
					obj.prob = score;
					proposals.push_back(obj);
				}
			}
		}
	}

	void qsort_descent_inplace(std::vector<Object>& objects) {
		std::sort(objects.begin(), objects.end(),
				  [](const Object& a, const Object& b) { return a.prob > b.prob; });
	}

	void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold) {
		picked.clear();
		const int n = objects.size();
		std::vector<float> areas(n);
		for (int i = 0; i < n; i++) areas[i] = objects[i].bbox.w * objects[i].bbox.h;
		for (int i = 0; i < n; i++) {
			const Object& a = objects[i];
			int keep = 1;
			for (int j = 0; j < (int)picked.size(); j++) {
				const Object& b = objects[picked[j]];
				float inter_x1 = std::max(a.bbox.x, b.bbox.x);
				float inter_y1 = std::max(a.bbox.y, b.bbox.y);
				float inter_x2 = std::min(a.bbox.x + a.bbox.w, b.bbox.x + b.bbox.w);
				float inter_y2 = std::min(a.bbox.y + a.bbox.h, b.bbox.y + b.bbox.h);
				float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
				float union_area = areas[i] + areas[picked[j]] - inter_area;
				if (inter_area / union_area > nms_threshold) { keep = 0; break; }
			}
			if (keep) picked.push_back(i);
		}
	}
};

extern "C" {
	IAIModel* CreateAIModel() { return new Yolo11Constructionsite(); }
	void DestroyAIModel(IAIModel* p) { delete p; }
}
