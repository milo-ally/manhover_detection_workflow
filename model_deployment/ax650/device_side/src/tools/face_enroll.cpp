#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include "plugins/letterbox_utils.hpp"
extern "C" {
#include "common/common_sys.h"
}

#include <opencv2/opencv.hpp>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr int kFaceDetInputW = 640;
constexpr int kFaceDetInputH = 640;
constexpr int kArcInputW = 112;
constexpr int kArcInputH = 112;
constexpr int kEmbedSize = 512;
constexpr int kRegMax = 16;
constexpr float kFaceDetConfDefault = 0.35f;
constexpr float kFaceDetNmsDefault = 0.45f;
constexpr int kFaceDetClassNum = 1;
}

struct Args {
    std::string imagePath;
    std::string personName;
    std::string faceDetModel;
    std::string arcfaceModel;
    std::string landmarkDat;
    std::string dbTxt;
};

struct Rect {
    float x, y, w, h;
};
struct Obj {
    Rect bbox;
    float prob;
};
struct HeadData {
    float* box;
    float* cls;
    bool isCombined;
    bool isNchw;
    int H;
    int W;
};

static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](std::string& out) -> bool {
            if (i + 1 >= argc) return false;
            out = argv[++i];
            return true;
        };
        if (k == "--image") {
            if (!next(args.imagePath)) return false;
        } else if (k == "--name") {
            if (!next(args.personName)) return false;
        } else if (k == "--face-det-model") {
            if (!next(args.faceDetModel)) return false;
        } else if (k == "--arcface-model") {
            if (!next(args.arcfaceModel)) return false;
        } else if (k == "--landmark-dat") {
            if (!next(args.landmarkDat)) return false;
        } else if (k == "--db-txt") {
            if (!next(args.dbTxt)) return false;
        } else {
            return false;
        }
    }
    return !args.imagePath.empty() && !args.personName.empty() && !args.faceDetModel.empty() &&
           !args.arcfaceModel.empty() && !args.landmarkDat.empty() && !args.dbTxt.empty();
}

static std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    if (!file.read(buf.data(), size)) return {};
    return buf;
}

static void l2Normalize(std::vector<float>& v) {
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x) * static_cast<double>(x);
    if (sum <= 1e-12) return;
    float inv = 1.0f / static_cast<float>(std::sqrt(sum));
    for (float& x : v) x *= inv;
}

static void sortByProb(std::vector<Obj>& objs) {
    std::sort(objs.begin(), objs.end(), [](const Obj& a, const Obj& b) { return a.prob > b.prob; });
}

static void nms(const std::vector<Obj>& objs, std::vector<int>& picked, float thr) {
    picked.clear();
    const int n = static_cast<int>(objs.size());
    std::vector<float> areas(n);
    for (int i = 0; i < n; ++i) areas[i] = objs[i].bbox.w * objs[i].bbox.h;
    for (int i = 0; i < n; ++i) {
        int keep = 1;
        for (int j = 0; j < static_cast<int>(picked.size()); ++j) {
            const Obj& a = objs[i];
            const Obj& b = objs[picked[j]];
            float inter_x1 = std::max(a.bbox.x, b.bbox.x);
            float inter_y1 = std::max(a.bbox.y, b.bbox.y);
            float inter_x2 = std::min(a.bbox.x + a.bbox.w, b.bbox.x + b.bbox.w);
            float inter_y2 = std::min(a.bbox.y + a.bbox.h, b.bbox.y + b.bbox.h);
            float inter = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
            float uni = areas[i] + areas[picked[j]] - inter;
            if (uni > 0.f && inter / uni > thr) {
                keep = 0;
                break;
            }
        }
        if (keep) picked.push_back(i);
    }
}

static void generateFaceProposals(AX_ENGINE_IO_INFO_T* ioInfo, AX_ENGINE_IO_T& ioData, std::vector<Obj>& proposals, float confThresh) {
    std::map<int, HeadData> heads;
    for (unsigned int i = 0; i < ioInfo->nOutputSize; ++i) {
        auto& out = ioInfo->pOutputs[i];
        float* data = reinterpret_cast<float*>(ioData.pOutputs[i].pVirAddr);
        if (!data) continue;

        int C_nhwc = out.pShape[3];
        int H_nhwc = out.pShape[1];
        int W_nhwc = out.pShape[2];
        int C_nchw = out.pShape[1];
        int H_nchw = out.pShape[2];
        int W_nchw = out.pShape[3];
        bool is_nhwc = (C_nhwc == 4 * kRegMax + kFaceDetClassNum) || (C_nhwc == 4 * kRegMax) || (C_nhwc == kFaceDetClassNum);
        bool is_nchw = (C_nchw == 4 * kRegMax + kFaceDetClassNum) || (C_nchw == 4 * kRegMax) || (C_nchw == kFaceDetClassNum);
        if (!is_nhwc && !is_nchw) continue;
        bool use_nchw = (!is_nhwc && is_nchw);
        int H = use_nchw ? H_nchw : H_nhwc;
        int W = use_nchw ? W_nchw : W_nhwc;
        int C = use_nchw ? C_nchw : C_nhwc;
        if (H <= 0 || W <= 0) continue;

        if (heads.find(H) == heads.end()) heads[H] = {nullptr, nullptr, false, use_nchw, H, W};
        heads[H].isNchw = use_nchw;
        heads[H].W = W;
        if (C == 4 * kRegMax + kFaceDetClassNum) {
            heads[H].box = data;
            heads[H].cls = data;
            heads[H].isCombined = true;
        } else if (C == 4 * kRegMax) {
            heads[H].box = data;
        } else if (C == kFaceDetClassNum) {
            heads[H].cls = data;
        }
    }

    for (auto& kv : heads) {
        int H = kv.first;
        HeadData& head = kv.second;
        if (!head.box || !head.cls) continue;

        int W = head.W;
        int stride = kFaceDetInputW / H;
        int cCombined = 4 * kRegMax + kFaceDetClassNum;
        int cBox = 4 * kRegMax;
        auto at_nhwc = [&](const float* base, int h, int w, int c, int C) -> float {
            return base[(h * W + w) * C + c];
        };
        auto at_nchw = [&](const float* base, int h, int w, int c) -> float {
            return base[(c * H + h) * W + w];
        };

        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float clsRaw = 0.0f;
                if (head.isCombined) {
                    clsRaw = head.isNchw ? at_nchw(head.cls, h, w, cBox) : at_nhwc(head.cls, h, w, cBox, cCombined);
                } else {
                    clsRaw = head.isNchw ? at_nchw(head.cls, h, w, 0) : at_nhwc(head.cls, h, w, 0, kFaceDetClassNum);
                }
                float score = sigmoid(clsRaw);
                if (score < confThresh) continue;

                float pred[4] = {0};
                for (int k = 0; k < 4; ++k) {
                    float expSum = 0.f, weighted = 0.f;
                    for (int r = 0; r < kRegMax; ++r) {
                        int c = k * kRegMax + r;
                        float raw = 0.0f;
                        if (head.isCombined) {
                            raw = head.isNchw ? at_nchw(head.box, h, w, c) : at_nhwc(head.box, h, w, c, cCombined);
                        } else {
                            raw = head.isNchw ? at_nchw(head.box, h, w, c) : at_nhwc(head.box, h, w, c, cBox);
                        }
                        float e = std::exp(raw);
                        expSum += e;
                        weighted += e * r;
                    }
                    pred[k] = weighted / expSum;
                }

                float cx = (w + 0.5f) * stride;
                float cy = (h + 0.5f) * stride;
                float x1 = cx - pred[0] * stride;
                float y1 = cy - pred[1] * stride;
                float x2 = cx + pred[2] * stride;
                float y2 = cy + pred[3] * stride;
                proposals.push_back({{x1, y1, x2 - x1, y2 - y1}, score});
            }
        }
    }
}

static bool detectLargestFace(
    AX_ENGINE_HANDLE detHandle,
    AX_ENGINE_IO_INFO_T* detInfo,
    AX_ENGINE_IO_T& detIo,
    const cv::Mat& bgr,
    cv::Rect& outRect,
    float confThresh,
    float nmsThresh) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat inputRgb;
    ai_letterbox::LetterboxInfo lb = ai_letterbox::letterbox(rgb, inputRgb, kFaceDetInputW, kFaceDetInputH);
    memcpy(detIo.pInputs[0].pVirAddr, inputRgb.data, kFaceDetInputW * kFaceDetInputH * 3);
    AX_SYS_MflushCache(detIo.pInputs[0].phyAddr, detIo.pInputs[0].pVirAddr, detIo.pInputs[0].nSize);
    if (AX_ENGINE_RunSync(detHandle, &detIo) != 0) return false;

    std::vector<Obj> props;
    generateFaceProposals(detInfo, detIo, props, confThresh);
    if (props.empty()) return false;
    sortByProb(props);
    std::vector<int> picked;
    nms(props, picked, nmsThresh);
    if (picked.empty()) return false;

    int bestIdx = picked[0];
    float bestArea = -1.f;
    for (int idx : picked) {
        const Obj& o = props[idx];
        float area = o.bbox.w * o.bbox.h;
        if (area > bestArea) {
            bestArea = area;
            bestIdx = idx;
        }
    }

    Obj best = props[bestIdx];
    ai_letterbox::scale_bbox_to_original(best.bbox.x, best.bbox.y, best.bbox.w, best.bbox.h, lb);
    int x = std::max(0, static_cast<int>(best.bbox.x));
    int y = std::max(0, static_cast<int>(best.bbox.y));
    int w = static_cast<int>(best.bbox.w);
    int h = static_cast<int>(best.bbox.h);
    if (w <= 0 || h <= 0) return false;
    if (x + w > bgr.cols) w = bgr.cols - x;
    if (y + h > bgr.rows) h = bgr.rows - y;
    if (w <= 0 || h <= 0) return false;
    outRect = cv::Rect(x, y, w, h);
    return true;
}

static bool alignFaceWithRect(const cv::Mat& bgr, const cv::Rect& faceRect, const dlib::shape_predictor& predictor, cv::Mat& outRgb) {
    dlib::cv_image<dlib::bgr_pixel> cimg(bgr);
    dlib::rectangle rect(faceRect.x, faceRect.y, faceRect.x + faceRect.width - 1, faceRect.y + faceRect.height - 1);
    dlib::full_object_detection shape = predictor(cimg, rect);
    if (shape.num_parts() < 68) return false;

    auto avgPt = [&](int i, int j) -> cv::Point2f {
        return cv::Point2f((shape.part(i).x() + shape.part(j).x()) * 0.5f, (shape.part(i).y() + shape.part(j).y()) * 0.5f);
    };

    std::vector<cv::Point2f> src = {
        avgPt(36, 39), avgPt(42, 45), cv::Point2f(shape.part(30).x(), shape.part(30).y()),
        cv::Point2f(shape.part(48).x(), shape.part(48).y()), cv::Point2f(shape.part(54).x(), shape.part(54).y())
    };
    static const std::vector<cv::Point2f> dst = {
        cv::Point2f(38.2946f, 51.6963f), cv::Point2f(73.5318f, 51.5014f), cv::Point2f(56.0252f, 71.7366f),
        cv::Point2f(41.5493f, 92.3655f), cv::Point2f(70.7299f, 92.2041f)
    };

    cv::Mat M = cv::estimateAffinePartial2D(src, dst, cv::noArray(), cv::RANSAC);
    if (M.empty()) return false;
    cv::Mat aligned;
    cv::warpAffine(bgr, aligned, M, cv::Size(kArcInputW, kArcInputH), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::cvtColor(aligned, outRgb, cv::COLOR_BGR2RGB);
    return true;
}

static bool updateDbTxt(const std::string& path, const std::string& name, const std::vector<float>& emb) {
    std::vector<std::string> lines;
    {
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string n;
            if (!(iss >> n)) continue;
            if (n == name) continue;
            lines.push_back(line);
        }
    }
    std::ostringstream oss;
    oss << name;
    for (float x : emb) oss << " " << x;
    lines.push_back(oss.str());
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) return false;
    for (const auto& l : lines) out << l << "\n";
    return true;
}

static void freeIo(AX_ENGINE_IO_T& io) {
    if (io.pInputs) {
        for (unsigned int i = 0; i < io.nInputSize; ++i) {
            AX_SYS_MemFree(io.pInputs[i].phyAddr, io.pInputs[i].pVirAddr);
            delete[] io.pInputs[i].pStride;
        }
        delete[] io.pInputs;
        io.pInputs = nullptr;
    }
    if (io.pOutputs) {
        for (unsigned int i = 0; i < io.nOutputSize; ++i) {
            AX_SYS_MemFree(io.pOutputs[i].phyAddr, io.pOutputs[i].pVirAddr);
            delete[] io.pOutputs[i].pStride;
        }
        delete[] io.pOutputs;
        io.pOutputs = nullptr;
    }
}

static bool allocIo(AX_ENGINE_IO_INFO_T* info, AX_ENGINE_IO_T& io, unsigned int inSizeOverride = 0) {
    io.pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
    io.nInputSize = info->nInputSize;
    io.pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
    io.nOutputSize = info->nOutputSize;
    for (unsigned int i = 0; i < info->nInputSize; ++i) {
        memset(&io.pInputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));
        AX_U32 size = inSizeOverride ? inSizeOverride : info->pInputs[i].nSize;
        AX_U64 phy = 0;
        AX_VOID* vir = NULL;
        if (AX_SYS_MemAlloc(&phy, &vir, size, 128, (const AX_S8*)"face_enroll_in") != 0) return false;
        io.pInputs[i].nSize = size;
        io.pInputs[i].phyAddr = phy;
        io.pInputs[i].pVirAddr = vir;
        io.pInputs[i].pStride = new AX_S32[4];
        io.pInputs[i].pStride[0] = (inSizeOverride == 0) ? info->pInputs[i].pShape[2] * 3 : ((inSizeOverride == (kFaceDetInputW * kFaceDetInputH * 3)) ? (kFaceDetInputW * 3) : (kArcInputW * 3));
    }
    for (unsigned int i = 0; i < info->nOutputSize; ++i) {
        memset(&io.pOutputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));
        AX_U64 phy = 0;
        AX_VOID* vir = NULL;
        if (AX_SYS_MemAlloc(&phy, &vir, info->pOutputs[i].nSize, 128, (const AX_S8*)"face_enroll_out") != 0) return false;
        io.pOutputs[i].nSize = info->pOutputs[i].nSize;
        io.pOutputs[i].phyAddr = phy;
        io.pOutputs[i].pVirAddr = vir;
        io.pOutputs[i].pStride = new AX_S32[4];
    }
    return true;
}

int main(int argc, char** argv) {
    float detConf = kFaceDetConfDefault;
    float detNms = kFaceDetNmsDefault;
    if (const char* c = std::getenv("FACE_DET_CONF")) detConf = static_cast<float>(atof(c));
    if (const char* n = std::getenv("FACE_DET_NMS")) detNms = static_cast<float>(atof(n));
    if (detConf < 0.01f) detConf = 0.01f;
    if (detConf > 0.99f) detConf = 0.99f;
    if (detNms < 0.01f) detNms = 0.01f;
    if (detNms > 0.99f) detNms = 0.99f;
    std::cerr << "[face_enroll] detector thresholds: conf=" << detConf << ", nms=" << detNms << "\n";
    Args args;
    if (!parseArgs(argc, argv, args)) {
        std::cerr << "Usage: face_enroll --image <jpg> --name <id> --face-det-model <axmodel> --arcface-model <axmodel> --landmark-dat <dat> --db-txt <txt>\n";
        return 2;
    }

    cv::Mat bgr = cv::imread(args.imagePath, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "failed to read image\n";
        return 3;
    }

    dlib::shape_predictor predictor;
    try {
        dlib::deserialize(args.landmarkDat) >> predictor;
    } catch (...) {
        std::cerr << "failed to load landmark dat\n";
        return 4;
    }

    COMMON_SYS_POOL_CFG_T poolcfg[] = {{640, 640, 2048, AX_FORMAT_YUV420_SEMIPLANAR, 2}};
    COMMON_SYS_ARGS_T sysArgs = {0};
    sysArgs.nPoolCfgCnt = 1;
    sysArgs.pPoolCfg = poolcfg;
    if (COMMON_SYS_Init(&sysArgs) != 0) return 5;
    AX_ENGINE_NPU_ATTR_T npuAttr;
    memset(&npuAttr, 0, sizeof(npuAttr));
    npuAttr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (AX_ENGINE_Init(&npuAttr) != 0) {
        COMMON_SYS_DeInit();
        return 6;
    }

    int code = 0;
    AX_ENGINE_HANDLE detHandle = nullptr;
    AX_ENGINE_HANDLE arcHandle = nullptr;
    AX_ENGINE_IO_T detIo = {0};
    AX_ENGINE_IO_T arcIo = {0};
    do {
        std::vector<char> detModel = readFile(args.faceDetModel);
        std::vector<char> arcModel = readFile(args.arcfaceModel);
        if (detModel.empty() || arcModel.empty()) {
            code = 7;
            break;
        }
        if (AX_ENGINE_CreateHandle(&detHandle, detModel.data(), detModel.size()) != 0 || !detHandle) {
            code = 8;
            break;
        }
        if (AX_ENGINE_CreateHandle(&arcHandle, arcModel.data(), arcModel.size()) != 0 || !arcHandle) {
            code = 9;
            break;
        }
        AX_ENGINE_IO_INFO_T* detInfo = nullptr;
        AX_ENGINE_IO_INFO_T* arcInfo = nullptr;
        AX_ENGINE_GetIOInfo(detHandle, &detInfo);
        AX_ENGINE_GetIOInfo(arcHandle, &arcInfo);
        if (!detInfo || !arcInfo) {
            code = 10;
            break;
        }
        if (!allocIo(detInfo, detIo, kFaceDetInputW * kFaceDetInputH * 3)) {
            code = 11;
            break;
        }
        if (!allocIo(arcInfo, arcIo, kArcInputW * kArcInputH * 3)) {
            code = 12;
            break;
        }

        cv::Rect faceRect;
        if (!detectLargestFace(detHandle, detInfo, detIo, bgr, faceRect, detConf, detNms)) {
            std::cerr << "[face_enroll] detectLargestFace failed\n";
            code = 13;
            break;
        }

        cv::Mat alignedRgb;
        if (!alignFaceWithRect(bgr, faceRect, predictor, alignedRgb)) {
            code = 14;
            break;
        }

        memcpy(arcIo.pInputs[0].pVirAddr, alignedRgb.data, kArcInputW * kArcInputH * 3);
        AX_SYS_MflushCache(arcIo.pInputs[0].phyAddr, arcIo.pInputs[0].pVirAddr, arcIo.pInputs[0].nSize);
        if (AX_ENGINE_RunSync(arcHandle, &arcIo) != 0) {
            code = 15;
            break;
        }

        size_t cnt = arcIo.pOutputs[0].nSize / sizeof(float);
        cnt = std::min(cnt, static_cast<size_t>(kEmbedSize));
        const float* ptr = reinterpret_cast<const float*>(arcIo.pOutputs[0].pVirAddr);
        std::vector<float> emb(ptr, ptr + cnt);
        l2Normalize(emb);
        if (!updateDbTxt(args.dbTxt, args.personName, emb)) {
            code = 16;
            break;
        }
        std::cout << "{\"ok\":true,\"name\":\"" << args.personName << "\",\"dim\":" << emb.size() << "}\n";
    } while (0);

    freeIo(detIo);
    freeIo(arcIo);
    if (detHandle) AX_ENGINE_DestroyHandle(detHandle);
    if (arcHandle) AX_ENGINE_DestroyHandle(arcHandle);
    AX_ENGINE_Deinit();
    COMMON_SYS_DeInit();
    return code;
}

