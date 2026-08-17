#include "ai_interface.h"
#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include "c_api.h"
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#if __has_include(<dlib/image_processing.h>) && __has_include(<dlib/opencv.h>)
#define HAS_DLIB_LANDMARK 1
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#else
#define HAS_DLIB_LANDMARK 0
#endif

namespace {
constexpr int kEmbedSize = 512;
// 與錄入靜態圖相比，即時影像角度/光照變化大，0.45 常導致「永遠 unknown」；略降並允許環境變數覆寫。
constexpr float kDefaultThreshold = 0.36f;
constexpr float kDefaultTop1Top2Margin = 0.06f;
constexpr int kInputW = 112;
constexpr int kInputH = 112;
constexpr int kMinFaceSize = 40;
constexpr float kMinFaceQualityScore = 60.0f;
const char* kUnknownLabel = "unknown_face";
const char* kDefaultLandmarkPath = "../models/shape_predictor_68_face_landmarks.dat";

// 庫內人數多時，與當前畫面次像的人也會拿到偏高相似度，Top2 易貼近 Top1；固定 margin 會讓畫面在「正名」與 unknown 間跳動。
static float effectiveTop2Margin(float base, size_t gallerySize) {
    if (gallerySize <= 8) return base;
    const float scale = std::sqrt(8.f / static_cast<float>(gallerySize));
    return std::max(0.012f, base * scale);
}
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

struct FaceIdentity {
    std::string name;
    std::vector<float> embedding;
};

class FaceRecognitionModel : public IAIModel {
public:
    int Init(const char* model_path) override {
        printf("[FaceRecognition] Loading model: %s\n", model_path);
        std::vector<char> model_buffer = read_model_file(model_path);
        if (model_buffer.empty()) return -1;

        int ret = AX_ENGINE_CreateHandle(&m_handle, model_buffer.data(), model_buffer.size());
        if (ret != 0) {
            printf("[Error] CreateHandle failed: 0x%x\n", ret);
            return -1;
        }

        AX_ENGINE_GetIOInfo(m_handle, &m_io_info);
        if (!m_io_info || m_io_info->nInputSize == 0 || m_io_info->nOutputSize == 0) {
            printf("[Error] Invalid io info\n");
            return -1;
        }

        m_io_data.pInputs = new AX_ENGINE_IO_BUFFER_T[m_io_info->nInputSize];
        m_io_data.nInputSize = m_io_info->nInputSize;
        m_io_data.pOutputs = new AX_ENGINE_IO_BUFFER_T[m_io_info->nOutputSize];
        m_io_data.nOutputSize = m_io_info->nOutputSize;

        for (unsigned int i = 0; i < m_io_info->nInputSize; ++i) {
            memset(&m_io_data.pInputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));
            AX_U32 size = kInputW * kInputH * 3;
            AX_U64 phy = 0;
            AX_VOID* vir = NULL;
            AX_SYS_MemAlloc(&phy, &vir, size, 128, (const AX_S8*)"ax_face_rec_input");
            m_io_data.pInputs[i].nSize = size;
            m_io_data.pInputs[i].phyAddr = phy;
            m_io_data.pInputs[i].pVirAddr = vir;
            m_io_data.pInputs[i].pStride = new AX_S32[4];
            m_io_data.pInputs[i].pStride[0] = kInputW * 3;
        }

        for (unsigned int i = 0; i < m_io_info->nOutputSize; ++i) {
            memset(&m_io_data.pOutputs[i], 0, sizeof(AX_ENGINE_IO_BUFFER_T));
            auto& info = m_io_info->pOutputs[i];
            m_io_data.pOutputs[i].nSize = info.nSize;
            AX_U64 phy = 0;
            AX_VOID* vir = NULL;
            AX_SYS_MemAlloc(&phy, &vir, info.nSize, 128, (const AX_S8*)"ax_face_rec_output");
            m_io_data.pOutputs[i].phyAddr = phy;
            m_io_data.pOutputs[i].pVirAddr = vir;
            m_io_data.pOutputs[i].pStride = new AX_S32[4];
        }

        const char* dbPathEnv = std::getenv("FACE_DB_PATH");
        const std::string dbPath = (dbPathEnv && *dbPathEnv) ? dbPathEnv : "../models/known_faces_arcface.txt";
        loadKnownFaces(dbPath);

        if (const char* t = std::getenv("FACE_REC_THRESHOLD")) {
            threshold_ = static_cast<float>(atof(t));
        }
        if (const char* m = std::getenv("FACE_REC_MARGIN")) {
            top1Top2Margin_ = static_cast<float>(atof(m));
        }
        if (const char* q = std::getenv("FACE_REC_MIN_QUALITY")) {
            minFaceQualityScore_ = static_cast<float>(atof(q));
        }
        if (const char* s = std::getenv("FACE_REC_MIN_FACE_SIZE")) {
            minFaceSize_ = atoi(s);
        }
        if (const char* e = std::getenv("FACE_REC_SKIP_MARGIN")) {
            skipMargin_ = (atoi(e) != 0);
        }
        if (const char* e = std::getenv("FACE_REC_DEBUG")) {
            debugEveryN_ = std::max(1, atoi(e));
        }
        if (minFaceSize_ < 1) minFaceSize_ = 1;
        if (minFaceQualityScore_ < 0.0f) minFaceQualityScore_ = 0.0f;
        printf("[FaceRecognition] thresholds: rec=%.3f margin=%.3f skipMargin=%d minFace=%d minQuality=%.2f\n",
               threshold_, top1Top2Margin_, skipMargin_ ? 1 : 0, minFaceSize_, minFaceQualityScore_);
#if HAS_DLIB_LANDMARK
        const char* landmarkPathEnv = std::getenv("FACE_LANDMARK_PATH");
        const std::string landmarkPath = (landmarkPathEnv && *landmarkPathEnv) ? landmarkPathEnv : kDefaultLandmarkPath;
        try {
            dlib::deserialize(landmarkPath) >> shapePredictor_;
            landmarkReady_ = true;
            printf("[FaceRecognition] landmark predictor loaded: %s\n", landmarkPath.c_str());
        } catch (...) {
            landmarkReady_ = false;
            printf("[FaceRecognition][FATAL] landmark predictor load failed: %s\n", landmarkPath.c_str());
            return -1;
        }
#else
        printf("[FaceRecognition][FATAL] dlib landmark support not built. Refuse to start recognition plugin.\n");
        return -1;
#endif
        return 0;
    }

    void GetInputSize(int* w, int* h) override {
        *w = kInputW;
        *h = kInputH;
    }

    int Inference(const AX_VIDEO_FRAME_T* pFrame, AI_RESULT_T* pResult) override {
        if (!m_handle) return -1;
        pResult->nObjSize = 0;
        if (knownFaces_.empty()) {
            static int sNoDb = 0;
            if ((++sNoDb % 60) == 0) printf("[FaceRecognition] skip: knownFaces is empty\n");
            return 0;
        }
        if (pFrame->u32Width < (AX_U32)minFaceSize_ || pFrame->u32Height < (AX_U32)minFaceSize_) {
            static int sSmall = 0;
            if ((++sSmall % 60) == 0) {
                printf("[FaceRecognition] skip: face crop too small (%ux%u), min=%d\n",
                       pFrame->u32Width, pFrame->u32Height, minFaceSize_);
            }
            return 0;
        }

        cv::Mat nv12_mat(pFrame->u32Height * 3 / 2, pFrame->u32Width, CV_8UC1, (void*)pFrame->u64VirAddr[0], pFrame->u32PicStride[0]);
        cv::Mat bgr_mat;
        cv::cvtColor(nv12_mat, bgr_mat, cv::COLOR_YUV2BGR_NV12);
        if (bgr_mat.empty()) return -1;

        float quality = 0.0f;
        if (!checkFaceQuality(bgr_mat, quality, minFaceQualityScore_)) {
            static int sQuality = 0;
            if ((++sQuality % 60) == 0) {
                printf("[FaceRecognition] skip: quality too low (%.2f < %.2f)\n", quality, minFaceQualityScore_);
            }
            return 0;
        }

        cv::Mat aligned = makeArcfaceInputWithLandmark(bgr_mat);
        if (aligned.empty()) {
            static int sAlign = 0;
            if ((++sAlign % 60) == 0) printf("[FaceRecognition] skip: landmark align failed\n");
            return 0;
        }
        memcpy(m_io_data.pInputs[0].pVirAddr, aligned.data, kInputW * kInputH * 3);
        AX_SYS_MflushCache(m_io_data.pInputs[0].phyAddr, m_io_data.pInputs[0].pVirAddr, m_io_data.pInputs[0].nSize);

        int ret = AX_ENGINE_RunSync(m_handle, &m_io_data);
        if (ret != 0) return -1;

        std::vector<float> embedding = getEmbedding();
        if (embedding.empty()) return -1;
        l2Normalize(embedding);

        std::string bestName = kUnknownLabel;
        float bestScore = -1.0f;
        float secondScore = -1.0f;
        for (const auto& id : knownFaces_) {
            float sim = cosineSimilarity(embedding, id.embedding);
            if (sim > bestScore) {
                secondScore = bestScore;
                bestScore = sim;
                bestName = id.name;
            } else if (sim > secondScore) {
                secondScore = sim;
            }
        }
        // 僅在「庫內 ≥2 人」時做 Top1–Top2 間隔判斷；單人庫不應因 margin 拒判。
        // 若第二高分無效（維持 -1），不套用 margin。
        const bool marginApplicable =
            !skipMargin_ && knownFaces_.size() > 1 && secondScore > -0.5f;
        const float effMargin = effectiveTop2Margin(top1Top2Margin_, knownFaces_.size());
        const bool rejectByMargin = marginApplicable && (bestScore - secondScore) < effMargin;
        const bool rejectByThreshold = bestScore < threshold_;
        const std::string candidateName = bestName;
        if (rejectByThreshold || rejectByMargin) {
            bestName = kUnknownLabel;
        }
        if (debugEveryN_ > 0) {
            static int sDbg = 0;
            if ((++sDbg % debugEveryN_) == 0) {
                printf("[FaceRecognition] match: candidate=%s best=%.4f second=%.4f thr=%.3f effMargin=%.3f (base=%.3f) marginOn=%d -> %s (thr_fail=%d margin_fail=%d)\n",
                       candidateName.c_str(), bestScore, secondScore, threshold_, effMargin, top1Top2Margin_, marginApplicable ? 1 : 0,
                       bestName.c_str(), rejectByThreshold ? 1 : 0, rejectByMargin ? 1 : 0);
            }
        }

        pResult->nObjSize = 1;
        pResult->objects[0].x = 0.0f;
        pResult->objects[0].y = 0.0f;
        pResult->objects[0].w = 1.0f;
        pResult->objects[0].h = 1.0f;
        pResult->objects[0].score = bestScore;
        pResult->objects[0].class_id = (bestName == kUnknownLabel) ? -1 : 0;
        snprintf(pResult->objects[0].label, 32, "%s", bestName.c_str());
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
        knownFaces_.clear();
        return 0;
    }

private:
    AX_ENGINE_HANDLE m_handle = nullptr;
    AX_ENGINE_IO_INFO_T* m_io_info = nullptr;
    AX_ENGINE_IO_T m_io_data = {0};
    std::vector<FaceIdentity> knownFaces_;
    float threshold_ = kDefaultThreshold;
    float top1Top2Margin_ = kDefaultTop1Top2Margin;
    int minFaceSize_ = kMinFaceSize;
    float minFaceQualityScore_ = kMinFaceQualityScore;
    bool skipMargin_ = false;
    int debugEveryN_ = 0;
#if HAS_DLIB_LANDMARK
    dlib::shape_predictor shapePredictor_;
    bool landmarkReady_ = false;
#endif

    static void l2Normalize(std::vector<float>& v) {
        double s = 0.0;
        for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
        if (s <= 1e-12) return;
        float inv = 1.0f / static_cast<float>(sqrt(s));
        for (float& x : v) x *= inv;
    }

    static float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty()) return 0.0f;
        size_t n = std::min(a.size(), b.size());
        double dot = 0.0;
        for (size_t i = 0; i < n; ++i) dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        return static_cast<float>(dot);
    }

    std::vector<float> getEmbedding() const {
        if (!m_io_data.pOutputs || m_io_data.nOutputSize == 0) return {};
        // 優先選「元素數 ≥512」的輸出；否則取元素數最多的張量（避免非特徵輸出在 [0]）
        unsigned pick = 0;
        size_t bestCnt = 0;
        for (unsigned i = 0; i < m_io_data.nOutputSize; ++i) {
            size_t cnt = m_io_data.pOutputs[i].nSize / sizeof(float);
            if (cnt >= static_cast<size_t>(kEmbedSize)) {
                pick = i;
                bestCnt = cnt;
                break;
            }
            if (cnt > bestCnt) {
                bestCnt = cnt;
                pick = i;
            }
        }
        const float* ptr = reinterpret_cast<const float*>(m_io_data.pOutputs[pick].pVirAddr);
        if (!ptr || bestCnt == 0) return {};
        size_t count = std::min(bestCnt, static_cast<size_t>(kEmbedSize));
        return std::vector<float>(ptr, ptr + count);
    }

    static bool checkFaceQuality(const cv::Mat& bgr, float& score, float minQuality) {
        if (bgr.empty()) return false;
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        cv::Mat lap;
        cv::Laplacian(gray, lap, CV_64F);
        cv::Scalar mu, sigma;
        cv::meanStdDev(lap, mu, sigma);
        score = static_cast<float>(sigma[0] * sigma[0]);
        return score >= minQuality;
    }

    static cv::Mat makeArcfaceInput(const cv::Mat& bgr) {
        if (bgr.empty()) return cv::Mat();
        // 商用常見 fallback：未使用 landmarks 時做中心裁切，減少背景干擾
        int side = std::min(bgr.cols, bgr.rows);
        int x = (bgr.cols - side) / 2;
        int y = (bgr.rows - side) / 2;
        cv::Rect roi(x, y, side, side);
        cv::Mat crop = bgr(roi).clone();
        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_LINEAR);
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        return rgb;
    }

    cv::Mat makeArcfaceInputWithLandmark(const cv::Mat& bgr) const {
#if HAS_DLIB_LANDMARK
        if (!landmarkReady_ || bgr.empty()) return cv::Mat();

        dlib::cv_image<dlib::bgr_pixel> cimg(bgr);
        dlib::rectangle rect(0, 0, bgr.cols - 1, bgr.rows - 1);
        dlib::full_object_detection shape = shapePredictor_(cimg, rect);
        if (shape.num_parts() < 68) return cv::Mat();

        auto avgPt = [&](int i, int j) -> cv::Point2f {
            float x = (shape.part(i).x() + shape.part(j).x()) * 0.5f;
            float y = (shape.part(i).y() + shape.part(j).y()) * 0.5f;
            return cv::Point2f(x, y);
        };

        std::vector<cv::Point2f> src = {
            avgPt(36, 39),
            avgPt(42, 45),
            cv::Point2f((float)shape.part(30).x(), (float)shape.part(30).y()),
            cv::Point2f((float)shape.part(48).x(), (float)shape.part(48).y()),
            cv::Point2f((float)shape.part(54).x(), (float)shape.part(54).y())
        };

        static const std::vector<cv::Point2f> dst = {
            cv::Point2f(38.2946f, 51.6963f),
            cv::Point2f(73.5318f, 51.5014f),
            cv::Point2f(56.0252f, 71.7366f),
            cv::Point2f(41.5493f, 92.3655f),
            cv::Point2f(70.7299f, 92.2041f)
        };

        cv::Mat M = cv::estimateAffinePartial2D(src, dst, cv::noArray(), cv::RANSAC);
        if (M.empty()) return cv::Mat();

        cv::Mat aligned;
        cv::warpAffine(bgr, aligned, M, cv::Size(kInputW, kInputH), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::Mat rgb;
        cv::cvtColor(aligned, rgb, cv::COLOR_BGR2RGB);
        return rgb;
#else
        return cv::Mat();
#endif
    }

    void loadKnownFaces(const std::string& path) {
        knownFaces_.clear();
        std::ifstream f(path);
        if (!f.is_open()) {
            printf("[FaceRecognition] known faces db not found: %s\n", path.c_str());
            return;
        }

        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            FaceIdentity id;
            if (!(iss >> id.name)) continue;
            float x = 0.0f;
            while (iss >> x) id.embedding.push_back(x);
            if (id.embedding.empty()) continue;
            if (static_cast<int>(id.embedding.size()) != kEmbedSize) {
                printf("[FaceRecognition] warn: identity \"%s\" has dim %zu (expected %d), cosine may be biased\n",
                       id.name.c_str(), id.embedding.size(), kEmbedSize);
            }
            l2Normalize(id.embedding);
            knownFaces_.push_back(std::move(id));
        }
        printf("[FaceRecognition] loaded known faces: %zu (db=%s)\n", knownFaces_.size(), path.c_str());
    }
};

extern "C" {
IAIModel* CreateAIModel() { return new FaceRecognitionModel(); }
void DestroyAIModel(IAIModel* p) { delete p; }
}
