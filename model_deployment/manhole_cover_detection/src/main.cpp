#include "ai_interface.h"
#include "ax_engine_api.h"
#include "ax_sys_api.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

static bool bgrToNv12(const cv::Mat& bgr, std::vector<unsigned char>& nv12) {
    if (bgr.empty() || bgr.type() != CV_8UC3 || (bgr.cols & 1) || (bgr.rows & 1)) {
        return false;
    }

    cv::Mat i420;
    cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);

    const size_t ySize = static_cast<size_t>(bgr.cols) * bgr.rows;
    const size_t uvSize = ySize / 2;
    nv12.resize(ySize + uvSize);
    std::memcpy(nv12.data(), i420.data, ySize);

    const unsigned char* u = i420.data + ySize;
    const unsigned char* v = u + ySize / 4;
    unsigned char* uv = nv12.data() + ySize;
    for (size_t i = 0; i < ySize / 4; ++i) {
        uv[i * 2] = u[i];
        uv[i * 2 + 1] = v[i];
    }
    return true;
}

static void drawResult(cv::Mat& frame, const AI_RESULT_T& result) {
    for (AX_U32 i = 0; i < result.nObjSize && i < MAX_DETECT_OBJ_NUM; ++i) {
        const AI_OBJ_T& object = result.objects[i];
        const int x = std::max(0, std::min(frame.cols - 1,
                                           static_cast<int>(object.x * frame.cols)));
        const int y = std::max(0, std::min(frame.rows - 1,
                                           static_cast<int>(object.y * frame.rows)));
        const int w = std::max(1, std::min(frame.cols - x,
                                           static_cast<int>(object.w * frame.cols)));
        const int h = std::max(1, std::min(frame.rows - y,
                                           static_cast<int>(object.h * frame.rows)));
        cv::rectangle(frame, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);

        char label[96] = {};
        std::snprintf(label, sizeof(label), "%s %.2f", object.label, object.score);
        cv::putText(frame, label, cv::Point(x, std::max(20, y - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
}

static void printUsage(const char* program) {
    std::fprintf(stderr,
                 "usage: %s --input input.mp4 --output output.mp4 --model model.axmodel "
                 "[--plugin libmanhole_plugin.so]\n",
                 program);
    std::fprintf(stderr, "       %s input.mp4 output.mp4 model.axmodel\n", program);
}

static bool parseArgs(int argc, char** argv, std::string& inputPath,
                      std::string& outputPath, std::string& modelPath,
                      std::string& pluginPath) {
    std::vector<std::string> positional;
    pluginPath = "./libmanhole_plugin.so";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            return false;
        }
        if (arg == "--input" || arg == "--mp4-in") {
            if (++i >= argc) return false;
            inputPath = argv[i];
        } else if (arg == "--output" || arg == "--mp4-out") {
            if (++i >= argc) return false;
            outputPath = argv[i];
        } else if (arg == "--model" || arg == "--mp4-model") {
            if (++i >= argc) return false;
            modelPath = argv[i];
        } else if (arg == "--plugin") {
            if (++i >= argc) return false;
            pluginPath = argv[i];
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            return false;
        } else {
            positional.push_back(arg);
        }
    }

    if (inputPath.empty() && positional.size() == 3) inputPath = positional[0];
    if (outputPath.empty() && positional.size() == 3) outputPath = positional[1];
    if (modelPath.empty() && positional.size() == 3) modelPath = positional[2];
    return !inputPath.empty() && !outputPath.empty() && !modelPath.empty();
}

int main(int argc, char** argv) {
    std::string inputPath;
    std::string outputPath;
    std::string modelPath;
    std::string pluginPath;
    if (!parseArgs(argc, argv, inputPath, outputPath, modelPath, pluginPath)) {
        printUsage(argv[0]);
        return 1;
    }

    cv::VideoCapture reader(inputPath);
    if (!reader.isOpened()) {
        std::fprintf(stderr, "failed to open input video: %s\n", inputPath.c_str());
        return 1;
    }

    const int width = static_cast<int>(reader.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(reader.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = reader.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0 || fps > 240.0) fps = 25.0;
    if (width <= 0 || height <= 0 || (width & 1) || (height & 1)) {
        std::fprintf(stderr, "input video must have positive even dimensions\n");
        return 1;
    }

    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::fprintf(stderr, "failed to open output video: %s\n", outputPath.c_str());
        return 1;
    }

    void* plugin = nullptr;
    IAIModel* model = nullptr;
    CreateAIModelFunc create = nullptr;
    DestroyAIModelFunc destroy = nullptr;
    AX_U64 phy = 0;
    AX_VOID* vir = nullptr;
    bool sysReady = false;
    bool engineReady = false;
    int status = 1;
    std::string error;
    cv::Mat bgr;
    std::vector<unsigned char> nv12;
    const AX_U32 frameSize = static_cast<AX_U32>(width * height * 3 / 2);

    AX_S32 ret = AX_SYS_Init();
    if (ret != 0) {
        error = "AX_SYS_Init failed: " + std::to_string(ret);
        goto cleanup;
    }
    sysReady = true;

    {
        AX_ENGINE_NPU_ATTR_T attr = {};
        attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        ret = AX_ENGINE_Init(&attr);
        if (ret != 0) {
            error = "AX_ENGINE_Init failed: " + std::to_string(ret);
            goto cleanup;
        }
        engineReady = true;
    }

    plugin = dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!plugin && pluginPath == "./libmanhole_plugin.so") {
        pluginPath = "./bin/libmanhole_plugin.so";
        plugin = dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    }
    if (!plugin) {
        const char* dlError = dlerror();
        error = "dlopen failed for " + pluginPath + ": " + (dlError ? dlError : "unknown error");
        goto cleanup;
    }
    create = reinterpret_cast<CreateAIModelFunc>(dlsym(plugin, "CreateAIModel"));
    destroy = reinterpret_cast<DestroyAIModelFunc>(dlsym(plugin, "DestroyAIModel"));
    if (!create || !destroy) {
        const char* dlError = dlerror();
        error = "dlsym CreateAIModel/DestroyAIModel failed: ";
        error += dlError ? dlError : "symbol not found";
        goto cleanup;
    }

    model = create();
    if (!model) {
        error = "CreateAIModel returned null";
        goto cleanup;
    }
    ret = model->Init(modelPath.c_str());
    if (ret != 0) {
        error = "model Init failed: " + std::to_string(ret) + ", path=" + modelPath;
        goto cleanup;
    }

    ret = AX_SYS_MemAlloc(&phy, &vir, frameSize, 128,
                          reinterpret_cast<const AX_S8*>("debug_nv12"));
    if (ret != 0) {
        error = "AX_SYS_MemAlloc failed: " + std::to_string(ret);
        goto cleanup;
    }

    while (reader.read(bgr)) {
        if (!bgrToNv12(bgr, nv12) || nv12.size() != frameSize) {
            error = "BGR to NV12 conversion failed";
            goto cleanup;
        }
        std::memcpy(vir, nv12.data(), nv12.size());

        AX_VIDEO_FRAME_T frame = {};
        frame.u32Width = static_cast<AX_U32>(width);
        frame.u32Height = static_cast<AX_U32>(height);
        frame.u32PicStride[0] = static_cast<AX_U32>(width);
        frame.u32PicStride[1] = static_cast<AX_U32>(width);
        frame.u32FrameSize = frameSize;
        frame.enImgFormat = AX_FORMAT_YUV420_SEMIPLANAR;
        frame.u64PhyAddr[0] = phy;
        frame.u64PhyAddr[1] = phy + static_cast<AX_U64>(width * height);
        frame.u64VirAddr[0] = reinterpret_cast<AX_U64>(vir);
        frame.u64VirAddr[1] = reinterpret_cast<AX_U64>(vir) + width * height;

        AI_RESULT_T result = {};
        ret = model->Inference(&frame, &result);
        if (ret != 0) {
            error = "Inference failed: " + std::to_string(ret);
            goto cleanup;
        }
        drawResult(bgr, result);
        writer.write(bgr);
    }

    status = 0;

cleanup:
    if (vir) AX_SYS_MemFree(phy, vir);
    if (model) {
        model->Deinit();
        if (destroy) destroy(model);
    }
    if (plugin) dlclose(plugin);
    if (engineReady) AX_ENGINE_Deinit();
    if (sysReady) AX_SYS_Deinit();
    writer.release();
    reader.release();
    if (status != 0) std::fprintf(stderr, "%s\n", error.c_str());
    return status;
}
