#include "ai_interface.h"
#include "ax_engine_api.h"
#include "ax_sys_api.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <cmath>
#include <sstream>
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

static bool isRtspUrl(const std::string& value) {
    return value.rfind("rtsp://", 0) == 0 || value.rfind("rtsps://", 0) == 0;
}

static bool openInputVideo(cv::VideoCapture& reader, const std::string& inputPath) {
    if (!isRtspUrl(inputPath)) {
        return reader.open(inputPath);
    }

    // SSH port forwarding carries TCP only. Force OpenCV's FFmpeg backend to
    // avoid an UDP RTP side channel that cannot cross the tunnel.
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 1);
    return reader.open(inputPath, cv::CAP_FFMPEG);
}

static std::string shellQuote(const std::string& value) {
    std::string quoted = "'";
    for (char character : value) {
        if (character == '\'') {
            quoted += "'\\''";
        } else {
            quoted += character;
        }
    }
    quoted += "'";
    return quoted;
}

static void printUsage(const char* program) {
    std::fprintf(stderr,
                 "usage: %s --input input.mp4|input.rtsp --output output.mp4|output.rtsp "
                 "--model model.axmodel [--plugin libmanhole_plugin.so] "
                 "[--conf-thres 0.25] [--iou-thres 0.45] [--encoder mpeg4]\n",
                 program);
}

static bool parseArgs(int argc, char** argv, std::string& inputPath,
                      std::string& outputPath, std::string& modelPath,
                      std::string& pluginPath, float& confThreshold,
                      float& iouThreshold, std::string& encoder) {
    pluginPath = "./libmanhole_plugin.so";
    confThreshold = 0.25f;
    iouThreshold = 0.45f;
    encoder = "mpeg4";
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
        } else if (arg == "--conf-thres") {
            if (++i >= argc) return false;
            try {
                confThreshold = std::stof(argv[i]);
            } catch (...) {
                return false;
            }
        } else if (arg == "--iou-thres") {
            if (++i >= argc) return false;
            try {
                iouThreshold = std::stof(argv[i]);
            } catch (...) {
                return false;
            }
        } else if (arg == "--encoder") {
            if (++i >= argc) return false;
            encoder = argv[i];
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            return false;
        } else {
            std::fprintf(stderr, "positional arguments are not supported: %s\n", arg.c_str());
            return false;
        }
    }

    return !inputPath.empty() && !outputPath.empty() && !modelPath.empty() &&
           std::isfinite(confThreshold) && std::isfinite(iouThreshold) &&
           confThreshold >= 0.0f && iouThreshold >= 0.0f && !encoder.empty();
}

int main(int argc, char** argv) {
    std::string inputPath;
    std::string outputPath;
    std::string modelPath;
    std::string pluginPath;
    float confThreshold = 0.25f;
    float iouThreshold = 0.45f;
    std::string encoder;
    if (!parseArgs(argc, argv, inputPath, outputPath, modelPath, pluginPath,
                   confThreshold, iouThreshold, encoder)) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string confValue = std::to_string(confThreshold);
    const std::string iouValue = std::to_string(iouThreshold);
    setenv("MANHOLE_CONF_THRESH", confValue.c_str(), 1);
    setenv("MANHOLE_NMS_THRESH", iouValue.c_str(), 1);

    cv::VideoCapture reader;
    if (isRtspUrl(inputPath)) {
        std::fprintf(stdout, "[INFO] rtsp_input_transport=tcp\n");
    }
    openInputVideo(reader, inputPath);
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

    cv::VideoWriter writer;
    FILE* streamPipe = nullptr;
    if (isRtspUrl(outputPath)) {
        std::ostringstream command;
        command << "ffmpeg -loglevel warning -f rawvideo -pix_fmt bgr24"
                << " -s " << width << "x" << height
                << " -r " << fps << " -i pipe:0 -an"
                << " -c:v " << shellQuote(encoder)
                << " -b:v 4000k";
        if (encoder == "libx264") {
            command << " -preset ultrafast -tune zerolatency";
        }
        command << " -pix_fmt yuv420p -f rtsp -rtsp_transport tcp "
                << shellQuote(outputPath);
        std::fprintf(stdout, "[INFO] output_mode=rtsp encoder=%s\n", encoder.c_str());
        std::fprintf(stdout, "[INFO] ffmpeg command: %s\n", command.str().c_str());
        streamPipe = popen(command.str().c_str(), "w");
        if (!streamPipe) {
            std::fprintf(stderr, "failed to start FFmpeg RTSP output: %s\n", outputPath.c_str());
            return 1;
        }
    } else {
        writer.open(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::fprintf(stderr, "failed to open output video: %s\n", outputPath.c_str());
            return 1;
        }
        std::fprintf(stdout, "[INFO] output_mode=file\n");
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
    unsigned long long frameCount = 0;

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
        ++frameCount;
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
        if (streamPipe) {
            if (!bgr.isContinuous()) bgr = bgr.clone();
            const size_t bytes = bgr.total() * bgr.elemSize();
            if (std::fwrite(bgr.data, 1, bytes, streamPipe) != bytes) {
                error = "FFmpeg RTSP output pipe closed";
                goto cleanup;
            }
            std::fflush(streamPipe);
        } else {
            writer.write(bgr);
        }
        std::fprintf(stdout, "[INFO] frame=%llu detections=%u\n",
                     frameCount, static_cast<unsigned int>(result.nObjSize));
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
    if (streamPipe) {
        pclose(streamPipe);
        streamPipe = nullptr;
    } else {
        writer.release();
    }
    reader.release();
    if (status != 0) {
        std::fprintf(stderr, "%s\n", error.c_str());
    } else {
        std::fprintf(stdout, "[INFO] output video: %s, frames: %llu\n",
                     outputPath.c_str(), frameCount);
    }
    return status;
}
