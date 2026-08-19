// RK3588 井盖检测 demo 主程序。
// 与 AX650 的 src/main.cpp 同构：-c/-m offline|stream/-o/--mediamtx*/--enable-raw 参数、
// ConfigService(/dev/shm/ai_config.json) + VideoStreamManager、日志前缀一致。
// 仅底层不同：OpenCV 解码 + FFmpeg h264_rkmpp 输出，不依赖 AX 硬件流水线。

#include <signal.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <set>

#include "manager/video_stream_manager.h"
#include "manager/config_service.h"
#include "../utilities/sample_log.h"

extern "C" volatile int gLoopExit = 0;

extern "C" void __sigExit(int iSigNo) {
    ALOGN("Catch signal %d, exiting...", iSigNo);
    gLoopExit = 1;
}

static void print_help(const char* program) {
    printf("Usage: %s [-c config.json] -m <offline|stream> [-o output.mp4]\n", program);
    printf("  -c <file>           streams configuration JSON (default: config/streams_config.json)\n");
    printf("  -m <offline|stream> offline writes a boxed MP4; stream pushes h264_rkmpp to RTSP/MediaMTX\n");
    printf("  -o <file>           offline output MP4 path (offline only)\n");
    printf("  --mediamtx IP:PORT  MediaMTX endpoint (compat; RTSP output is configured in streams_config.json)\n");
    printf("  --mediamtx-host H   MediaMTX host (compat)\n");
    printf("  --mediamtx-port P   MediaMTX port (compat)\n");
    printf("  --enable-raw        enable raw stream (not supported on RK3588 yet)\n");
    printf("  -h                  show this help\n");
    printf("Input source is configured in streams_config.json (local file or RTSP URL).\n");
}

int main(int argc, char* argv[]) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);
    signal(SIGTERM, __sigExit);

    int ret = 0;
    std::string runMode = "stream";
    std::string offlineOutputPath;

    // 命令行参数：MediaMTX 地址（优先级高于配置文件/环境变量，兼容 ax650 CLI）
    std::string mediamtx_host_cmd;
    std::string mediamtx_port_cmd;
    std::string mediamtx_endpoint_cmd;  // --mediamtx IP:PORT 格式
    bool cli_enable_raw_all = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            runMode = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            offlineOutputPath = argv[++i];
        } else if (strcmp(argv[i], "--mediamtx-host") == 0 && i + 1 < argc) {
            mediamtx_host_cmd = argv[++i];
        } else if (strcmp(argv[i], "--mediamtx-port") == 0 && i + 1 < argc) {
            mediamtx_port_cmd = argv[++i];
        } else if (strcmp(argv[i], "--mediamtx") == 0 && i + 1 < argc) {
            mediamtx_endpoint_cmd = argv[++i];
        } else if (strcmp(argv[i], "--enable-raw") == 0) {
            cli_enable_raw_all = true;
        }
    }

    if (runMode != "offline" && runMode != "stream") {
        ALOGE("Invalid mode '%s'; expected offline or stream", runMode.c_str());
        print_help(argv[0]);
        return -1;
    }
    if (runMode == "offline" && offlineOutputPath.empty()) {
        ALOGE("Offline mode requires -o <output.mp4>");
        print_help(argv[0]);
        return -1;
    }
    if (cli_enable_raw_all) {
        ALOGW("[Main] --enable-raw is not supported on RK3588 yet, ignored");
    }

    // 组装 MediaMTX 端点（兼容 ax650：命令行 > 配置文件 > 环境变量）
    std::string mediamtx_endpoint_for_config;
    if (!mediamtx_endpoint_cmd.empty()) {
        mediamtx_endpoint_for_config = mediamtx_endpoint_cmd;
    } else if (!mediamtx_host_cmd.empty()) {
        mediamtx_endpoint_for_config = mediamtx_host_cmd + ":" +
            (mediamtx_port_cmd.empty() ? "8000" : mediamtx_port_cmd);
    }

    // 配置服务：监控 /dev/shm/ai_config.json 的热更新
    ConfigService configService("/dev/shm/ai_config.json");
    configService.startMonitoring();

    VideoStreamManager streamManager(configService);

    // 配置文件路径：-c 指定或默认
    std::string streamsConfigPath;
    bool useConfigFile = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            streamsConfigPath = argv[i + 1];
            useConfigFile = true;
            break;
        }
    }
    if (!useConfigFile) {
        streamsConfigPath = "config/streams_config.json";
        ALOGN("[Main] Using default streams config: %s", streamsConfigPath.c_str());
    }

    ALOGN("[Main] Loading streams from config file: %s", streamsConfigPath.c_str());
    if (!streamManager.loadStreamsFromConfig(streamsConfigPath, mediamtx_endpoint_for_config,
                                             runMode == "offline", offlineOutputPath)) {
        ALOGE("[Main] Failed to load streams from config file");
        ret = -1;
        goto EXIT;
    }

    // 注册配置监听，动态响应配置变化
    configService.registerConfigListener([&](const ConfigUpdate& update) {
        streamManager.handleConfigUpdate(update);
    });

    // 启动所有流
    ALOGN("Starting all streams...");
    for (auto& stream : streamManager.getStreams()) {
        ALOGN("Starting stream %d (inputSource=%s, enableAI=%d)...",
              stream->getStreamId(), stream->getInputSource().c_str(),
              stream->isAIEnabled() ? 1 : 0);
        if (!stream->start()) {
            ALOGE("Failed to start stream %d", stream->getStreamId());
        }
    }
    ALOGN("All streams started. Total streams: %zu", streamManager.getStreams().size());

    // 在所有流启动后初始化 OSD 管理（对齐 ax650 调用顺序）
    ALOGN("[Main] Initializing OSD for all AI streams...");
    for (auto& stream : streamManager.getStreams()) {
        if (stream->isAIEnabled()) {
            streamManager.initializeOSDForAIStream(stream->getStreamId());
        }
    }

    // 读取初始配置并应用
    {
        const std::string currentModel = configService.getCurrentModel();
        ALOGN("[Main] Current model from config: %s", currentModel.c_str());
        if (currentModel != "none" && !currentModel.empty()) {
            ConfigUpdate initialUpdate;
            initialUpdate.modelName = currentModel;
            initialUpdate.modelPath = configService.getModelPath(currentModel);
            initialUpdate.confThreshold = configService.getConfThreshold();
            initialUpdate.nmsThreshold = configService.getNmsThreshold();
            initialUpdate.valid = configService.isConfigValid();
            initialUpdate.streamId = -1;  // 全局更新
            streamManager.handleConfigUpdate(initialUpdate);
        }
    }

    ALOGN("System Ready");
    ALOGN("Input sources: %zu", streamManager.getStreams().size());
    ALOGN("Config file: /dev/shm/ai_config.json");

    while (!gLoopExit && !configService.isShutdownRequested()) {
        if (runMode == "offline" && streamManager.allEnded()) {
            ALOGN("[Main] Offline input completed");
            break;
        }
        sleep(1);
    }

EXIT:
    configService.stopMonitoring();
    ALOGN("App Exit");
    return ret;
}
