#include <signal.h>
#include <unistd.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <cstring>
#include <atomic>
#include <ctime>
#include <cstdint>
#include <array>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <cstdio>
#include <cctype>
#include "manager/video_stream_manager.h" 
#include "manager/config_service.h"
#include "utilities/sample_log.h"  
#include "ax_engine_api.h"
#include "ax_engine_type.h" 

extern "C" {
    #include "common_sys.h"
}

#include "common/video_demux.hpp"

extern "C" volatile AX_S32 gLoopExit = 0;

extern "C" AX_VOID __sigExit(int iSigNo) {
    ALOGN("Catch signal %d, exiting...", iSigNo);
    gLoopExit = 1;
}

static bool has_annexb_start_code(const uint8_t* data, int len) {
    if (!data || len < 4) return false;
    for (int i = 0; i + 3 < len && i < 64; ++i) {
        if (data[i] == 0x00 && data[i + 1] == 0x00) {
            if (data[i + 2] == 0x01) return true;
            if (i + 3 < len && data[i + 2] == 0x00 && data[i + 3] == 0x01) return true;
        }
    }
    return false;
}

static bool looks_like_avcc(const uint8_t* data, int len) {
    if (!data || len < 6) return false;
    if (has_annexb_start_code(data, len)) return false;
    const uint32_t nalu_len =
        (static_cast<uint32_t>(data[0]) << 24) |
        (static_cast<uint32_t>(data[1]) << 16) |
        (static_cast<uint32_t>(data[2]) << 8) |
        static_cast<uint32_t>(data[3]);
    if (nalu_len == 0 || nalu_len + 4u > static_cast<uint32_t>(len)) return false;
    const uint8_t nalu_type = data[4] & 0x1F;
    return nalu_type > 0 && nalu_type <= 23;
}

static int avcc_to_annexb(const uint8_t* src, int len, uint8_t* dst, int dst_cap) {
    if (!src || !dst || len <= 0 || dst_cap < len + 4) return -1;
    int si = 0;
    int di = 0;
    while (si + 4 <= len) {
        uint32_t nalu_len =
            (static_cast<uint32_t>(src[si]) << 24) |
            (static_cast<uint32_t>(src[si + 1]) << 16) |
            (static_cast<uint32_t>(src[si + 2]) << 8) |
            static_cast<uint32_t>(src[si + 3]);
        si += 4;
        if (nalu_len == 0 || si + static_cast<int>(nalu_len) > len) return -1;
        if (di + 4 + static_cast<int>(nalu_len) > dst_cap) return -1;
        dst[di++] = 0x00;
        dst[di++] = 0x00;
        dst[di++] = 0x00;
        dst[di++] = 0x01;
        memcpy(dst + di, src + si, static_cast<size_t>(nalu_len));
        di += static_cast<int>(nalu_len);
        si += static_cast<int>(nalu_len);
    }
    return di;
}

static void print_help(const char* program) {
    printf("Usage: %s [-c config.json] -m <offline|stream> [-o output.h264]\n", program);
    printf("  -c <file>           configuration JSON (default: config/streams_config.json)\n");
    printf("  -m <offline|stream> offline writes H.264; stream pushes MediaMTX/RTP\n");
    printf("  -o <file>           offline H.264 elementary-stream output (offline only)\n");
    printf("  -h                  show this help\n");
    printf("Input limitation: H.264 only; H.265/HEVC is unsupported.\n");
}

static bool has_suffix(const std::string& value, const char* suffix) {
    if (!suffix || value.size() < strlen(suffix)) return false;
    std::string a = value.substr(value.size() - strlen(suffix));
    std::string b = suffix;
    for (char& c : a) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    for (char& c : b) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    return a == b;
}

static std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') quoted += "'\\''";
        else quoted += c;
    }
    return quoted + "'";
}

int main(int argc, char *argv[]) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);

    int ret = 0;
    char input_source[256] = "test.h264";
    std::vector<std::string> input_sources;
    std::string runMode = "stream";
    std::string offlineOutputPath;
    std::string offlineRawOutputPath;

    // 命令行參數：MediaMTX 地址（優先級高於環境變量）
    std::string mediamtx_host_cmd;
    std::string mediamtx_port_cmd;
    std::string mediamtx_endpoint_cmd;  // 用於 --mediamtx IP:PORT 格式

    // 解析命令行参数，支持多个输入源
    // 先解析特殊參數（-m, -c, --mediamtx-host, --mediamtx-port, --mediamtx），記錄哪些參數是特殊參數，避免將它們誤認為輸入源
    std::set<int> specialArgIndices;  // 記錄特殊參數值的索引
    std::map<int, std::string> streamModels;  // stream_index -> model_name
    // 僅在端雲比對時推 raw：--enable-raw 為全部輸入源；--enable-raw 0,2 僅指定 0-based 輸入索引
    bool cli_enable_raw_all = false;
    std::set<size_t> cli_raw_source_indices;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            runMode = argv[++i];
            specialArgIndices.insert(i);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            offlineOutputPath = argv[++i];
            specialArgIndices.insert(i);
        }
    }
    for (int i = 1; i < argc; i++) {
        // 解析 -m 參數（模型配置）
        if (strncmp(argv[i], "-m", 2) == 0 && i + 1 < argc) {
            char* modelArg = argv[i + 1];
            char* colon = strchr(modelArg, ':');
            if (colon) {
                int streamIdx = atoi(modelArg);
                std::string modelName = colon + 1;
                streamModels[streamIdx] = modelName;
                specialArgIndices.insert(i + 1);  // 標記下一個參數是模型參數值
                i++;  // 跳過下一個參數（模型參數）
            }
        }
        // 解析 --mediamtx-host 參數
        else if (strcmp(argv[i], "--mediamtx-host") == 0 && i + 1 < argc) {
            mediamtx_host_cmd = argv[i + 1];
            specialArgIndices.insert(i + 1);  // 標記下一個參數是特殊參數值
            i++;  // 跳過下一個參數
        }
        // 解析 --mediamtx-port 參數
        else if (strcmp(argv[i], "--mediamtx-port") == 0 && i + 1 < argc) {
            mediamtx_port_cmd = argv[i + 1];
            specialArgIndices.insert(i + 1);  // 標記下一個參數是特殊參數值
            i++;  // 跳過下一個參數
        }
        // 解析 --mediamtx 參數（IP:PORT 格式）
        else if (strcmp(argv[i], "--mediamtx") == 0 && i + 1 < argc) {
            mediamtx_endpoint_cmd = argv[i + 1];
            specialArgIndices.insert(i + 1);  // 標記下一個參數是特殊參數值
            i++;  // 跳過下一個參數
        }
        else if (strcmp(argv[i], "--enable-raw") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string tok = argv[i + 1];
                bool ok = !tok.empty();
                for (size_t k = 0; k < tok.size() && ok; k++) {
                    char c = tok[k];
                    if ((c < '0' || c > '9') && c != ',') ok = false;
                }
                if (ok) {
                    size_t start = 0;
                    while (start < tok.size()) {
                        size_t comma = tok.find(',', start);
                        std::string num = (comma == std::string::npos) ? tok.substr(start)
                                                                       : tok.substr(start, comma - start);
                        if (!num.empty()) {
                            cli_raw_source_indices.insert(static_cast<size_t>(atoi(num.c_str())));
                        }
                        if (comma == std::string::npos) break;
                        start = comma + 1;
                    }
                    specialArgIndices.insert(i + 1);
                    i++;
                } else {
                    cli_enable_raw_all = true;
                }
            } else {
                cli_enable_raw_all = true;
            }
        }
    }
    
    // 解析輸入源（跳過所有以 - 開頭的參數和特殊參數值）
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            // 跳過以 - 開頭的參數（如 -m, -c, --mediamtx-host 等）
            if (argv[i][0] == '-') {
                continue;
            }
            // 跳過特殊參數值（如 0:pose, IP地址等）
            if (specialArgIndices.find(i) != specialArgIndices.end()) {
                continue;
            }
            input_sources.push_back(argv[i]);
        }
    }
    
    if (input_sources.empty()) {
        input_sources.push_back(input_source);
    }

    // 配置系统内存池，分别为主码流和AI流分配内存
    COMMON_SYS_POOL_CFG_T poolcfg[] = {
        {1920, 1088, 2048, AX_FORMAT_YUV420_SEMIPLANAR, 60},
        {640, 640, 2048, AX_FORMAT_YUV420_SEMIPLANAR, 40},
    };
    
    // 初始化系统参数结构体
    COMMON_SYS_ARGS_T tCommonArgs = {0};
    tCommonArgs.nPoolCfgCnt = 2;
    tCommonArgs.pPoolCfg = poolcfg;
    
    // 初始化系统资源，失败则退出
    if (COMMON_SYS_Init(&tCommonArgs) != 0) {
        ALOGE("COMMON_SYS_Init failed");
        return -1;
    }

    // 初始化NPU（神经网络处理单元）属性
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE; // 禁用虚拟NPU
    if (AX_ENGINE_Init(&npu_attr) != 0) {
        ALOGE("AX_ENGINE_Init failed");
        COMMON_SYS_DeInit();
        return -1;
    }

    // 配置服务，监控配置文件变化
    ConfigService configService("/dev/shm/ai_config.json");
    configService.startMonitoring();
    
    // 创建视频流管理器
    VideoStreamManager streamManager(configService);
    
    // 檢查是否使用配置文件模式（-c 參數）
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
        useConfigFile = true;
        ALOGN("[Main] Using default streams config: %s", streamsConfigPath.c_str());
    }
    
    // 變量聲明必須在 goto 之前，避免跳過初始化
    std::vector<std::string> sourcesToOpen;
    std::vector<VideoDemux*> demuxes;
    std::vector<VideoStreamManager*> streamManagers;
    std::string currentModel;
    
    // 定義回調數據結構，用於在回調函數中傳遞輸入源信息
    // 每路來源採用 latest-frame 模式：永遠只處理最新幀，避免回調阻塞導致延遲累積
    struct DemuxCallbackData {
        VideoStreamManager* manager;
        std::string sourceUrl;
        std::vector<uint8_t> latestFrame;
        int latestLen;
        bool hasLatestFrame;
        std::mutex latestMutex;
        std::condition_variable latestCv;
        std::atomic<bool> workerRunning;
        std::atomic<bool> endOfStream;
        std::thread workerThread;
        std::atomic<uint64_t> lastFrameTime{0};  // 最後一次收到幀的時間（time_t），用於無幀診斷
        std::atomic<uint64_t> overwriteCount{0};
        std::atomic<uint64_t> callbackFrames{0};
        std::atomic<uint64_t> callbackCostUs{0};
        std::atomic<uint64_t> workerOutFrames{0};
        std::atomic<uint64_t> avccConvertCount{0};
        std::atomic<uint64_t> avccConvertFailCount{0};
        std::atomic<uint64_t> avccConvertCostUs{0};
        std::atomic<uint64_t> metricsWindowStartMs{0};
        DemuxCallbackData() : latestLen(0), hasLatestFrame(false), workerRunning(false), endOfStream(false) {
            latestFrame.resize(256 * 1024);
        }
    };
    std::vector<DemuxCallbackData*> callbackDataList;  // 保存指針以便後續釋放
    std::vector<bool> demuxWarned;   // 無幀診斷：是否已打過警告（避免 goto EXIT 跨過初始化）
    unsigned demuxCheckCounter = 0;

    if (useConfigFile) {
        // 從配置文件加載多流配置
        ALOGN("Loading streams from config file: %s", streamsConfigPath.c_str());
        
        // 準備 MediaMTX 地址（優先使用命令行參數）
        std::string mediamtx_endpoint_for_config;
        if (!mediamtx_endpoint_cmd.empty()) {
            mediamtx_endpoint_for_config = mediamtx_endpoint_cmd;
        } else if (!mediamtx_host_cmd.empty()) {
            if (!mediamtx_port_cmd.empty()) {
                mediamtx_endpoint_for_config = mediamtx_host_cmd + ":" + mediamtx_port_cmd;
            } else {
                mediamtx_endpoint_for_config = mediamtx_host_cmd + ":8000";
            }
        }
        
        if (runMode != "offline" && runMode != "stream") {
            ALOGE("Invalid mode '%s'; expected offline or stream", runMode.c_str());
            ret = -1;
            goto EXIT;
        }
        if (runMode == "offline" && offlineOutputPath.empty()) {
            ALOGE("Offline mode requires -o <output.h264>");
            ret = -1;
            goto EXIT;
        }
        if (runMode == "offline") {
            offlineRawOutputPath = has_suffix(offlineOutputPath, ".mp4")
                ? offlineOutputPath + ".tmp.h264"
                : offlineOutputPath;
        }
        if (!streamManager.loadStreamsFromConfig(streamsConfigPath, mediamtx_endpoint_for_config,
                                                 runMode == "offline", offlineRawOutputPath)) {
            ALOGE("Failed to load streams from config file");
            ret = -1;
            goto EXIT;
        }
        
        // 從已配置的流中提取唯一的輸入源
        std::set<std::string> uniqueSources;
        for (const auto& stream : streamManager.getStreams()) {
            std::string source = stream.getInputSource();
            if (!source.empty()) {
                uniqueSources.insert(source);
            }
        }
        sourcesToOpen.assign(uniqueSources.begin(), uniqueSources.end());
        ALOGN("Found %zu unique input sources from config", sourcesToOpen.size());
    } else {
        // 命令行模式：為每個輸入源創建流配置（參考 sample_multi_demux 的設計）
        // 每個輸入源可以配置不同的模型，實現多模型並行
        int streamIdBase = 1;
        // raw 主碼流（无 OSD），用于 Server 推理输入
        struct RawCandidate {
            size_t inputIndex;
            std::string inputSource;
            uint16_t rawPort;   // MediaMTX udp+rtp 端口（对应 liveX_raw）
            int vdecGroup;      // 与对应 osd 主码流共享 VDEC，避免额外解码
            std::string mediamtxHost; // MediaMTX host（从命令行/环境解析出来）
        };
        std::vector<RawCandidate> rawCandidates;
        
        // streamModels 已經在上面解析了，這裡直接使用
        
        for (size_t i = 0; i < input_sources.size(); i++) {
        // 配置主码流（MediaMTX推送或RTSP输出）
        StreamConfig rtspConfig;
        rtspConfig.streamId = streamIdBase++;
        rtspConfig.inputSource = input_sources[i];
        rtspConfig.inputCodec = "auto";
        // 动态分配唯一的 VDEC/IVPS 组，避免 0/1 硬编码冲突
        // 每個輸入源使用獨立的VDEC組，避免不同輸入源的流混在一起
        int mainVdecGrp = 0, mainIvpsGrp = 0;
        // 使用循環索引作為基礎組號，確保每個輸入源使用不同的VDEC組
        // 但實際分配時會檢查已使用的組，自動跳過衝突
        int baseGroupForSource = i * 2;  // 每個輸入源預留2個組（主碼流+AI流）
        streamManager.getNextAvailableGroup(baseGroupForSource, mainVdecGrp, mainIvpsGrp);
        rtspConfig.ivpsGroup = mainIvpsGrp;
        rtspConfig.vdecGroup = mainVdecGrp;
        
        ALOGN("[Main] Allocated main stream resources for source '%s': VDEC_Group=%d, IVPS_Group=%d", 
              input_sources[i].c_str(), mainVdecGrp, mainIvpsGrp);
        
        rtspConfig.isMediaMTXOutput = true;
        
        // 設置 MediaMTX 地址（優先級：命令行參數 > 環境變量 > 預設值）
        // 為每個流使用不同的端口，避免流混在一起
        // 端口計算：基礎端口 + streamId * 2（為每個流預留2個端口：RTP和RTCP）
        std::string mediamtx_base_endpoint;
        if (!mediamtx_endpoint_cmd.empty()) {
            // 命令行參數：--mediamtx IP:PORT 格式
            mediamtx_base_endpoint = mediamtx_endpoint_cmd;
            ALOGN("[Main] Using MediaMTX endpoint from command line (--mediamtx): %s", mediamtx_endpoint_cmd.c_str());
        } else if (!mediamtx_host_cmd.empty()) {
            // 命令行參數：--mediamtx-host 和 --mediamtx-port
            if (!mediamtx_port_cmd.empty()) {
                mediamtx_base_endpoint = mediamtx_host_cmd + ":" + mediamtx_port_cmd;
            } else {
                mediamtx_base_endpoint = mediamtx_host_cmd + ":8000";
            }
            ALOGN("[Main] Using MediaMTX endpoint from command line: %s", mediamtx_base_endpoint.c_str());
        } else {
            // 檢查環境變量
            const char* mediamtx_host = getenv("MEDIAMTX_HOST");
            const char* mediamtx_port = getenv("MEDIAMTX_RTP_PORT");
            if (mediamtx_host) {
                if (mediamtx_port) {
                    mediamtx_base_endpoint = std::string(mediamtx_host) + ":" + std::string(mediamtx_port);
                } else {
                    mediamtx_base_endpoint = std::string(mediamtx_host) + ":8000";
                }
                ALOGN("[Main] Using MediaMTX endpoint from environment variable: %s", mediamtx_base_endpoint.c_str());
            } else {
                mediamtx_base_endpoint = "127.0.0.1:8000";
                ALOGN("[Main] Using default MediaMTX endpoint: %s", mediamtx_base_endpoint.c_str());
            }
        }
        
        // 為每個流分配不同的端口，避免流混在一起
        // 解析基礎端點
        std::string mediamtx_host;
        uint16_t mediamtx_base_port = 8000;
        size_t colon_pos = mediamtx_base_endpoint.find(':');
        if (colon_pos != std::string::npos) {
            mediamtx_host = mediamtx_base_endpoint.substr(0, colon_pos);
            mediamtx_base_port = (uint16_t)atoi(mediamtx_base_endpoint.substr(colon_pos + 1).c_str());
        } else {
            mediamtx_host = mediamtx_base_endpoint;
        }
        
        // 為每個輸入源的主碼流分配不同的端口：基礎端口 + i * 2
        // i 是輸入源的索引（從0開始），所以第一個輸入源使用基礎端口，第二個輸入源使用基礎端口+2
        // 這樣可以確保每個輸入源的主碼流對應正確的 MediaMTX 路徑（live1, live2, ...）
        uint16_t stream_port = mediamtx_base_port + i * 2;
        rtspConfig.mediamtxEndpoint = mediamtx_host + ":" + std::to_string(stream_port);
        
        ALOGN("[Main] Stream %d (input source %zu) will push to MediaMTX: %s (base port: %d, stream port: %d, expected path: live%zu)", 
              rtspConfig.streamId, i, rtspConfig.mediamtxEndpoint.c_str(), mediamtx_base_port, stream_port, i + 1);
        
        rtspConfig.outputWidth = 1920;
        rtspConfig.outputHeight = 1080;
        rtspConfig.fps = 30;
        // 主碼流也需要啟用 OSD（但不進行 AI 推理，OSD 數據從對應的 AI 流獲取）
        // 這樣檢測框才能顯示在推送到 MediaMTX 的流上
        rtspConfig.enableAI = true;  // 啟用 OSD（但不進行 AI 推理，推理由 AI 流完成）
        // 雖然 enableAI=true，但主碼流不會加載模型，因為沒有設置 modelPath
        // OSD 渲染器會在 VideoStream 構造函數中根據 enableAI 創建
        streamManager.addStream(rtspConfig);

        // raw 主碼流候選：僅在 --enable-raw / --enable-raw i,j 時建立（端雲比對用，避免雙路 RTP 常駐）
        const bool wantRaw = cli_enable_raw_all || (cli_raw_source_indices.count(i) > 0);
        if (wantRaw) {
            RawCandidate rawCand;
            rawCand.inputIndex = i;
            rawCand.inputSource = input_sources[i];
            rawCand.rawPort = static_cast<uint16_t>(stream_port + 1);
            rawCand.vdecGroup = mainVdecGrp; // 與 osd 主碼流共享 VDEC
            rawCand.mediamtxHost = mediamtx_host;
            rawCandidates.push_back(std::move(rawCand));
            ALOGN("[Main] RAW stream enabled for input index %zu (expected live%zu_raw)", i, i + 1);
        } else {
            ALOGN("[Main] RAW stream skipped for input index %zu (use --enable-raw or --enable-raw %zu for benchmark)",
                  i, i);
        }

        // 配置AI流（用于AI推理）- 每个输入源可以有独立的AI流和模型
        StreamConfig aiConfig;
        aiConfig.streamId = streamIdBase++;
        aiConfig.inputSource = input_sources[i];
        aiConfig.inputCodec = "auto";
        
        // AI流與主碼流共用VDEC組（因為來自同一個輸入源）
        // 只需要為AI流分配新的IVPS組
        int aiIvpsGrp;
        int dummyVdecGrp;
        streamManager.getNextAvailableGroup(mainIvpsGrp + 1, dummyVdecGrp, aiIvpsGrp);
        aiConfig.ivpsGroup = aiIvpsGrp;
        aiConfig.vdecGroup = mainVdecGrp;  // 與主碼流共用VDEC組（相同輸入源）
        
        ALOGN("[Main] Allocated AI stream resources for source '%s': VDEC_Group=%d (shared with main), AI_IVPS_Group=%d", 
              input_sources[i].c_str(), mainVdecGrp, aiIvpsGrp);
        
        aiConfig.enableAI = true;  // 标记为AI流
        // AI流的outputWidth/outputHeight是IVPS的輸出尺寸，不是VDEC的解碼尺寸
        // VDEC會輸出原始流尺寸（1920x1080），IVPS會將其縮放到640x640
        aiConfig.outputWidth = 640;
        aiConfig.outputHeight = 640;
        // AI 支路降頻：避免與主碼流爭搶資源造成 WebRTC 預覽卡頓
        aiConfig.fps = 15;
        
        // 為每個流配置模型
        // 優先使用命令行指定的模型，否則使用配置文件中的模型
        if (streamModels.find(i) != streamModels.end()) {
            // 命令行指定了該流的模型
            aiConfig.modelName = streamModels[i];
            aiConfig.modelPath = configService.getModelPath(streamModels[i]);
            aiConfig.isCommandLineModel = true;  // 標記為命令行指定的模型
            ALOGN("Stream %zu configured with model from command line: %s", i, streamModels[i].c_str());
        } else {
            // 使用配置文件中的模型（如果有的話）
            std::string defaultModel = configService.getCurrentModel();
            if (defaultModel != "none" && !defaultModel.empty()) {
                aiConfig.modelName = defaultModel;
                aiConfig.modelPath = configService.getModelPath(defaultModel);
                aiConfig.isCommandLineModel = false;  // 標記為配置文件指定的模型
            }
        }
        
        // 從配置服務獲取閾值
        aiConfig.confThreshold = configService.getConfThreshold();
        aiConfig.nmsThreshold = configService.getNmsThreshold();
        
        streamManager.addStream(aiConfig);
        }

        // 创建 raw 主码流（无 OSD）：避免打乱 AI streamId 顺序
        // IVPS group 选择空闲值，VDEC group 与对应 osd 主码流共享
        if (rawCandidates.empty()) {
            ALOGN("[Main] No RAW main streams (--enable-raw not set); cloud-side pixel benchmark uses OSD path only unless raw enabled.");
        }
        int rawIvpsBaseGroup = 0;
        for (auto& cand : rawCandidates) {
            int tmpVdecDummy, freeIvps;
            streamManager.getNextAvailableGroup(rawIvpsBaseGroup, tmpVdecDummy, freeIvps);
            rawIvpsBaseGroup = freeIvps + 1;

            StreamConfig rawConfig;
            rawConfig.streamId = streamIdBase++;
            rawConfig.inputSource = cand.inputSource;
            rawConfig.inputCodec = "auto";
            rawConfig.ivpsGroup = freeIvps;
            rawConfig.vdecGroup = cand.vdecGroup; // 共享 VDEC
            rawConfig.isMediaMTXOutput = true;
            rawConfig.enableAI = false; // raw：不画 OSD
            rawConfig.mediamtxEndpoint = cand.mediamtxHost + ":" + std::to_string(cand.rawPort);

            rawConfig.outputWidth = 1920;
            rawConfig.outputHeight = 1080;
            rawConfig.fps = 30;

            streamManager.addStream(rawConfig);
            ALOGN("[Main] Loaded RAW main stream (cmd mode): inputIndex=%zu endpoint=%s ivpsGroup=%d vdecGroup=%d streamId=%d rawPort=%u",
                  cand.inputIndex,
                  rawConfig.mediamtxEndpoint.c_str(),
                  rawConfig.ivpsGroup,
                  rawConfig.vdecGroup,
                  rawConfig.streamId,
                  cand.rawPort);
        }

        sourcesToOpen = input_sources;
        ALOGN("[Main] Command line mode: sourcesToOpen.size() = %zu", sourcesToOpen.size());
        for (size_t i = 0; i < sourcesToOpen.size(); i++) {
            ALOGN("[Main] sourcesToOpen[%zu] = %s", i, sourcesToOpen[i].c_str());
        }
    }
    
    // 启动所有流
    ALOGN("Starting all streams...");
    for (auto& stream : streamManager.getStreams()) {
        ALOGN("Starting stream %d (inputSource=%s, isMediaMTXOutput=%d, enableAI=%d)...", 
              stream.getStreamId(), 
              stream.getInputSource().c_str(),
              stream.isMediaMTXOutput() ? 1 : 0,
              stream.isAIEnabled() ? 1 : 0);
        if (!stream.start()) {
            ALOGE("Failed to start stream %d", stream.getStreamId());
        } else {
            ALOGN("Stream %d started successfully (IVPS_Group=%d, VDEC_Group=%d, output_type=%d)", 
                  stream.getStreamId(),
                  stream.getIvpsGroup(),
                  stream.getVdecGroup(),
                  stream.getPipeline() ? stream.getPipeline()->m_output_type : -1);
        }
    }
    ALOGN("All streams started. Total streams: %zu", streamManager.getStreams().size());
    
    // 在所有流啟動後再初始化 OSD，確保主碼流的 pipeline 已經創建完成
    // 這樣可以確保 initializeOSDForStream 能找到對應的主碼流 pipeline
    ALOGN("[Main] Initializing OSD for all AI streams...");
    for (auto& stream : streamManager.getStreams()) {
        if (stream.isAIEnabled() && !stream.isMediaMTXOutput() && stream.getAIProcessor()) {
            ALOGN("[Main] Initializing OSD for AI stream %d", stream.getStreamId());
            streamManager.initializeOSDForAIStream(stream.getStreamId());
        }
    }
    fflush(stdout);  // 強制刷新輸出緩衝區
    
    ALOGN("[Main] About to register config listener...");
    // 注册配置更新回调，动态响应配置变化
    configService.registerConfigListener([&](const ConfigUpdate& update) {
        streamManager.handleConfigUpdate(update);
    });
    ALOGN("[Main] Config listener registered");
    
    ALOGN("[Main] About to read initial config...");
    // 尝试读取初始配置并应用
    currentModel = configService.getCurrentModel();
    ALOGN("[Main] Current model from config: %s", currentModel.c_str());
    if (currentModel != "none" && !currentModel.empty()) {
        ConfigUpdate initialUpdate;
        initialUpdate.modelName = currentModel;
        initialUpdate.modelPath = configService.getModelPath(currentModel);  // 获取完整路径
        initialUpdate.confThreshold = configService.getConfThreshold();
        initialUpdate.nmsThreshold = configService.getNmsThreshold();
        initialUpdate.valid = configService.isConfigValid();
        initialUpdate.streamId = -1;  // 全局更新
        initialUpdate.cameraId = -1;
        ALOGN("[Main] Applying initial config update...");
        streamManager.handleConfigUpdate(initialUpdate);
        ALOGN("[Main] Initial config update applied");
    } else {
        ALOGN("[Main] No initial config to apply (currentModel is none or empty)");
    }
    ALOGN("[Main] Initial config processing completed");
    fflush(stdout);  // 強制刷新輸出緩衝區

    // 创建视频解复用器，支持多路输入
    // 參考 sample_multi_demux：每個輸入源有獨立的demux
    ALOGN("[Main] Creating demux for %zu input sources", sourcesToOpen.size());
    for (size_t i = 0; i < sourcesToOpen.size(); i++) {
        ALOGN("[Main] Processing input source %zu: %s", i, sourcesToOpen[i].c_str());
        VideoDemux* demux = new VideoDemux();
        
        // 創建回調數據結構
        DemuxCallbackData* cbData = new DemuxCallbackData();
        cbData->manager = &streamManager;
        cbData->sourceUrl = sourcesToOpen[i];

        // 每路來源獨立 worker：從 latest-frame 緩衝取幀，避免 RTSP 回調線程被下游阻塞
        cbData->workerRunning.store(true);
        cbData->workerThread = std::thread([](DemuxCallbackData* d) {
            std::vector<uint8_t> localFrame;
            while (d->workerRunning.load()) {
                int len = 0;
                {
                    std::unique_lock<std::mutex> lk(d->latestMutex);
                    d->latestCv.wait(lk, [d] {
                        return !d->workerRunning.load() || d->hasLatestFrame;
                    });
                    if (!d->workerRunning.load()) break;
                    len = d->latestLen;
                    if (len <= 0) {
                        d->hasLatestFrame = false;
                        continue;
                    }
                    if ((int)localFrame.size() < len) localFrame.resize((size_t)len);
                    memcpy(localFrame.data(), d->latestFrame.data(), (size_t)len);
                    d->hasLatestFrame = false;
                }

                pipeline_buffer_t buf = {0};
                buf.p_vir = localFrame.data();
                buf.n_size = len;
                d->manager->processFrame(&buf, d->sourceUrl);
                d->workerOutFrames.fetch_add(1, std::memory_order_relaxed);
            }
        }, cbData);
        callbackDataList.push_back(cbData);
        
        // 定義無捕獲的 lambda（可以轉換為函數指針）
        VideoDemuxCallback callback = [](const void* buff, int len, void* reserve) {
            DemuxCallbackData* cbData = static_cast<DemuxCallbackData*>(reserve);
            if (len == 0) {
                if (cbData) {
                    cbData->endOfStream.store(true);
                    cbData->latestCv.notify_one();
                }
                return;
            }
            if (gLoopExit) return;
            auto cb_begin = std::chrono::steady_clock::now();

            if (!cbData || !cbData->manager) return;
            cbData->lastFrameTime.store(static_cast<uint64_t>(time(nullptr)));

            static int frame_count = 0;
            if (++frame_count % 300 == 0) {
                ALOGN("[Demux] Received frame %d, len=%d, source=%s", 
                      frame_count, len, cbData->sourceUrl.c_str());
            }
            
            // RTSP 回調路徑只做「拷貝到 latest-frame 緩衝」，快速返回。
            // 真正送 VDEC 的工作交給來源專屬 worker。
            size_t needed = static_cast<size_t>(len) + 4;
            {
                std::lock_guard<std::mutex> lk(cbData->latestMutex);
                if (cbData->hasLatestFrame) {
                    cbData->overwriteCount.fetch_add(1, std::memory_order_relaxed);
                }
                if (cbData->latestFrame.size() < needed) {
                    cbData->latestFrame.resize(needed);
                }
                uint8_t* copy = cbData->latestFrame.data();
                memcpy(copy, buff, static_cast<size_t>(len));
                if (looks_like_avcc(static_cast<const uint8_t*>(buff), len)) {
                    auto avcc_begin = std::chrono::steady_clock::now();
                    int converted = avcc_to_annexb(
                        static_cast<const uint8_t*>(buff),
                        len,
                        copy,
                        static_cast<int>(cbData->latestFrame.size()));
                    if (converted > 0) {
                        len = converted;
                        cbData->avccConvertCount.fetch_add(1, std::memory_order_relaxed);
                        static int avcc_convert_count = 0;
                        if ((++avcc_convert_count % 600) == 0) {
                            ALOGN("[Demux] AVCC->AnnexB converted frames=%d, source=%s",
                                  avcc_convert_count, cbData->sourceUrl.c_str());
                        }
                    } else {
                        cbData->avccConvertFailCount.fetch_add(1, std::memory_order_relaxed);
                    }
                    uint64_t avcc_cost_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - avcc_begin).count());
                    cbData->avccConvertCostUs.fetch_add(avcc_cost_us, std::memory_order_relaxed);
                }
                cbData->latestLen = len;
                cbData->hasLatestFrame = true;
            }
            cbData->latestCv.notify_one();
            uint64_t cb_cost_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - cb_begin).count());
            cbData->callbackFrames.fetch_add(1, std::memory_order_relaxed);
            cbData->callbackCostUs.fetch_add(cb_cost_us, std::memory_order_relaxed);
        };
        
        ALOGN("[Main] Attempting to open input source %zu: %s", i, sourcesToOpen[i].c_str());
        // RTSP 不再在 demux 降幀（見 video_demux.hpp）；fps 僅用於本地檔/mp4 的 throttle
        bool openResult = demux->Open(sourcesToOpen[i].c_str(), runMode == "stream", callback, cbData, 30.0);
        if (openResult) {
            demuxes.push_back(demux);
            ALOGN("[Main] Successfully opened input source %zu: %s", i, sourcesToOpen[i].c_str());
        } else {
            ALOGE("[Main] Failed to open input source %zu: %s", i, sourcesToOpen[i].c_str());
            delete demux;
            delete cbData;  // 釋放回調數據
            callbackDataList.pop_back();  // 移除失敗的條目
        }
    }
    
    ALOGN("[Main] Demux creation completed. Total demuxes: %zu", demuxes.size());
    
    if (demuxes.empty()) {
        ALOGE("No valid input sources opened");
        ret = -1;
        goto EXIT;
    }

    ALOGN("System Ready");
    ALOGN("Input sources: %zu", sourcesToOpen.size());
    ALOGN("Total streams: %zu", streamManager.getStreams().size());
    ALOGN("Config file: /dev/shm/ai_config.json");
    ALOGN("Note: Configure MediaMTX with: source: udp+rtp://0.0.0.0:8000");
    
    // 無幀診斷：若某路超過 45 秒未收到幀，打一次警告（便於判斷是否 RTSP 斷開導致斷流）
    static const unsigned kDemuxNoFrameWarnSec = 45;
    demuxWarned.resize(callbackDataList.size(), false);
    demuxCheckCounter = 0;
    while (!gLoopExit && !configService.isShutdownRequested()) {
        if (runMode == "offline") {
            bool allEnded = !callbackDataList.empty();
            for (const auto* cbData : callbackDataList)
                allEnded = allEnded && cbData->endOfStream.load();
            if (allEnded) {
                ALOGN("[Main] Offline input completed");
                break;
            }
        }
        sleep(1);
        demuxCheckCounter++;
        time_t now_wall = time(nullptr);
        for (size_t i = 0; i < callbackDataList.size(); i++) {
            DemuxCallbackData* cbData = callbackDataList[i];
            // 每 10 秒：無幀診斷（與 1 秒聚合埋點分離，避免「10 秒才 exchange 一次」把累計誤當成 fps）
            if (demuxCheckCounter % 10 == 0) {
                uint64_t t = cbData->lastFrameTime.load();
                if (t != 0) {
                    if (static_cast<time_t>(t) + static_cast<time_t>(kDemuxNoFrameWarnSec) < now_wall) {
                        if (!demuxWarned[i]) {
                            ALOGW("[Demux] No frame from source for %us: %s (possible RTSP timeout/disconnect)",
                                  kDemuxNoFrameWarnSec, cbData->sourceUrl.c_str());
                            demuxWarned[i] = true;
                        }
                    } else {
                        demuxWarned[i] = false;
                    }
                }
            }
            uint64_t now_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            uint64_t win_start = cbData->metricsWindowStartMs.load(std::memory_order_relaxed);
            if (win_start == 0) {
                cbData->metricsWindowStartMs.store(now_ms, std::memory_order_relaxed);
            } else if (now_ms - win_start >= 1000) {
                const uint64_t win_ms = (now_ms >= win_start) ? (now_ms - win_start) : 1;
                uint64_t cb_frames = cbData->callbackFrames.exchange(0, std::memory_order_relaxed);
                uint64_t cb_cost = cbData->callbackCostUs.exchange(0, std::memory_order_relaxed);
                uint64_t worker_out = cbData->workerOutFrames.exchange(0, std::memory_order_relaxed);
                uint64_t overwrite = cbData->overwriteCount.exchange(0, std::memory_order_relaxed);
                uint64_t avcc_ok = cbData->avccConvertCount.exchange(0, std::memory_order_relaxed);
                uint64_t avcc_fail = cbData->avccConvertFailCount.exchange(0, std::memory_order_relaxed);
                uint64_t avcc_cost = cbData->avccConvertCostUs.exchange(0, std::memory_order_relaxed);
                const uint64_t cb_s = (cb_frames * 1000ULL) / win_ms;
                const uint64_t out_s = (worker_out * 1000ULL) / win_ms;
                ALOGN("[LOWLAT][DEMUX] src=%s rtsp_cb_per_s=%llu worker_per_s=%llu overwrite=%llu cb_avg_us=%llu avcc_ok=%llu avcc_fail=%llu avcc_avg_us=%llu win_ms=%llu",
                      cbData->sourceUrl.c_str(),
                      (unsigned long long)cb_s,
                      (unsigned long long)out_s,
                      (unsigned long long)overwrite,
                      (unsigned long long)((cb_frames > 0) ? (cb_cost / cb_frames) : 0),
                      (unsigned long long)avcc_ok,
                      (unsigned long long)avcc_fail,
                      (unsigned long long)((avcc_ok + avcc_fail > 0) ? (avcc_cost / (avcc_ok + avcc_fail)) : 0),
                      (unsigned long long)win_ms);
                cbData->metricsWindowStartMs.store(now_ms, std::memory_order_relaxed);
            }
        }
    }
    
    // 停止所有demux
    for (auto* demux : demuxes) {
        demux->Stop();
        delete demux;
    }
    demuxes.clear();
    
    // 釋放回調數據
    for (auto* cbData : callbackDataList) {
        cbData->workerRunning.store(false);
        cbData->latestCv.notify_all();
        if (cbData->workerThread.joinable()) {
            cbData->workerThread.join();
        }
        delete cbData;
    }
    callbackDataList.clear();

    if (runMode == "offline") {
        for (auto& stream : streamManager.getStreams()) {
            stream.stop();
        }
        if (has_suffix(offlineOutputPath, ".mp4")) {
            const std::string command =
                "ffmpeg -y -f h264 -framerate 30 -i " + shell_quote(offlineRawOutputPath) +
                " -c:v copy " + shell_quote(offlineOutputPath);
            ALOGN("[Main] Muxing offline H.264 to MP4: %s", offlineOutputPath.c_str());
            const int muxRet = std::system(command.c_str());
            if (muxRet == 0) {
                std::remove(offlineRawOutputPath.c_str());
                ALOGN("[Main] Offline MP4 generated: %s", offlineOutputPath.c_str());
            } else {
                ALOGE("[Main] FFmpeg MP4 mux failed (%d); raw H.264 kept at %s",
                      muxRet, offlineRawOutputPath.c_str());
                ret = -1;
            }
        }
    }

EXIT:
    // 资源释放由RAII自动完成
    AX_ENGINE_Deinit();
    COMMON_SYS_DeInit();
    ALOGN("App Exit");
    return ret;
}
