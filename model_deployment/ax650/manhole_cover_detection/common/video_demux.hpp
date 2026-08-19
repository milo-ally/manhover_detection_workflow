#pragma once
#include "string"
#include "cstring"
#include "vector"
#include <memory>
#include "thread"
#include "../utilities/sample_log.h"
#include "../../third-party/mp4demux/Mp4Demuxer.h"
#include "RTSPClient.h"
#include "RTPSource.h"  // 需要 RTP_FRAME_TYPE 定義

#define ADJUST_INDICES(start, end, len) \
    if (end > len)                      \
        end = len;                      \
    else if (end < 0)                   \
    {                                   \
        end += len;                     \
        if (end < 0)                    \
            end = 0;                    \
    }                                   \
    if (start < 0)                      \
    {                                   \
        start += len;                   \
        if (start < 0)                  \
            start = 0;                  \
    }

typedef void (*VideoDemuxCallback)(const void *buff, int len, void *reserve);

class frame_rater
{
public:
    frame_rater()
    {
        set_fps(30.0);
    }

    void set_fps(double fps)
    {
        time_between_frames = std::chrono::duration_cast<decltype(time_between_frames)>(std::chrono::duration<double>(1.0 / fps));
        tp = std::chrono::steady_clock::now();
    }

    void sleep()
    {
        // add to time point
        tp += time_between_frames;

        // and sleep until that time point
        std::this_thread::sleep_until(tp);
    }

    /** 在回調入口調用：若已落後超過一幀則丟幀以追趕實時；返回 true 表示應處理本幀，false 表示應丟棄本幀。
     * 睡眠上限 2ms，避免多路 demux 時單路長時間阻塞導致其他路收幀延遲或 i/o timeout。 */
    bool throttle()
    {
        using namespace std::chrono;
        auto now = steady_clock::now();
        if (tp + time_between_frames < now) {
            while (tp + time_between_frames < now)
                tp += time_between_frames;
            return false;
        }
        auto wake = tp;
        tp += time_between_frames;
        auto max_sleep = microseconds(2000);
        if (wake > now && wake - now > max_sleep)
            wake = now + max_sleep;
        if (wake > now)
            std::this_thread::sleep_until(wake);
        return true;
    }

    // RTSP 實時流：不要在回調線程 sleep（會抖動/卡頓）；只做「最多 fps」的 gate。
    // 返回 true: 放行本幀；false: 丟棄本幀（用於降幀而非追趕）。
    bool gate_no_sleep()
    {
        using namespace std::chrono;
        auto now = steady_clock::now();
        if (now < tp) {
            return false;
        }
        tp = now + time_between_frames;
        return true;
    }

private:
    // a duration with a length of 1/FPS seconds
    std::chrono::nanoseconds time_between_frames;

    // the time point we'll add to in every loop
    std::chrono::time_point<std::chrono::steady_clock, decltype(time_between_frames)> tp;
};

class VideoDemux
{
private:
    frame_rater frame_rater_;
    std::string url = "";
    std::vector<std::pair<VideoDemuxCallback, void *>> cbs;
    // std::vector<void *> reserves;
    bool loopPlay = true;
    volatile int gLoopExit = 0;

    std::shared_ptr<std::thread> th_h264_demux = nullptr;
    RTSPClient *rtspClient = nullptr;
    mp4_handle_t mp4_handle = nullptr;

    // 匹配函数：endswith与startwith的内部调用函数
    static int _string_tailmatch(const std::string &self, const std::string &substr, int start, int end, int direction)
    {
        int selflen = (int)self.size();
        int slen = (int)substr.size();

        const char *str = self.c_str();
        const char *sub = substr.c_str();

        // 对输入的范围进行校准
        ADJUST_INDICES(start, end, selflen);

        // 字符串头部匹配（即startswith）
        if (direction < 0)
        {
            if (start + slen > selflen)
                return 0;
        }
        // 字符串尾部匹配（即endswith）
        else
        {
            if (end - start < slen || start > selflen)
                return 0;
            if (end - slen > start)
                start = end - slen;
        }
        if (end - start >= slen)
            // mcmcmp函数用于比较buf1与buf2的前n个字节
            return !std::memcmp(str + start, sub, slen);
        return 0;
    }

    static bool endswith(const std::string &str, const std::string &suffix, int start = 0, int end = INT32_MAX)
    {
        // 调用＿string＿tailmatch函数，参数+1表示字符串尾部匹配
        int result = _string_tailmatch(str, suffix, start, end, +1);
        return static_cast<bool>(result);
    }
    static bool startswith(const std::string &str, const std::string &suffix, int start = 0, int end = INT32_MAX)
    {
        // 调用＿string＿tailmatch函数，参数-1表示字符串头部匹配
        int result = _string_tailmatch(str, suffix, start, end, -1);
        return static_cast<bool>(result);
    }

#define NAL_CODED_SLICE_CRA 21
#define NAL_CODED_SLICE_IDR 5

    typedef struct _SAMPLE_BSPARSER
    {
        FILE *fInput;
        int sSize;
    } SAMPLE_BSPARSER_T;

    typedef enum _SAMPLE_BSBOUNDAR_YTYPE
    {
        BSPARSER_NO_BOUNDARY = 0,
        BSPARSER_BOUNDARY = 1,
        BSPARSER_BOUNDARY_NON_SLICE_NAL = 2
    } SAMPLE_BSBOUNDAR_YTYPE_E;

    static int FindNextStartCode(SAMPLE_BSPARSER_T *tBsInfo, uint32_t *uZeroCount)
    {
        int i;
        int sStart = ftello(tBsInfo->fInput);
        *uZeroCount = 0;

        /* Scan for the beginning of the packet. */
        for (i = 0; i < tBsInfo->sSize && i < tBsInfo->sSize - sStart; i++)
        {
            unsigned char byte;
            int ret_val = fgetc(tBsInfo->fInput);
            if (ret_val == EOF)
                return ftello(tBsInfo->fInput);
            byte = (unsigned char)ret_val;
            switch (byte)
            {
            case 0:
                *uZeroCount = *uZeroCount + 1;
                break;
            case 1:
                /* If there's more than three leading zeros, consider only three
                 * of them to be part of this packet and the rest to be part of
                 * the previous packet. */
                if (*uZeroCount > 3)
                    *uZeroCount = 3;
                if (*uZeroCount >= 2)
                {
                    return ftello(tBsInfo->fInput) - *uZeroCount - 1;
                }
                *uZeroCount = 0;
                break;
            default:
                *uZeroCount = 0;
                break;
            }
        }
        return ftello(tBsInfo->fInput);
    }

    static uint32_t CheckAccessUnitBoundaryH264(FILE *fInput, int sNalBegin)
    {
        uint32_t uBoundary = BSPARSER_NO_BOUNDARY;
        uint32_t uNalType, uVal;

        int sStart = ftello(fInput);

        fseeko(fInput, sNalBegin, SEEK_SET);
        uNalType = (getc(fInput) & 0x1F);

        if (uNalType > NAL_CODED_SLICE_IDR)
            uBoundary = BSPARSER_BOUNDARY_NON_SLICE_NAL;
        else
        {
            uVal = getc(fInput);
            /* Check if first mb in slice is 0(ue(v)). */
            if (uVal & 0x80)
                uBoundary = BSPARSER_BOUNDARY;
        }

        fseeko(fInput, sStart, SEEK_SET);
        return uBoundary;
    }

    static int StreamParserReadFrameH264(SAMPLE_BSPARSER_T *tBsInfo, unsigned char *sBuffer,
                                         int *sSize)
    {
        int sBegin, sEnd, sStrmLen;
        uint32_t sReadLen;
        uint32_t uZeroCount = 0;

        uint32_t uTmp = 0;
        int sNalBegin;
        /* TODO(min): to extract exact one frame instead of a NALU */

        sBegin = FindNextStartCode(tBsInfo, &uZeroCount);
        sNalBegin = sBegin + uZeroCount + 1;
        uTmp = CheckAccessUnitBoundaryH264(tBsInfo->fInput, sNalBegin);
        sEnd = sNalBegin = FindNextStartCode(tBsInfo, &uZeroCount);

        if (sEnd != sBegin && uTmp != BSPARSER_BOUNDARY_NON_SLICE_NAL)
        {
            do
            {
                sEnd = sNalBegin;
                sNalBegin += uZeroCount + 1;

                /* Check access unit boundary for next NAL */
                uTmp = CheckAccessUnitBoundaryH264(tBsInfo->fInput, sNalBegin);
                if (uTmp == BSPARSER_NO_BOUNDARY)
                {
                    sNalBegin = FindNextStartCode(tBsInfo, &uZeroCount);
                }
                else if (uTmp == BSPARSER_BOUNDARY_NON_SLICE_NAL)
                {
                    do
                    {
                        sNalBegin = FindNextStartCode(tBsInfo, &uZeroCount);
                        if (sEnd == sNalBegin)
                            break;
                        sEnd = sNalBegin;
                        sNalBegin += uZeroCount + 1;
                        uTmp = CheckAccessUnitBoundaryH264(tBsInfo->fInput, sNalBegin);
                    } while (uTmp == BSPARSER_BOUNDARY_NON_SLICE_NAL);

                    if (sEnd == sNalBegin)
                    {
                        break;
                    }
                    else if (uTmp == BSPARSER_NO_BOUNDARY)
                    {
                        sNalBegin = FindNextStartCode(tBsInfo, &uZeroCount);
                    }
                }
            } while (uTmp != BSPARSER_BOUNDARY);
        }

        if (sEnd == sBegin)
        {
            return 0; /* End of stream */
        }
        fseeko(tBsInfo->fInput, sBegin, SEEK_SET);
        if (*sSize < sEnd - sBegin)
        {
            *sSize = sEnd - sBegin;
            return 0; /* Insufficient buffer size */
        }

        sStrmLen = sEnd - sBegin;
        sReadLen = fread(sBuffer, 1, sStrmLen, tBsInfo->fInput);

        return sReadLen;
    }

    static void _h264_demux_thread(VideoDemux *demux)
    {
        ALOGI("h264 demux thread +++");
        // [關鍵修復] 使用 thread_local 靜態變量緩存緩衝區，避免每次循環都分配 3MB
        static thread_local std::vector<unsigned char> cbuffer;
        static thread_local int cachedSize = 0;
        const int BUFFER_SIZE = 3 * 1024 * 1024;
        
        // 只在需要時調整大小
        if (cbuffer.size() < BUFFER_SIZE) {
            cbuffer.resize(BUFFER_SIZE);
            cachedSize = BUFFER_SIZE;
        }
        
        do
        {
            SAMPLE_BSPARSER_T tStreamInfo = {0};

            FILE *fInput = NULL;
            fInput = fopen(demux->url.c_str(), "rb");
            if (fInput == NULL)
            {
                ALOGE("Unable to open input file\n");
                for (size_t i = 0; i < demux->cbs.size(); i++)
                {
                    demux->cbs[i].first(nullptr, 0, demux->cbs[i].second);
                }
                return;
            }
            fseek(fInput, 0L, SEEK_END);
            int sLen = ftell(fInput);
            rewind(fInput);
            tStreamInfo.fInput = fInput;
            tStreamInfo.sSize = sLen;

            uint32_t sReadLen = 0;
            while (!demux->gLoopExit)
            {
                // [關鍵修復] sSize 必須是可變的，因為 StreamParserReadFrameH264 會修改它
                // 每次循環開始時重置為緩衝區大小
                int sSize = BUFFER_SIZE;
                sReadLen = StreamParserReadFrameH264(&tStreamInfo, cbuffer.data(), &sSize);
                if (sReadLen == 0)
                {
                    for (size_t i = 0; i < demux->cbs.size(); i++)
                    {
                        demux->cbs[i].first(nullptr, 0, demux->cbs[i].second);
                    }
                    break;
                }
                else
                {
                    if (demux->frame_rater_.throttle())
                    {
                        for (size_t i = 0; i < demux->cbs.size(); i++)
                        {
                            demux->cbs[i].first(cbuffer.data(), sReadLen, demux->cbs[i].second);
                        }
                    }
                }
            }
            fclose(fInput);
        } while (demux->loopPlay && !demux->gLoopExit);
        ALOGI("h264 demux thread ---");
    }

    static int _mp4_frame_callback(const void *buff, int len, frame_type_e type, void *reserve)
    {
        VideoDemux *demux = (VideoDemux *)reserve;
        if (!demux->frame_rater_.throttle())
            return 0;
        for (size_t i = 0; i < demux->cbs.size(); i++)
        {
            demux->cbs[i].first(buff, len, demux->cbs[i].second);
        }
        return 0;
    }

    static void _rtsp_frame_callback(void *arg, RTP_FRAME_TYPE frame_type, int64_t timestamp, unsigned char *buf, int len)
    {
        VideoDemux *demux = (VideoDemux *)arg;
        switch (frame_type)
        {
        case FRAME_TYPE_VIDEO:
            // 已編碼視頻（H.264/H.265）不可在送入硬解前丟幀：任一整幀 AU 被丟會打斷參考鏈，
            // 典型現象為運動區塊馬賽克/花屏直到下一個 IDR。降幀請在編碼輸出端（VENC dst fps）做。
            for (size_t i = 0; i < demux->cbs.size(); i++)
            {
                demux->cbs[i].first(buf, len, demux->cbs[i].second);
            }
            break;
        case FRAME_TYPE_AUDIO:
            break;
        case FRAME_TYPE_ETC:
            break;
        default:
            break;
        }
    }

public:
    VideoDemux(/* args */) {}
    ~VideoDemux() {}

    void Stop()
    {
        gLoopExit = 1;
        if (mp4_handle)
        {
            mp4_close(&mp4_handle);
            mp4_handle = nullptr;
        }
        if (rtspClient)
        {
            rtspClient->closeURL();
            delete rtspClient;
            rtspClient = nullptr;
        }
        if (th_h264_demux.get())
        {
            th_h264_demux->join();
            th_h264_demux = nullptr;
        }
        cbs.clear();
    }

    bool Open(std::string _url, bool _loopPlay, VideoDemuxCallback _cb, void *_reserve, double fps = 30.0)
    {
        Stop();
        frame_rater_.set_fps(fps);
        gLoopExit = 0;
        url = _url;
        loopPlay = _loopPlay;
        // cb = _cb;
        // reserve = _reserve;
        if (_cb)
        {
            cbs.push_back({_cb, _reserve});
        }

        if (startswith(url, "rtsp://"))
        {
            ALOGI("rtsp demux");
            rtspClient = new RTSPClient;
            if (rtspClient->openURL(url.c_str(), 1, 2) != 0)
            {
                ALOGE("rtsp open failed");
                return false;
            }
            if (rtspClient->playURL(_rtsp_frame_callback, this, NULL, NULL) != 0)
            {
                ALOGE("rtsp play failed");
                return false;
            }
        }
        else if (endswith(url, ".mp4"))
        {
            ALOGI("mp4 demux");
            mp4_handle = mp4_open(url.c_str(), _mp4_frame_callback, loopPlay ? 1 : 0, this);
        }
        else if (endswith(url, ".h264"))
        {
            ALOGI("h264 stream demux");
            th_h264_demux.reset(new std::thread(_h264_demux_thread, this));
        }
        else
        {
            ALOGE("unknown video demux");
            return false;
        }
        return true;
    }

    void AddCbs(VideoDemuxCallback _cb, void *_reserve)
    {
        if (_cb)
        {
            cbs.push_back({_cb, _reserve});
        }
    }
};
