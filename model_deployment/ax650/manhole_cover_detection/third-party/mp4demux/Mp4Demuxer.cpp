#include "Mp4Demuxer.h"
#define MINIMP4_IMPLEMENTATION
#define ENABLE_AUDIO 0
#define ENABLE_MMAP 0
#include "minimp4/minimp4.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <thread>
#include <vector>
#include <string>
#include <memory>
#include <fcntl.h>
#include <sys/mman.h>

typedef struct
{
    uint8_t *buffer;
    ssize_t size;
} INPUT_BUFFER;

typedef struct
{
#if !ENABLE_MMAP
    std::shared_ptr<uint8_t> buf_h264;
    ssize_t h264_size;
#endif
    std::vector<uint8_t> spspps_buffer;
    std::vector<uint8_t> cache;

    std::string path;
    mp4_frame_callback cb;
    std::thread th_demux;
    void *reserve;
    int loopPlay;
    int loopExit = 0;
} Mp4DemuxerHandle;

static uint8_t *preload(const char *path, ssize_t *data_size)
{
    FILE *file = fopen(path, "rb");
    uint8_t *data;
    *data_size = 0;
    if (!file)
        return 0;
    if (fseek(file, 0, SEEK_END))
        exit(1);
    *data_size = (ssize_t)ftell(file);
    if (*data_size < 0)
        exit(1);
    if (fseek(file, 0, SEEK_SET))
        exit(1);
    data = (unsigned char *)malloc(*data_size);
    if (!data)
        exit(1);
    if ((ssize_t)fread(data, 1, *data_size, file) != *data_size)
        exit(1);
    fclose(file);
    return data;
}

static int read_callback(int64_t offset, void *buffer, size_t size, void *token)
{
    INPUT_BUFFER *buf = (INPUT_BUFFER *)token;
    if (size == 0)
        return 0;
    if (!buf || !buffer || offset < 0 || static_cast<uint64_t>(offset) >= static_cast<uint64_t>(buf->size))
        return 1;
    const size_t available = static_cast<size_t>(buf->size - offset);
    const size_t to_copy = MINIMP4_MIN(size, available);
    memcpy(buffer, buf->buffer + offset, to_copy);
    return to_copy != size;
}

void pth_demux(Mp4DemuxerHandle *handle)
{
#if ENABLE_MMAP
    printf("Read video(%s) file by mmap.\n", handle->path.c_str());
    auto *file_fp = fopen(handle->path.c_str(), "r");
    if (!file_fp)
    {
        printf("Read video(%s) file failed.\n", handle->path.c_str());
        return;
    }
    fseek(file_fp, 0, SEEK_END);
    int file_size = ftell(file_fp);
    fclose(file_fp);
    int fd = open(handle->path.c_str(), O_RDWR, 0644);
    uint8_t *file_addr = (uint8_t *)mmap(NULL, file_size, PROT_WRITE, MAP_SHARED, fd, 0);
    INPUT_BUFFER buf = {file_addr, file_size};
#else
    printf("Read whole video(%s) file to memory.\n", handle->path.c_str());
    handle->buf_h264.reset(preload(handle->path.c_str(), &handle->h264_size), std::default_delete<uint8_t[]>());
    if (!handle->buf_h264 || handle->h264_size <= 0)
    {
        printf("Read video(%s) file failed or empty.\n", handle->path.c_str());
        return;
    }
    INPUT_BUFFER buf = {handle->buf_h264.get(), handle->h264_size};
    int file_size = handle->h264_size;
    uint8_t *file_addr = handle->buf_h264.get();
#endif
    do
    {
        int ntrack = 0;
        unsigned /*ntrack, */ i;
        int spspps_bytes;
        const void *spspps;
        MP4D_demux_t mp4 = {0};
        const int open_ret = MP4D_open(&mp4, read_callback, &buf, file_size);
        if (open_ret == 0 || mp4.track_count == 0 || !mp4.track)
        {
            printf("Open MP4(%s) failed or contains no tracks: ret=%d size=%d tracks=%u.\n",
                   handle->path.c_str(), open_ret, file_size, mp4.track_count);
            MP4D_close(&mp4);
            return;
        }

        MP4D_track_t *tr = mp4.track + ntrack;
        unsigned sum_duration = 0;
        i = 0;
        if (tr->handler_type == MP4D_HANDLER_TYPE_VIDE)
        { // assume h264
#define USE_SHORT_SYNC 0
            char sync[4] = {0, 0, 0, 1};
            const bool is_hevc = (tr->object_type_indication == MP4_OBJECT_TYPE_HEVC);
            const unsigned nal_length_size = (tr->dsi && tr->dsi_bytes > 21)
                                                  ? ((tr->dsi[21] & 0x03u) + 1u)
                                                  : 4u;
            if (is_hevc && tr->dsi && tr->dsi_bytes >= 23) {
                // Convert HEVCDecoderConfigurationRecord (hvcC) VPS/SPS/PPS
                // arrays to Annex-B parameter-set NAL units for AX_VDEC.
                size_t pos = 22;
                const uint8_t num_arrays = tr->dsi[pos++];
                for (uint8_t a = 0; a < num_arrays && pos + 3 <= tr->dsi_bytes; ++a) {
                    const uint8_t nal_type = tr->dsi[pos++] & 0x3f;
                    const uint16_t num_nalus = (uint16_t(tr->dsi[pos]) << 8) | tr->dsi[pos + 1];
                    pos += 2;
                    for (uint16_t n = 0; n < num_nalus && pos + 2 <= tr->dsi_bytes; ++n) {
                        const uint16_t nal_bytes = (uint16_t(tr->dsi[pos]) << 8) | tr->dsi[pos + 1];
                        pos += 2;
                        if (pos + nal_bytes > tr->dsi_bytes) break;
                        if (nal_type == HEVC_NAL_VPS || nal_type == HEVC_NAL_SPS || nal_type == HEVC_NAL_PPS) {
                            handle->spspps_buffer.resize(size_t(nal_bytes) + 4);
                            memcpy(handle->spspps_buffer.data(), sync, 4);
                            memcpy(handle->spspps_buffer.data() + 4, tr->dsi + pos, nal_bytes);
                            if (handle->cb)
                                handle->cb(handle->spspps_buffer.data(), int(nal_bytes) + 4, ft_video, handle->reserve);
                        }
                        pos += nal_bytes;
                    }
                }
            }
            if (!is_hevc) {
            while ((spspps = MP4D_read_sps(&mp4, ntrack, i, &spspps_bytes)) && !handle->loopExit)
            {
                // fwrite(sync + USE_SHORT_SYNC, 1, 4 - USE_SHORT_SYNC, fout);
                // fwrite(spspps, 1, spspps_bytes, fout);
                if (handle->cb)
                {
                    if (spspps_bytes + 4 - USE_SHORT_SYNC > int(handle->spspps_buffer.size()))
                    {
                        handle->spspps_buffer.resize(spspps_bytes + 4 - USE_SHORT_SYNC);
                    }
                    memcpy(handle->spspps_buffer.data(), sync + USE_SHORT_SYNC, 4 - USE_SHORT_SYNC);
                    memcpy(handle->spspps_buffer.data() + 4 - USE_SHORT_SYNC, spspps, spspps_bytes);
                    handle->cb(handle->spspps_buffer.data(), spspps_bytes + 4 - USE_SHORT_SYNC, ft_video, handle->reserve);
                    // handle->cb(spspps, spspps_bytes, ft_video, handle->reserve);
                }

                i++;
            }
            i = 0;
            while ((spspps = MP4D_read_pps(&mp4, ntrack, i, &spspps_bytes)) && !handle->loopExit)
            {
                // fwrite(sync + USE_SHORT_SYNC, 1, 4 - USE_SHORT_SYNC, fout);
                // fwrite(spspps, 1, spspps_bytes, fout);
                if (handle->cb)
                {
                    if (spspps_bytes + 4 - USE_SHORT_SYNC > int(handle->spspps_buffer.size()))
                    {
                        handle->spspps_buffer.resize(spspps_bytes + 4 - USE_SHORT_SYNC);
                    }
                    memcpy(handle->spspps_buffer.data(), sync + USE_SHORT_SYNC, 4 - USE_SHORT_SYNC);
                    memcpy(handle->spspps_buffer.data() + 4 - USE_SHORT_SYNC, spspps, spspps_bytes);
                    handle->cb(handle->spspps_buffer.data(), spspps_bytes + 4 - USE_SHORT_SYNC, ft_video, handle->reserve);
                    // handle->cb(spspps, spspps_bytes, ft_video, handle->reserve);
                }
                i++;
            }
            }
            for (i = 0; i < mp4.track[ntrack].sample_count && !handle->loopExit; i++)
            {
                unsigned frame_bytes, timestamp, duration;
                MP4D_file_offset_t ofs = MP4D_frame_offset(&mp4, ntrack, i, &frame_bytes, &timestamp, &duration);
                uint8_t *mem = file_addr + ofs;
                sum_duration += duration;
                while (frame_bytes)
                {
                    if (frame_bytes < nal_length_size) {
                        printf("error: demux sample has invalid NAL length\n");
                        break;
                    }
                    uint32_t nal_size = 0;
                    for (unsigned n = 0; n < nal_length_size; ++n)
                        nal_size = (nal_size << 8) | mem[n];
                    const uint32_t input_size = nal_size + nal_length_size;
                    const uint32_t output_size = nal_size + 4;
                    if (nal_size == 0 || input_size > frame_bytes) {
                        printf("error: demux sample NAL size is invalid\n");
                        break;
                    }
                    if (handle->cache.size() < output_size)
                    {
                        handle->cache.resize(output_size);
                    }

                    memcpy(handle->cache.data() + 4, mem + nal_length_size, nal_size);
                    handle->cache[0] = 0;
                    handle->cache[1] = 0;
                    handle->cache[2] = 0;
                    handle->cache[3] = 1;
                    if (handle->cb)
                    {
                        handle->cb(handle->cache.data(), output_size, ft_video, handle->reserve);
                    }
                    // fwrite(mem + USE_SHORT_SYNC, 1, size - USE_SHORT_SYNC, fout);
                    frame_bytes -= input_size;
                    mem += input_size;
                }
            }
        }
        MP4D_close(&mp4);
    } while (handle->loopPlay && !handle->loopExit);
#if ENABLE_MMAP
    munmap(file_addr, file_size);
#endif

    if (handle->cb)
    {
        handle->cb(0, 0, ft_video, handle->reserve);
    }
}

mp4_handle_t mp4_open(const char *path, mp4_frame_callback cb, int loopPlay, void *reserve)
{
    Mp4DemuxerHandle *handle = new Mp4DemuxerHandle;
    handle->path = path;
    handle->cb = cb;
    handle->reserve = reserve;
    handle->loopPlay = loopPlay;
    // Initialize all fields before starting the worker. Otherwise the worker
    // can call the callback with an uninitialized reserve pointer.
    handle->th_demux = std::thread(pth_demux, handle);
    return handle;
}

void mp4_close(mp4_handle_t *mp4handle)
{
    Mp4DemuxerHandle *handle = (Mp4DemuxerHandle *)*mp4handle;
    if (handle)
    {
        handle->loopExit = 1;
        handle->th_demux.join();
        delete handle;
        handle = nullptr;
        *mp4handle = nullptr;
    }
}
