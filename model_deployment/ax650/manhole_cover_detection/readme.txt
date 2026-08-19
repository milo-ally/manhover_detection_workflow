AX650 manhole-cover demo

Compile on the AX650 build environment:

    cd device_side
    cmake -S . -B build
    cmake --build build -j$(nproc)

The output files are bin/demo and bin/libmanhole_plugin.so.

The program uses the original device_side chain: ConfigService, VideoStreamManager,
VideoDemux, IVPS/OSD and VENC/RTP. Only the manhole-cover plugin is built and
loaded. The default configuration is config/streams_config.json; an alternative
streams configuration can be passed with -c.

Input limitation: this demo supports H.264 only. H.265/HEVC input is not supported;
convert the source video to H.264 before using offline inference.

Offline video (writes MP4 automatically when the output suffix is .mp4):

    ./bin/demo -c config/streams_config.json -m offline -o output.mp4

Use `-o output.h264` when a raw H.264 elementary stream is desired.

Set input_source to a local file and output to the original MediaMTX/RTP
configuration when validating an offline or board-side file source.

Stream mode (RTSP input or MediaMTX/RTP output):

    ./bin/demo -c config/streams_config.json -m stream

Set input_source to rtsp://... in streams_config.json. The original VDEC/IVPS/
VENC/RTP path is used; FFmpeg is not added to the output path.

Help:

    ./bin/demo -h
