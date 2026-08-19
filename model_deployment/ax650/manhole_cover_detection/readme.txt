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

Offline video:

    ./bin/demo -c config/streams_config.json

Set input_source to a local file and output to the original MediaMTX/RTP
configuration when validating an offline or board-side file source.

RTSP input/output:

    ./bin/demo -c config/streams_config.json

Set input_source to rtsp://... in streams_config.json. The original VDEC/IVPS/
VENC/RTP path is used; FFmpeg is not added to the output path.
