cmake_minimum_required(VERSION 3.16)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

project(manhole_cover_detection LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(RKNPU2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rknpu2" CACHE PATH "RKNPU2 runtime root")

# ===== MPP（档位2：硬件解码/编码，对应 AX650 VDEC/VENC）=====
# 头文件来源：rockchip-linux/mpp tag 1.1.0（commit c08762ebfadeb4e986d2fed993bc7a54862d3ebe）inc/
# 板端系统路径：/usr/include/rockchip + /usr/lib/librockchip_mpp*.so
# 也可放到 ${CMAKE_SOURCE_DIR}/third-party/mpp（见 readme.txt「MPP/RGA 依赖」）
find_path(MPP_INCLUDE_DIR rk_mpi.h
    PATHS /usr/include/rockchip
          ${CMAKE_SOURCE_DIR}/third-party/mpp/inc
)
find_library(MPP_LIB
    NAMES rockchip_mpp
    PATHS /usr/lib /usr/lib/aarch64-linux-gnu
          ${CMAKE_SOURCE_DIR}/third-party/mpp/lib
)

# ===== RGA（档位2：2D 缩放/格式转换，对应 AX650 IVPS）=====
# 头文件来源：rknn_model_zoo 提交 bad6c733.../3rdparty/librga/include
# 板端系统路径：/usr/include/rga + /usr/lib/librga.so
# 也可放到 ${CMAKE_SOURCE_DIR}/third-party/rga
find_path(RGA_INCLUDE_DIR im2d.h
    PATHS /usr/include/rga
          /usr/include
          ${CMAKE_SOURCE_DIR}/third-party/rga/include
)
find_library(RGA_LIB
    NAMES rga
    PATHS /usr/lib /usr/lib/aarch64-linux-gnu
          ${CMAKE_SOURCE_DIR}/third-party/rga/lib
)

if(NOT MPP_INCLUDE_DIR OR NOT MPP_LIB OR NOT RGA_INCLUDE_DIR OR NOT RGA_LIB)
    message(FATAL_ERROR
        "MPP/RGA development files were not found. Install board dev packages "
        "(librockchip-mpp-dev/librga-dev) or populate third-party/mpp and "
        "third-party/rga per readme.txt 'MPP/RGA dependency' section.")
endif()
message(STATUS "MPP include: ${MPP_INCLUDE_DIR}, lib: ${MPP_LIB}")
message(STATUS "RGA include: ${RGA_INCLUDE_DIR}, lib: ${RGA_LIB}")

# ===== OpenCV（OSD CPU 画框用 imgproc）=====
set(OPENCV_INCLUDE_HINTS
    /usr/include/opencv4
    /usr/local/include/opencv4
)
set(OPENCV_LIBRARY_HINTS
    /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
    /lib/${CMAKE_LIBRARY_ARCHITECTURE}
    /usr/local/lib
)
if(OpenCV_DIR)
    list(APPEND OPENCV_INCLUDE_HINTS
        "${OpenCV_DIR}/../../../../include/opencv4"
        "${OpenCV_DIR}/../../../../../include/opencv4"
    )
    list(APPEND OPENCV_LIBRARY_HINTS
        "${OpenCV_DIR}/../.."
        "${OpenCV_DIR}/../../.."
    )
endif()

find_path(OPENCV_INCLUDE_DIR
    NAMES opencv2/core.hpp
    HINTS ${OPENCV_INCLUDE_HINTS}
)
find_file(OPENCV_CORE_LIBRARY
    NAMES libopencv_core.so libopencv_core.so.4.5.4d libopencv_core.so.4.5d
    HINTS ${OPENCV_LIBRARY_HINTS}
)
find_file(OPENCV_IMGPROC_LIBRARY
    NAMES libopencv_imgproc.so libopencv_imgproc.so.4.5.4d libopencv_imgproc.so.4.5d
    HINTS ${OPENCV_LIBRARY_HINTS}
)
if(NOT OPENCV_INCLUDE_DIR OR NOT OPENCV_CORE_LIBRARY OR NOT OPENCV_IMGPROC_LIBRARY)
    message(FATAL_ERROR
        "OpenCV core/imgproc development files were not found. "
        "Install them or set -DOpenCV_DIR=/path/to/opencv4.")
endif()
set(OpenCV_INCLUDE_DIRS "${OPENCV_INCLUDE_DIR}")
set(OpenCV_LIBS "${OPENCV_IMGPROC_LIBRARY}" "${OPENCV_CORE_LIBRARY}")
message(STATUS "OpenCV include: ${OPENCV_INCLUDE_DIR}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# ===== FFmpeg（档位2：H264Demux 解封装，libavformat）=====
# 优先用 pkg-config（板端 Rockchip multimedia FFmpeg 通常带 .pc，能给出完整链接
# 参数，避免静态 .a 逐个缺符号：swresample/drm/zlib/bz2/lzma...）
find_package(PkgConfig)
set(AV_LIBS "")
set(AVFORMAT_INCLUDE_DIR "")
if(PKG_CONFIG_FOUND)
    pkg_check_modules(AVFORMAT_PC QUIET libavformat libavcodec libavutil)
    if(AVFORMAT_PC_FOUND)
        # LIBRARIES 为共享链接参数；LIBRARIES_PRIVATE 为静态 .pc 的传递依赖（swresample/drm 等）
        set(AV_LIBS ${AVFORMAT_PC_LIBRARIES} ${AVFORMAT_PC_LIBRARIES_PRIVATE})
        set(AVFORMAT_INCLUDE_DIR "${AVFORMAT_PC_INCLUDE_DIRS}")
        message(STATUS "FFmpeg via pkg-config: ${AV_LIBS}")
    endif()
endif()

if(NOT AV_LIBS)
    find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h
        PATHS /usr/include /usr/include/aarch64-linux-gnu
    )
    # 优先共享运行库（带版本号 .so.XX），避免静态 .a 的传递依赖链
    find_library(AVFORMAT_LIB
        NAMES libavformat.so.61 libavformat.so.60 libavformat.so.59 libavformat.so.58 libavformat
        PATHS /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu /usr/lib /lib /usr/local/lib
    )
    find_library(AVCODEC_LIB
        NAMES libavcodec.so.61 libavcodec.so.60 libavcodec.so.59 libavcodec.so.58 libavcodec
        PATHS /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu /usr/lib /lib /usr/local/lib
    )
    find_library(AVUTIL_LIB
        NAMES libavutil.so.59 libavutil.so.58 libavutil.so.57 libavutil.so.56 libavutil
        PATHS /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu /usr/lib /lib /usr/local/lib
    )
    if(AVFORMAT_LIB AND AVCODEC_LIB AND AVUTIL_LIB)
        # 静态 .a 回退：补 FFmpeg 传递依赖（zlib/bz2/lzma/swresample/drm），
        # 否则报 undefined reference to 'inflateEnd'/'swr_*'/'drmGetVersion' 等
        set(AV_LIBS ${AVFORMAT_LIB} ${AVCODEC_LIB} ${AVUTIL_LIB}
            swresample z bz2 lzma drm m pthread dl)
        message(STATUS "FFmpeg manual: ${AV_LIBS}")
    endif()
endif()

if(NOT AV_LIBS OR NOT AVFORMAT_INCLUDE_DIR)
    message(FATAL_ERROR
        "FFmpeg libavformat/libavcodec/libavutil not found. Install dev packages "
        "(libavformat-dev/libavcodec-dev/libavutil-dev) or use the board's Rockchip "
        "multimedia FFmpeg shared libs.")
endif()

# ===== 井盖检测插件（主程序 dlopen 加载，内部使用 RKNPU2）=====
add_library(manhole_plugin SHARED plugins/model_manhole_cover.cpp)
target_include_directories(manhole_plugin PRIVATE
    include
    "${RKNPU2_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(manhole_plugin PRIVATE
    "${RKNPU2_DIR}/lib/librknnrt.so"
    ${OpenCV_LIBS}
    dl
    pthread
)
set_target_properties(manhole_plugin PROPERTIES
    BUILD_RPATH "$ORIGIN/../rknpu2/lib"
    INSTALL_RPATH "$ORIGIN/../rknpu2/lib"
)

# ===== 主程序（配置服务 + 多流管理 + 插件加载 + 档位2 MPP/RGA 流水线）=====
add_executable(demo
    src/main.cpp
    src/osd_renderer_interface.cpp
    src/manager/config_service.cpp
    src/manager/ai_processor.cpp
    src/manager/video_stream.cpp
    src/manager/video_stream_manager.cpp
    src/manager/inference_engine.cpp
    src/manager/inference_manager.cpp
    src/manager/osd_renderer.cpp
    common/rk_media.cpp
    common/h264_demux.cpp
    common/common_pipeline/rtp_pusher.c
)
target_include_directories(demo PRIVATE
    include
    include/manager
    utilities
    common
    common/common_pipeline
    ${MPP_INCLUDE_DIR}
    ${RGA_INCLUDE_DIR}
    ${AVFORMAT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(demo PRIVATE
    ${MPP_LIB}
    ${RGA_LIB}
    ${AV_LIBS}
    ${OpenCV_LIBS}
    dl
    pthread
    m
)

install(TARGETS demo DESTINATION bin)
install(TARGETS manhole_plugin DESTINATION bin)
