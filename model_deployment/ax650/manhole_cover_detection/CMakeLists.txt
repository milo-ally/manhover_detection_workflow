cmake_minimum_required(VERSION 3.10)
project(demo)

add_definitions(-DAXERA_TARGET_CHIP_AX650)
add_definitions(-DAXERA_BSP_VERSION="V2.0.0")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MSP_SDK_DIR "${CMAKE_SOURCE_DIR}/msp_sdk")
if(NOT EXISTS "${MSP_SDK_DIR}/include/ax_sys_api.h")
    message(FATAL_ERROR "Error: msp_sdk not found! Please copy ax650n_bsp_sdk/msp/out to ${MSP_SDK_DIR}")
endif()

set(COMMON_DIR "${CMAKE_SOURCE_DIR}/common")
set(UTILITIES_DIR "${CMAKE_SOURCE_DIR}/utilities")
set(THIRDPARTY_DIR "${CMAKE_SOURCE_DIR}/third-party")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(
    ${MSP_SDK_DIR}/include
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/manager
    ${CMAKE_SOURCE_DIR}/include/osd_renderers
    ${COMMON_DIR}
    ${UTILITIES_DIR}
    ${COMMON_DIR}/common_pipeline
    ${THIRDPARTY_DIR}/mp4demux
    ${THIRDPARTY_DIR}/RTSP/include
    /usr/local/include
    /usr/include
)

link_directories(
    ${MSP_SDK_DIR}/lib
    /soc/lib
    /usr/lib
    /lib
    /usr/local/lib
    ${THIRDPARTY_DIR}/lib
)

find_package(OpenCV REQUIRED)

file(GLOB SRC_COMMON_BASE "${COMMON_DIR}/*.c" "${COMMON_DIR}/*.cpp" "${UTILITIES_DIR}/*.c" "${UTILITIES_DIR}/*.cpp")
file(GLOB SRC_PIPELINE "${COMMON_DIR}/common_pipeline/ax650/*.cpp" "${COMMON_DIR}/common_pipeline/ax650/*.c")
file(GLOB SRC_RTP "${COMMON_DIR}/common_pipeline/rtp_pusher.c")
file(GLOB SRC_MP4 "${THIRDPARTY_DIR}/mp4demux/*.cpp")
file(GLOB_RECURSE SRC_RTSP_CLIENT 
    "${THIRDPARTY_DIR}/RTSP/src/RTSPClient/*.cpp"
    "${THIRDPARTY_DIR}/RTSP/src/RTSPClient/*.c"
    "${THIRDPARTY_DIR}/RTSP/src/Common/*.cpp"
    "${THIRDPARTY_DIR}/RTSP/src/Common/*.c"
    "${THIRDPARTY_DIR}/RTSP/src/Sock/*.cpp"
    "${THIRDPARTY_DIR}/RTSP/src/Sock/*.c"
    "${THIRDPARTY_DIR}/RTSP/src/OS_Common/*.cpp"
    "${THIRDPARTY_DIR}/RTSP/src/OS_Common/*.c"
    "${THIRDPARTY_DIR}/RTSP/src/Util/*.cpp"
    "${THIRDPARTY_DIR}/RTSP/src/Util/*.c"
)

file(GLOB SRC_MANAGER "${CMAKE_SOURCE_DIR}/src/manager/*.cpp")

file(GLOB SRC_OSD_RENDERER "${CMAKE_SOURCE_DIR}/src/osd_renderer_interface.cpp")
set(SRC_OSD_RENDERERS "")

# Some build setups may not pick up newly added files when re-running only `make`.
# Keep this renderer explicit to avoid missing objects at link time.

set(SRC_APP "${CMAKE_SOURCE_DIR}/src/main.cpp")
set(SRC_LIST_LIBS ${SRC_COMMON_BASE} ${SRC_PIPELINE} ${SRC_RTP} ${SRC_MP4} ${SRC_RTSP_CLIENT} ${SRC_MANAGER} ${SRC_OSD_RENDERER} ${SRC_OSD_RENDERERS})




add_library(manhole_plugin SHARED plugins/model_manhole_cover.cpp)
target_link_libraries(manhole_plugin ax_engine ax_sys ${OpenCV_LIBS})

add_executable(demo ${SRC_APP} ${SRC_LIST_LIBS})
set(AX_LIBS ax_sys ax_vdec ax_ivps ax_venc ax_vo ax_mipi ax_3a ax_proton ax_nt_stream ax_nt_ctrl ax_engine ax_interpreter)
set(DEMUX_LIBS avformat avcodec avutil swresample)
set(SYS_LIBS dl pthread m rt stdc++fs)
target_link_libraries(demo
    ${AX_LIBS}
    ${DEMUX_LIBS}
    ${OpenCV_LIBS}
    ${SYS_LIBS})
install(TARGETS demo DESTINATION bin)
install(TARGETS manhole_plugin DESTINATION bin)







