cmake_minimum_required(VERSION 3.10)
project(ax650n_helmet_demo)

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
set(AXDL_DIR "${CMAKE_SOURCE_DIR}/libaxdl")
set(BYTETRACK_DIR "${THIRDPARTY_DIR}/ByteTrack")

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
    ${AXDL_DIR}/include
    ${BYTETRACK_DIR}
    ${BYTETRACK_DIR}/include
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
find_package(dlib QUIET)

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
file(GLOB_RECURSE SRC_AXDL "${AXDL_DIR}/*.cpp" "${AXDL_DIR}/*.c")
file(GLOB_RECURSE SRC_BYTETRACK "${BYTETRACK_DIR}/*.cpp" "${BYTETRACK_DIR}/*.c")

# Manager源文件
file(GLOB SRC_MANAGER "${CMAKE_SOURCE_DIR}/src/manager/*.cpp")

# OSD 渲染器源文件
file(GLOB SRC_OSD_RENDERER "${CMAKE_SOURCE_DIR}/src/osd_renderer_interface.cpp")
file(GLOB SRC_OSD_RENDERERS "${CMAKE_SOURCE_DIR}/src/osd_renderers/*.cpp")

# Some build setups may not pick up newly added files when re-running only `make`.
# Keep this renderer explicit to avoid missing objects at link time.
list(APPEND SRC_OSD_RENDERERS "${CMAKE_SOURCE_DIR}/src/osd_renderers/human_detection_osd_renderer.cpp")
list(APPEND SRC_OSD_RENDERERS "${CMAKE_SOURCE_DIR}/src/osd_renderers/face_detection_osd_renderer.cpp")
list(APPEND SRC_OSD_RENDERERS "${CMAKE_SOURCE_DIR}/src/osd_renderers/face_recognition_osd_renderer.cpp")

set(SRC_APP "${CMAKE_SOURCE_DIR}/src/main.cpp")

set(SRC_LIST_LIBS ${SRC_COMMON_BASE} ${SRC_PIPELINE} ${SRC_RTP} ${SRC_MP4} ${SRC_RTSP_CLIENT} ${SRC_MANAGER} ${SRC_OSD_RENDERER} ${SRC_OSD_RENDERERS})

add_library(ByteTrack SHARED ${SRC_BYTETRACK})

add_library(axdl_lib SHARED ${SRC_AXDL})
target_link_libraries(axdl_lib ByteTrack ${OpenCV_LIBS} ax_sys ax_engine)

# 安全帽檢測插件
add_library(helmet_plugin SHARED plugins/model_helmet.cpp)
target_link_libraries(helmet_plugin axdl_lib ByteTrack ${OpenCV_LIBS})

# 跌倒偵測插件
add_library(fall_plugin SHARED plugins/model_fall.cpp)
target_link_libraries(fall_plugin axdl_lib ByteTrack ${OpenCV_LIBS})

# 作業安全檢測插件
add_library(constructionsite_plugin SHARED plugins/model_constructionsite.cpp)
target_link_libraries(constructionsite_plugin axdl_lib ByteTrack ${OpenCV_LIBS})

# 煙火偵測插件
add_library(smoke_fire_plugin SHARED plugins/model_smoke_fire.cpp)
target_link_libraries(smoke_fire_plugin axdl_lib ByteTrack ${OpenCV_LIBS})

# Manhole-cover five-class plugin: output0 [1,9,8400], non-DFL decoder.
add_library(manhole_cover_plugin SHARED plugins/model_manhole_cover.cpp)
target_link_libraries(manhole_cover_plugin axdl_lib ByteTrack ${OpenCV_LIBS})

# 車牌偵測插件
add_library(plate_detection_plugin SHARED plugins/model_plate_detection.cpp)
target_link_libraries(plate_detection_plugin axdl_lib ByteTrack ${OpenCV_LIBS})

# 人員偵測插件
add_library(human_detection_plugin SHARED plugins/model_human_detection.cpp)
target_link_libraries(human_detection_plugin axdl_lib ByteTrack ${OpenCV_LIBS})

# 行为检测插件
add_library(behavior_plugin SHARED plugins/model_behavior.cpp)
target_link_libraries(behavior_plugin axdl_lib ByteTrack ${OpenCV_LIBS})

# 人員聚集插件（共用人員偵測模型，ByteTrack+DBSCAN 輸出 group bbox）
add_library(crowd_plugin SHARED plugins/model_crowd.cpp)
target_link_libraries(crowd_plugin axdl_lib ByteTrack ${OpenCV_LIBS})

# 人臉檢測插件
add_library(face_detection_plugin SHARED plugins/model_face_detection.cpp)
target_link_libraries(face_detection_plugin axdl_lib ByteTrack ${OpenCV_LIBS})

# 人臉辨識插件（ArcFace）
add_library(face_recognition_plugin SHARED plugins/model_face_recognition.cpp)
target_link_libraries(face_recognition_plugin axdl_lib ByteTrack ${OpenCV_LIBS})
if(dlib_FOUND)
    target_link_libraries(face_recognition_plugin dlib::dlib)
    target_compile_definitions(face_recognition_plugin PRIVATE HAS_DLIB_LANDMARK=1)
endif()

# 為了向後兼容，創建符號鏈接或別名
add_library(yolo_plugin SHARED plugins/model_helmet.cpp)
target_link_libraries(yolo_plugin axdl_lib ByteTrack ${OpenCV_LIBS})

# Generic AI runtime entrypoint. The previous target name was demo_helmet.
add_executable(device_ai_demo ${SRC_APP} ${SRC_LIST_LIBS})

# ============ face_enroll 已禁用（井盖检测不需要） ============
# add_executable(face_enroll src/tools/face_enroll.cpp)
# target_sources(face_enroll PRIVATE
#     ${COMMON_DIR}/common_sys.c
# )
# ==============================================================

set(AX_LIBS ax_sys ax_vdec ax_ivps ax_venc ax_vo ax_mipi ax_3a ax_proton ax_nt_stream ax_nt_ctrl ax_engine ax_interpreter)
set(DEMUX_LIBS avformat avcodec avutil swresample)
set(SYS_LIBS dl pthread m rt stdc++fs)

target_link_libraries(device_ai_demo
    axdl_lib
    ByteTrack
    ${AX_LIBS}
    ${DEMUX_LIBS}
    ${OpenCV_LIBS}
    ${SYS_LIBS}
)

# ============ face_enroll 链接库（已禁用） ============
# target_link_libraries(face_enroll
#     ${AX_LIBS}
#     ${OpenCV_LIBS}
#     ${SYS_LIBS}
# )
# if(dlib_FOUND)
#     target_link_libraries(face_enroll dlib::dlib)
# else()
#     message(WARNING "dlib not found: face_enroll build may fail due to missing landmark dependency")
# endif()
# =====================================================

install(TARGETS device_ai_demo DESTINATION bin)
# install(TARGETS face_enroll DESTINATION bin)  # 已禁用
install(TARGETS yolo_plugin DESTINATION bin)
install(TARGETS helmet_plugin DESTINATION bin)
install(TARGETS fall_plugin DESTINATION bin)
install(TARGETS constructionsite_plugin DESTINATION bin)
install(TARGETS smoke_fire_plugin DESTINATION bin)
install(TARGETS manhole_cover_plugin DESTINATION bin)
install(TARGETS plate_detection_plugin DESTINATION bin)
install(TARGETS human_detection_plugin DESTINATION bin)
install(TARGETS behavior_plugin DESTINATION bin)
install(TARGETS crowd_plugin DESTINATION bin)
install(TARGETS face_detection_plugin DESTINATION bin)
install(TARGETS face_recognition_plugin DESTINATION bin)
install(TARGETS axdl_lib DESTINATION bin)
install(TARGETS ByteTrack DESTINATION bin)