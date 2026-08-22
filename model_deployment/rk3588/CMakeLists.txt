cmake_minimum_required(VERSION 3.16)
project(rk3588_ai_demo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")

option(RK3588_REQUIRE_RKNN "Fail configuration when the RKNN runtime is unavailable" OFF)
option(RK3588_ENABLE_RGA "Use Rockchip RGA for image conversion when available" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
find_package(dlib QUIET)
find_package(PkgConfig REQUIRED)
pkg_check_modules(AV REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale)

# RKNN headers/libraries may come from the board image, RKNN Toolkit2, or an
# explicitly supplied SDK.  Keeping these cache variables makes cross builds
# deterministic without baking one vendor image's absolute paths into the app.
set(RKNN_SDK_ROOT "" CACHE PATH "RKNN Toolkit2 runtime root")
find_path(RKNN_INCLUDE_DIR rknn_api.h HINTS
    "${CMAKE_SOURCE_DIR}/rk_sdk/include"
    "${RKNN_SDK_ROOT}/include"
    "${RKNN_SDK_ROOT}/rknpu2/runtime/Linux/librknn_api/include"
    /usr/include /usr/local/include)
find_library(RKNN_RUNTIME_LIBRARY NAMES rknnrt HINTS
    "${CMAKE_SOURCE_DIR}/rk_sdk/lib"
    "${CMAKE_SOURCE_DIR}/rk_sdk/lib/aarch64"
    "${RKNN_SDK_ROOT}/lib"
    "${RKNN_SDK_ROOT}/lib64"
    "${RKNN_SDK_ROOT}/aarch64"
    "${RKNN_SDK_ROOT}/rknpu2/runtime/Linux/librknn_api/aarch64"
    /usr/lib /usr/local/lib)

if(RKNN_INCLUDE_DIR AND RKNN_RUNTIME_LIBRARY)
    add_library(RKNN::Runtime UNKNOWN IMPORTED)
    set_target_properties(RKNN::Runtime PROPERTIES
        IMPORTED_LOCATION "${RKNN_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${RKNN_INCLUDE_DIR}")
    set(HAVE_RKNN TRUE)
    message(STATUS "RKNN runtime: ${RKNN_RUNTIME_LIBRARY}")
elseif(RK3588_REQUIRE_RKNN)
    message(FATAL_ERROR
        "RKNN runtime not found. Set RKNN_SDK_ROOT to the Toolkit2 Linux runtime, "
        "or configure with -DRK3588_REQUIRE_RKNN=OFF for the host-side CLI build.")
else()
    set(HAVE_RKNN FALSE)
    message(WARNING "RKNN runtime unavailable: building the host validation backend")
endif()

if(RK3588_ENABLE_RGA)
    find_path(RGA_INCLUDE_DIR im2d.h PATH_SUFFIXES rga)
    find_library(RGA_LIBRARY NAMES rga)
endif()

add_library(rkdl_lib SHARED
    librkdl/src/face_aligner.cpp
    librkdl/src/rknn_model.cpp
    librkdl/src/yolo_postprocess.cpp)
target_include_directories(rkdl_lib PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/librkdl/include")
target_link_libraries(rkdl_lib PUBLIC ${OpenCV_LIBS})
if(dlib_FOUND)
    target_compile_definitions(rkdl_lib PRIVATE HAVE_DLIB=1)
    target_link_libraries(rkdl_lib PRIVATE dlib::dlib)
else()
    message(WARNING "dlib unavailable: ArcFace uses an unaligned crop; detection and other models are unaffected")
endif()
if(HAVE_RKNN)
    target_compile_definitions(rkdl_lib PRIVATE HAVE_RKNN=1)
    target_link_libraries(rkdl_lib PRIVATE RKNN::Runtime)
endif()
if(RGA_INCLUDE_DIR AND RGA_LIBRARY)
    target_compile_definitions(rkdl_lib PRIVATE HAVE_RGA=1)
    target_include_directories(rkdl_lib PRIVATE "${RGA_INCLUDE_DIR}")
    target_link_libraries(rkdl_lib PRIVATE "${RGA_LIBRARY}")
endif()

add_library(rk_media SHARED
    common/rk3588/ffmpeg_pipeline.cpp
    common/rk3588/rtp_output.cpp)
target_include_directories(rk_media PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/common/rk3588")
target_include_directories(rk_media SYSTEM PRIVATE ${AV_INCLUDE_DIRS})
target_compile_options(rk_media PRIVATE ${AV_CFLAGS_OTHER})

# Some Rockchip images ship FFmpeg runtime .so files together with static
# archives, while their pkg-config file exposes only -l names. Prefer the
# versioned shared objects so a shared librk_media.so never absorbs non-PIC .a.
set(AV_SHARED_LIBS)
foreach(av_name IN ITEMS avformat avcodec avutil swscale swresample)
    file(GLOB av_candidates
        "/usr/lib/lib${av_name}.so.*"
        "/usr/lib/aarch64-linux-gnu/lib${av_name}.so.*"
        "/lib/lib${av_name}.so.*"
        "/lib/aarch64-linux-gnu/lib${av_name}.so.*")
    list(SORT av_candidates COMPARE NATURAL ORDER DESCENDING)
    if(av_candidates)
        list(GET av_candidates 0 av_shared)
        list(APPEND AV_SHARED_LIBS "${av_shared}")
    endif()
endforeach()
list(LENGTH AV_SHARED_LIBS AV_SHARED_COUNT)
if(AV_SHARED_COUNT EQUAL 5)
    target_link_libraries(rk_media PUBLIC ${AV_SHARED_LIBS} ${OpenCV_LIBS} pthread)
else()
    target_link_libraries(rk_media PUBLIC PkgConfig::AV ${OpenCV_LIBS} pthread)
endif()

add_library(ByteTrack SHARED
    third-party/ByteTrack/src/BYTETracker.cpp
    third-party/ByteTrack/src/utils.cpp
    third-party/ByteTrack/src/kalmanFilter.cpp
    third-party/ByteTrack/src/lapjv.cpp
    third-party/ByteTrack/src/STrack.cpp
    third-party/ByteTrack/src/bytetrack.cpp)
target_include_directories(ByteTrack PUBLIC
    "${CMAKE_SOURCE_DIR}/third-party/ByteTrack/include"
    "${CMAKE_SOURCE_DIR}/third-party/ByteTrack")
target_link_libraries(ByteTrack PUBLIC ${OpenCV_LIBS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ByteTrack PRIVATE -Wno-error)
endif()

add_library(rk_app_core STATIC
    src/rk3588/app_config.cpp
    src/rk3588/application.cpp
    src/rk3588/alarm_reporter.cpp
    utilities/http_client.cpp)
target_include_directories(rk_app_core PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/librkdl/include"
    "${CMAKE_SOURCE_DIR}/utilities")
target_link_libraries(rk_app_core PUBLIC rkdl_lib rk_media ByteTrack ${OpenCV_LIBS} dl pthread)

add_executable(demo_helmet src/main.cpp)
target_link_libraries(demo_helmet PRIVATE rk_app_core)

# Keep the AX deployment artifact names and dlopen contract.  Each RK plugin is
# a small policy wrapper around the shared RKNN runner; model-specific behavior
# is selected at compile time and all NPU ownership remains in librkdl_lib.
set(RK_PLUGIN_NAMES
    helmet fall constructionsite smoke_fire plate_detection human_detection
    behavior crowd face_detection face_recognition manhole yolo)
foreach(plugin IN LISTS RK_PLUGIN_NAMES)
    add_library(${plugin}_plugin SHARED plugins/rknn_plugin.cpp)
    target_compile_definitions(${plugin}_plugin PRIVATE RK_PLUGIN_NAME="${plugin}")
    target_link_libraries(${plugin}_plugin PRIVATE rkdl_lib ByteTrack)
endforeach()

add_executable(face_enroll src/tools/face_enroll_rk.cpp)
target_link_libraries(face_enroll PRIVATE rkdl_lib ${OpenCV_LIBS})

install(TARGETS demo_helmet face_enroll rkdl_lib rk_media ByteTrack
        RUNTIME DESTINATION bin LIBRARY DESTINATION bin)
foreach(plugin IN LISTS RK_PLUGIN_NAMES)
    install(TARGETS ${plugin}_plugin LIBRARY DESTINATION bin)
endforeach()

include(CTest)
if(BUILD_TESTING AND EXISTS "${CMAKE_SOURCE_DIR}/tests/rkdl_postprocess_test.cpp"
                   AND EXISTS "${CMAKE_SOURCE_DIR}/tests/app_config_test.cpp")
    add_executable(rkdl_postprocess_test tests/rkdl_postprocess_test.cpp)
    target_link_libraries(rkdl_postprocess_test PRIVATE rkdl_lib)
    add_executable(app_config_test tests/app_config_test.cpp src/rk3588/app_config.cpp)
    target_include_directories(app_config_test PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/utilities")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(rkdl_postprocess_test PRIVATE -UNDEBUG)
        target_compile_options(app_config_test PRIVATE -UNDEBUG)
    endif()
    set_target_properties(rkdl_postprocess_test app_config_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests")
    add_test(NAME rkdl_postprocess COMMAND rkdl_postprocess_test)
    add_test(NAME app_config COMMAND app_config_test)
    add_test(NAME cli_help COMMAND demo_helmet --help)
elseif(BUILD_TESTING)
    message(STATUS "Test sources are not present; skipping host-only test targets")
endif()
