cmake_minimum_required(VERSION 3.16)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

project(manhole_cover_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(RKNPU2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rknpu2" CACHE PATH "RKNPU2 runtime root")

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
find_file(OPENCV_VIDEOIO_LIBRARY
    NAMES libopencv_videoio.so libopencv_videoio.so.4.5.4d libopencv_videoio.so.4.5d
    HINTS ${OPENCV_LIBRARY_HINTS}
)

if(NOT OPENCV_INCLUDE_DIR OR NOT OPENCV_CORE_LIBRARY OR
   NOT OPENCV_IMGPROC_LIBRARY OR NOT OPENCV_VIDEOIO_LIBRARY)
    message(FATAL_ERROR
        "OpenCV core/imgproc/videoio development files were not found. "
        "Install them or set -DOpenCV_DIR=/path/to/opencv4.")
endif()

set(OpenCV_INCLUDE_DIRS "${OPENCV_INCLUDE_DIR}")
set(OpenCV_LIBS
    "${OPENCV_VIDEOIO_LIBRARY}"
    "${OPENCV_IMGPROC_LIBRARY}"
    "${OPENCV_CORE_LIBRARY}"
)
message(STATUS "OpenCV include: ${OPENCV_INCLUDE_DIR}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

add_executable(debug_demo
    plugins/model_manhole_cover.cpp
    src/main.cpp
)
target_include_directories(debug_demo PRIVATE
    include
    "${RKNPU2_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(debug_demo PRIVATE
    "${RKNPU2_DIR}/lib/librknnrt.so"
    ${OpenCV_LIBS}
    dl
    pthread
)
set_target_properties(debug_demo PROPERTIES
    BUILD_RPATH "$ORIGIN/../rknpu2/lib"
    INSTALL_RPATH "$ORIGIN/../rknpu2/lib"
)
