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

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

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
