cmake_minimum_required(VERSION 3.16)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc CACHE FILEPATH "aarch64 C compiler")
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++ CACHE FILEPATH "aarch64 C++ compiler")

project(manhole_cover_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(AX_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/msp_sdk" CACHE PATH "AX650N SDK root")

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${AX_SDK_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(${AX_SDK_DIR}/lib /soc/lib /usr/lib)

add_library(manhole_plugin SHARED plugins/model_manhole_cover.cpp)
target_link_libraries(manhole_plugin PRIVATE ax_engine ax_sys ${OpenCV_LIBS})
set_target_properties(manhole_plugin PROPERTIES OUTPUT_NAME manhole_plugin)

add_executable(debug_demo src/main.cpp)
target_link_libraries(debug_demo PRIVATE dl ax_engine ax_sys ${OpenCV_LIBS})

install(TARGETS manhole_plugin debug_demo DESTINATION bin)
