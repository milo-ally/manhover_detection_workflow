#!/usr/bin/env bash


# sudo apt install -y build-essential cmake libjpeg-dev libpng-dev libtiff-dev libwebp-dev libeigen3-dev libtbb-dev
# cd /tmp && wget https://github.com/opencv/opencv/archive/refs/tags/4.5.4.tar.gz -O opencv-4.5.4.tar.gz && tar -xf opencv-4.5.4.tar.gz
# cmake -S /tmp/opencv-4.5.4 -B /tmp/opencv-4.5.4-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/opencv-rk3588 -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_LIST=core,imgproc,imgcodecs,calib3d -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_OPENCL=OFF
# cmake --build /tmp/opencv-4.5.4-build -j2
# sudo cmake --install /tmp/opencv-4.5.4-build
# echo /opt/opencv-rk3588/lib | sudo tee /etc/ld.so.conf.d/opencv-rk3588.conf && sudo ldconfig
# grep -qxF 'export CMAKE_PREFIX_PATH=/opt/opencv-rk3588:$CMAKE_PREFIX_PATH' ~/.bashrc || echo 'export CMAKE_PREFIX_PATH=/opt/opencv-rk3588:$CMAKE_PREFIX_PATH' >> ~/.bashrc
# export CMAKE_PREFIX_PATH=/opt/opencv-rk3588:$CMAKE_PREFIX_PATH

set -euo pipefail

OPENCV_VERSION=${OPENCV_VERSION:-4.5.4}
INSTALL_PREFIX=${OPENCV_PREFIX:-/opt/opencv-rk3588}
JOBS=${OPENCV_BUILD_JOBS:-2}
SRC_DIR="/tmp/opencv-${OPENCV_VERSION}"
BUILD_DIR="/tmp/opencv-${OPENCV_VERSION}-build"
ARCHIVE="/tmp/opencv-${OPENCV_VERSION}.tar.gz"
URL="https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz"

# 安装编译依赖，避免拉入 Ubuntu 普通 FFmpeg 开发包。
sudo apt update
sudo apt install -y \
  build-essential cmake wget ca-certificates \
  libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
  libeigen3-dev libtbb-dev

# 下载并解压源码。
if [[ ! -d "${SRC_DIR}" ]]; then
  wget -O "${ARCHIVE}" "${URL}"
  tar -xf "${ARCHIVE}" -C /tmp
fi

# 只编译项目需要的模块，并关闭 OpenCV 自带 FFmpeg/GStreamer。
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_LIST=core,imgproc,imgcodecs,calib3d \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_opencv_python3=OFF \
  -DWITH_FFMPEG=OFF \
  -DWITH_GSTREAMER=OFF \
  -DWITH_OPENCL=OFF

cmake --build "${BUILD_DIR}" --parallel "${JOBS}"
sudo cmake --install "${BUILD_DIR}"

# 注册动态库，并永久设置 CMake 搜索路径。
echo "${INSTALL_PREFIX}/lib" | sudo tee /etc/ld.so.conf.d/opencv-rk3588.conf >/dev/null
sudo ldconfig
ENV_LINE="export CMAKE_PREFIX_PATH=${INSTALL_PREFIX}:\$CMAKE_PREFIX_PATH"
grep -qxF "${ENV_LINE}" "${HOME}/.bashrc" 2>/dev/null || echo "${ENV_LINE}" >> "${HOME}/.bashrc"
export CMAKE_PREFIX_PATH="${INSTALL_PREFIX}:${CMAKE_PREFIX_PATH:-}"

echo "OpenCV ${OPENCV_VERSION} 已安装到 ${INSTALL_PREFIX}"
