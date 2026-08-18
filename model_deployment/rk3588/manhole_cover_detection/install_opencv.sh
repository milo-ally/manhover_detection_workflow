#!/usr/bin/env bash
set -euo pipefail

PREFIX="${1:-$HOME/opt/opencv-dev}"

for command in apt-cache apt-get dpkg dpkg-deb find; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "missing command: $command" >&2
        exit 1
    fi
done

if [[ "$(dpkg --print-architecture)" != "arm64" && "$(dpkg --print-architecture)" != "amd64" ]]; then
    echo "unsupported Debian architecture: $(dpkg --print-architecture)" >&2
    exit 1
fi

mapfile -t opencv_dev_packages < <(
    apt-cache depends libopencv-dev |
        awk '$1 == "Depends:" && $2 ~ /^libopencv.*-dev$/ { print $2 }' |
        sort -u
)
opencv_dev_packages=(libopencv-dev "${opencv_dev_packages[@]}")

download_dir="$(mktemp -d)"
trap 'rm -rf "$download_dir"' EXIT

echo "Downloading OpenCV development packages without installing them..."
(
    cd "$download_dir"
    apt-get download "${opencv_dev_packages[@]}"
)

mkdir -p "$PREFIX"
for package in "$download_dir"/*.deb; do
    dpkg-deb -x "$package" "$PREFIX"
done

config="$(find "$PREFIX" -name OpenCVConfig.cmake -type f -print -quit)"
headers="$PREFIX/usr/include/opencv4/opencv2/core.hpp"
if [[ -z "$config" || ! -f "$headers" ]]; then
    echo "OpenCV development files were not found under: $PREFIX" >&2
    echo "config: $config" >&2
    echo "headers: $headers" >&2
    exit 1
fi

opencv_dir="$(dirname "$config")"
cat > "$PREFIX/opencv_env.sh" <<EOF
export OpenCV_DIR="$opencv_dir"
EOF

echo
echo "OpenCV development files are ready."
echo "OpenCV_DIR=$opencv_dir"
echo
echo "Build with:"
echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=\"$opencv_dir\""
echo "  cmake --build build -j2"
echo
echo "Optional environment setup:"
echo "  source \"$PREFIX/opencv_env.sh\""
