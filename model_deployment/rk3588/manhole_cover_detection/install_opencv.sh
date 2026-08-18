#!/usr/bin/env bash
set -euo pipefail

PREFIX="${1:-$HOME/opt/opencv-dev}"

for command in apt-cache apt-get dpkg dpkg-architecture dpkg-deb find; do
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

multiarch="$(dpkg-architecture -qDEB_HOST_MULTIARCH)"
system_lib_dir="/usr/lib/$multiarch"
prefix_lib_dir="$PREFIX/usr/lib/$multiarch"
mkdir -p "$prefix_lib_dir"
for system_dir in "/usr/lib/$multiarch" "/lib/$multiarch"; do
    for library in "$system_dir"/libopencv_*.so*; do
        [[ -e "$library" ]] || continue
        link="$prefix_lib_dir/$(basename "$library")"
        ln -sfn "$library" "$link"

        # Ubuntu's development metadata may require .so.4.5.4d while the
        # board runtime exposes .so.4.5d. Provide the metadata-compatible alias.
        filename="$(basename "$library")"
        if [[ "$filename" == *.so.*d ]]; then
            stem="${filename%%.so.*}.so"
            ln -sfn "$library" "$prefix_lib_dir/${stem}.4.5.4d"
        fi
    done
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
