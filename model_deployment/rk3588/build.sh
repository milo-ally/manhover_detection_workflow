#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=${RK3588_BUILD_DIR:-"${script_dir}/build"}

cmake_args=(
  -S "${script_dir}"
  -B "${build_dir}"
  -DCMAKE_BUILD_TYPE=Release
  -DRK3588_REQUIRE_RKNN=ON
)

if [[ -n "${RKNN_SDK_ROOT:-}" ]]; then
  cmake_args+=("-DRKNN_SDK_ROOT=${RKNN_SDK_ROOT}")
fi
if [[ -n "${RK3588_TOOLCHAIN_FILE:-}" ]]; then
  cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=${RK3588_TOOLCHAIN_FILE}")
fi

cmake "${cmake_args[@]}"

# No arguments builds every target. When selecting a plugin or library, also
# rebuild demo_helmet so the executable remains usable with the new .so.
build_targets=("$@")
if [[ ${#build_targets[@]} -gt 0 ]]; then
  has_demo=false
  for target in "${build_targets[@]}"; do
    [[ "$target" == "demo_helmet" ]] && has_demo=true
  done
  [[ "$has_demo" == false ]] && build_targets+=(demo_helmet)
fi

build_cmd=(cmake --build "${build_dir}" --parallel "${RK3588_BUILD_JOBS:-4}")
if [[ ${#build_targets[@]} -gt 0 ]]; then
  build_cmd+=(--target "${build_targets[@]}")
fi
"${build_cmd[@]}"