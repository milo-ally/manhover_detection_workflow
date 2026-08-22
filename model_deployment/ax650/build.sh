#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=${AX650N_BUILD_DIR:-"${script_dir}/build"}

cmake_args=(
  -S "${script_dir}"
  -B "${build_dir}"
  -DCMAKE_BUILD_TYPE=Release
)

if [[ -n "${AX650N_TOOLCHAIN_FILE:-}" ]]; then
  cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=${AX650N_TOOLCHAIN_FILE}")
fi

cmake "${cmake_args[@]}"
build_targets=("$@")
if [[ ${#build_targets[@]} -gt 0 ]]; then
  has_demo=false
  for target in "${build_targets[@]}"; do
    [[ "$target" == "demo_helmet" ]] && has_demo=true
  done
  [[ "$has_demo" == false ]] && build_targets+=(demo_helmet)
fi

build_cmd=(cmake --build "${build_dir}" --parallel "${AX650N_BUILD_JOBS:-4}")
if [[ ${#build_targets[@]} -gt 0 ]]; then
  build_cmd+=(--target "${build_targets[@]}")
fi
"${build_cmd[@]}"
