#!/usr/bin/env bash
# build.sh — Compile the NLI Benchmark Suite
#
# Usage:
#   ./build.sh            # Release build (default)
#   ./build.sh debug      # Debug build with sanitizers
#   ./build.sh clean      # Clean previous build
#
# Author: NLI Group 19, 2025-26

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${1:-Release}"

# ─── Handle clean ─────────────────────────────────────────────────────────────
if [[ "${BUILD_TYPE,,}" == "clean" ]]; then
    echo "[build.sh] Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    echo "[build.sh] Done."
    exit 0
fi

# ─── Capitalize build type ────────────────────────────────────────────────────
BUILD_TYPE="$(tr '[:lower:]' '[:upper:]' <<< "${BUILD_TYPE:0:1}")${BUILD_TYPE:1}"
if [[ "${BUILD_TYPE}" == "Debug" ]]; then
    BUILD_TYPE="Debug"
else
    BUILD_TYPE="Release"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   NLI Benchmark Suite — Build Script     ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Source dir  : ${SCRIPT_DIR}"
echo "  Build dir   : ${BUILD_DIR}"
echo "  Build type  : ${BUILD_TYPE}"
echo ""

# ─── Check CMake ──────────────────────────────────────────────────────────────
if ! command -v cmake &>/dev/null; then
    echo "[ERROR] cmake not found. Please install cmake >= 3.16."
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
echo "  CMake       : ${CMAKE_VERSION}"

# ─── Check compiler ───────────────────────────────────────────────────────────
if command -v g++ &>/dev/null; then
    GXX_VERSION=$(g++ --version | head -1)
    echo "  Compiler    : ${GXX_VERSION}"
fi

echo ""

# ─── Configure ────────────────────────────────────────────────────────────────
mkdir -p "${BUILD_DIR}"
echo "[1/2] Configuring..."
cmake -S "${SCRIPT_DIR}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      2>&1 | grep -v "^--$" | grep -v "^$" || true

echo ""

# ─── Build ────────────────────────────────────────────────────────────────────
NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
echo "[2/2] Building with ${NPROC} threads..."
cmake --build "${BUILD_DIR}" -j "${NPROC}"

echo ""
echo "✓ Build complete!"
echo ""
echo "  Executables:"
for exe in nli_benchmark nli_drift nli_ablation; do
    if [[ -f "${BUILD_DIR}/${exe}" ]]; then
        SIZE=$(du -sh "${BUILD_DIR}/${exe}" | cut -f1)
        echo "    ${BUILD_DIR}/${exe}  (${SIZE})"
    fi
done
echo ""
echo "  Run: ./run_all.sh"
echo ""
