################################################################################
# CMakeLists.txt — NLI Benchmark Suite (v2.0)
#
# Builds three executables:
#   nli_benchmark  — main read/write benchmark (B-Tree, ALEX, PGM, RMI, NLI variants)
#   nli_drift      — drift detection & repair benchmark
#   nli_ablation   — ablation study (component contribution analysis)
#
# Requirements: CMake >= 3.16, C++17 (GCC 9+, Clang 10+)
#
# Build:
#   mkdir -p build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build . -j$(nproc)
#
# Author: NLI Group 19, 2025-26
################################################################################

cmake_minimum_required(VERSION 3.16)
project(nli_benchmark VERSION 2.0.0 LANGUAGES CXX)

# ─── C++ Standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# ─── Build type default ───────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ─── Compiler flags ───────────────────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-O3 -march=native -mpopcnt -funroll-loops -ffast-math -flto)
        add_link_options(-flto)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options(/O2 /GL /arch:AVX2)
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-O0 -g -fsanitize=address,undefined)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# ─── Include paths ────────────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ─── Executable: Main Benchmark ───────────────────────────────────────────────
add_executable(nli_benchmark benchmark/main_benchmark.cpp)
target_include_directories(nli_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ─── Executable: Drift Benchmark ──────────────────────────────────────────────
add_executable(nli_drift benchmark/drift_benchmark.cpp)
target_include_directories(nli_drift PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ─── Executable: Ablation Study ───────────────────────────────────────────────
add_executable(nli_ablation benchmark/ablation_benchmark.cpp)
target_include_directories(nli_ablation PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Place binaries in build/ for run_all.sh to find them
set_target_properties(nli_benchmark nli_drift nli_ablation
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

message(STATUS "")
message(STATUS "=== NLI Benchmark Suite v2.0 ===")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Targets    : nli_benchmark  nli_drift  nli_ablation")
message(STATUS "")
