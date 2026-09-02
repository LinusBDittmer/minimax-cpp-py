# CMakeLists.txt — standalone build for the basic_usage example
# ==============================================================
#
# Self-contained CMake project; does NOT need to be part of the main
# minimax_cpppy source tree.
#
# Quick start (from this directory):
#
#   cmake -B build -S .
#   cmake --build build
#   ./build/basic_usage
#
# Debug build (enables parameter-validation assertions in the library):
#
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
#   cmake --build build

cmake_minimum_required(VERSION 3.18)
project(basic_usage_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# =============================================================================
# Locate minimax_cpppy
# =============================================================================
#
# Option A (default): library as a subdirectory of this repo.
# This file lives at examples/cpp/01_basic_usage/CMakeLists.txt, so the
# library root is three levels up: ../../../
#
# The options silence the library's own tests and Python bindings — we only
# need the static library.
set(MINIMAX_CPPPY_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(MINIMAX_CPPPY_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../   # path to minimax_cpppy root
    minimax_cpppy_build                     # build-artefact subdirectory
    EXCLUDE_FROM_ALL
)

# Option B: installed library via find_package.
# Uncomment if minimax_cpppy was installed with cmake --install:
#
# find_package(minimax_cpppy REQUIRED)


# =============================================================================
# Build the example executable
# =============================================================================

add_executable(basic_usage main.cpp)

# Linking against the minimax_cpppy target automatically:
#   - adds the library's public include directory (for minimax_cpppy/minimax.hpp)
#   - links libminimax_cpppy.a
#   - propagates compile definitions (e.g. MINIMAX_CPPPY_DEBUG_MODE__ in Debug)
target_link_libraries(basic_usage PRIVATE minimax_cpppy)
