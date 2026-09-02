# CMakeLists.txt — standalone build for the biased_vs_unbiased example
# ======================================================================
#
# Quick start (from this directory):
#
#   cmake -B build -S .
#   cmake --build build
#   ./build/biased_vs_unbiased

cmake_minimum_required(VERSION 3.18)
project(biased_vs_unbiased_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# =============================================================================
# Locate minimax_cpppy
# =============================================================================

set(MINIMAX_CPPPY_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(MINIMAX_CPPPY_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../   # path to minimax_cpppy root
    minimax_cpppy_build
    EXCLUDE_FROM_ALL
)


# =============================================================================
# Build the example executable
# =============================================================================

add_executable(biased_vs_unbiased main.cpp)
target_link_libraries(biased_vs_unbiased PRIVATE minimax_cpppy)
