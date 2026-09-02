# CMakeLists.txt — standalone build for the denominator_density example
# =======================================================================
#
# Quick start (from this directory):
#
#   cmake -B build -S .
#   cmake --build build
#   ./build/denominator_density

cmake_minimum_required(VERSION 3.18)
project(denominator_density_example LANGUAGES CXX)

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

add_executable(denominator_density main.cpp)

# Linking against minimax_cpppy provides:
#   - minimax_cpppy/minimax.hpp  (laplaceMinimax)
#   - minimax_cpppy/biasing.hpp  (DenominatorDensity, biasedLaplace)
target_link_libraries(denominator_density PRIVATE minimax_cpppy)
