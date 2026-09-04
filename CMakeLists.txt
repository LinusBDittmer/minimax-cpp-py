cmake_minimum_required(VERSION 3.18)
project(minimax_cpppy VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB MINIMAX_DATA_SRCS CONFIGURE_DEPENDS src/core/data_ext_*.cpp)

add_library(minimax_cpppy STATIC
    src/laplace.cpp
    src/biasing.cpp
    src/laplace_lp.cpp
    ${MINIMAX_DATA_SRCS}
)
set_target_properties(minimax_cpppy PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(minimax_cpppy
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_compile_options(minimax_cpppy PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

# The DD arithmetic uses std::fma (always IEEE-correct).  Enable -march=native to
# let the compiler contract it to a hardware FMA instruction for the big speedup;
# OFF by default so distributed/portable builds stay target-independent.
option(MINIMAX_CPPPY_NATIVE_ARCH "Optimise for the host CPU (-march=native, enables hardware FMA)" OFF)
if(MINIMAX_CPPPY_NATIVE_ARCH)
    target_compile_options(minimax_cpppy PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>
    )
endif()

target_compile_definitions(minimax_cpppy PUBLIC
    $<$<CONFIG:Debug>:MINIMAX_CPPPY_DEBUG_MODE__>
)

option(MINIMAX_CPPPY_BUILD_TESTS     "Build CTest suite"        OFF)
option(MINIMAX_CPPPY_BUILD_PYTHON    "Build Python bindings"    OFF)
option(MINIMAX_CPPPY_BUILD_TOOLS     "Build diagnostic CLI tools" OFF)
option(MINIMAX_CPPPY_BUILD_DOCS      "Build Doxygen documentation" OFF)

enable_testing()

if(MINIMAX_CPPPY_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(MINIMAX_CPPPY_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(MINIMAX_CPPPY_BUILD_TOOLS)
    foreach(t minimax_loss_func denominator_density laplace_lp_loss ln_quad_scan gen_table dump_reference)
        add_executable(${t} tools/${t}.cpp)
        target_include_directories(${t} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        )
        target_link_libraries(${t} PRIVATE minimax_cpppy)
        target_compile_options(${t} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
        )
    endforeach()
    # gen_table runs the full Remez sweep; keep it optimised even in Debug builds.
    target_compile_options(gen_table PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2>
    )
endif()

if(MINIMAX_CPPPY_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_PROJECT_NAME        "minimax-cpppy")
    set(DOXYGEN_PROJECT_BRIEF       "MINIMAX-optimal Laplace quadrature points for 1/x")
    set(DOXYGEN_OUTPUT_DIRECTORY    "${CMAKE_CURRENT_BINARY_DIR}/docs")
    set(DOXYGEN_EXTRACT_ALL         YES)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
    set(DOXYGEN_GENERATE_HTML       YES)
    set(DOXYGEN_GENERATE_LATEX      NO)
    set(DOXYGEN_QUIET               YES)
    doxygen_add_docs(docs
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        COMMENT "Generating Doxygen documentation"
    )
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS minimax_cpppy
    EXPORT  minimax_cpppy_targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/minimax_cpppy DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT minimax_cpppy_targets
    FILE      minimax_cpppy_targets.cmake
    NAMESPACE minimax_cpppy::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minimax_cpppy
)

configure_package_config_file(
    cmake/minimax_cpppyConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/minimax_cpppyConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minimax_cpppy
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/minimax_cpppyConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/minimax_cpppyConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/minimax_cpppyConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minimax_cpppy
)
