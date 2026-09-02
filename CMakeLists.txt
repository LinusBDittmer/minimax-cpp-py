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
    add_executable(minimax_loss_func tools/minimax_loss_func.cpp)
    target_include_directories(minimax_loss_func PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_link_libraries(minimax_loss_func PRIVATE minimax_cpppy)
    target_compile_options(minimax_loss_func PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
    )

    add_executable(denominator_density tools/denominator_density.cpp)
    target_include_directories(denominator_density PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_link_libraries(denominator_density PRIVATE minimax_cpppy)
    target_compile_options(denominator_density PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
    )

    add_executable(minimax_lp_loss tools/minimax_lp_loss.cpp)
    target_include_directories(minimax_lp_loss PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_link_libraries(minimax_lp_loss PRIVATE minimax_cpppy)
    target_compile_options(minimax_lp_loss PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
    )

    add_executable(ln_quad_scan tools/ln_quad_scan.cpp)
    target_include_directories(ln_quad_scan PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_link_libraries(ln_quad_scan PRIVATE minimax_cpppy)
    target_compile_options(ln_quad_scan PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
    )

    add_executable(gen_table tools/gen_table.cpp)
    target_include_directories(gen_table PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(gen_table PRIVATE minimax_cpppy)
    target_compile_options(gen_table PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -O2>
    )

    add_executable(dump_reference tools/dump_reference.cpp)
    target_include_directories(dump_reference PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/core ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(dump_reference PRIVATE minimax_cpppy)
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

install(TARGETS minimax_cpppy
    EXPORT  minimax_cpppy_targets
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/minimax_cpppy DESTINATION include)
install(EXPORT minimax_cpppy_targets
    FILE      minimax_cpppy_targets.cmake
    NAMESPACE minimax_cpppy::
    DESTINATION lib/cmake/minimax_cpppy
)
