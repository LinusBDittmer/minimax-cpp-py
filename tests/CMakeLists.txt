cmake_minimum_required(VERSION 3.18)

function(minimax_add_test name)
    cmake_parse_arguments(ARG "INTERNAL" "" "" ${ARGN})
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(ARG_INTERNAL)
        target_include_directories(${name} PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/core
        )
    endif()
    target_link_libraries(${name} PRIVATE minimax_cpppy)
    target_compile_options(${name} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

minimax_add_test(test_minimax_small)
minimax_add_test(test_minimax_molecular)
minimax_add_test(test_minimax_r10)
minimax_add_test(test_run_banner)
minimax_add_test(test_minimax_lp INTERNAL)

minimax_add_test(test_ext_table INTERNAL)
minimax_add_test(test_lookup_nonuniform INTERNAL)
minimax_add_test(test_grid INTERNAL)
minimax_add_test(test_dd128 INTERNAL)
minimax_add_test(test_quadrature INTERNAL)
minimax_add_test(test_ln_loss INTERNAL)
minimax_add_test(test_newton INTERNAL)
minimax_add_test(test_log_pretty INTERNAL)
minimax_add_test(test_log_derivatives INTERNAL)
minimax_add_test(test_ifft INTERNAL)
minimax_add_test(test_density)
minimax_add_test(test_biased_minimax INTERNAL)

# Build the executable without registering a monolithic test entry.
add_executable(test_table_regression test_table_regression.cpp)
target_include_directories(test_table_regression PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
)
target_link_libraries(test_table_regression PRIVATE minimax_cpppy)
target_compile_options(test_table_regression PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>)
target_compile_definitions(test_table_regression PRIVATE
    MINIMAX_REF_CSV="${CMAKE_SOURCE_DIR}/tests/data/reference_outputs.csv")
# One test per nlap. Fixture is subsampled (~40 lr points/nlap, was ~400) so
# even nlap=30 runs in a few seconds; no lr-band splitting needed.
foreach(n RANGE 1 30)
    add_test(NAME test_table_regression_nlap_${n}
             COMMAND test_table_regression ${n} ${n})
endforeach()

# Debug-only counter test: compiled in all configurations but skipped at
# runtime when MINIMAX_CPPPY_DEBUG_MODE__ is not defined (Release build).
# The source uses #ifndef MINIMAX_CPPPY_DEBUG_MODE__ to return early rather
# than #error, so it compiles cleanly in Release.
minimax_add_test(test_debug_counter)
set_tests_properties(test_debug_counter PROPERTIES
    DISABLED $<NOT:$<CONFIG:Debug>>
)
