set(PYBIND11_FINDPYTHON ON)
# pybind11 comes from build-system.requires (pip); scikit-build-core puts its
# CMake config on the prefix path. The third_party submodule is no longer used
# for the Python build, so sdist installs work without git submodules.
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_minimax_core _minimax_core.cpp)
target_link_libraries(_minimax_core PRIVATE minimax_cpppy)

set_target_properties(_minimax_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/minimax_cpppy"
)

add_custom_command(TARGET _minimax_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/minimax_cpppy/__init__.py"
        "${CMAKE_CURRENT_BINARY_DIR}/minimax_cpppy/__init__.py"
    COMMENT "Copying minimax_cpppy/__init__.py to build tree"
)

install(TARGETS _minimax_core DESTINATION minimax_cpppy)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_test(
        NAME pytest
        COMMAND ${Python3_EXECUTABLE} -m pytest "${CMAKE_SOURCE_DIR}/tests/python/" -v
    )
    set_tests_properties(pytest PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    )
endif()
