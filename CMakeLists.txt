cmake_minimum_required(VERSION 3.22)

project(R4 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src/r4live)

foreach (target ${TARGETS})
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src)
endforeach ()

set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/bin)
set_target_properties(${TARGETS} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR})
