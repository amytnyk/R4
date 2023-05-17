cmake_minimum_required(VERSION 3.22)

if (NOT DEFINED BUILD_LIVE_WITH_CUDA)
    set(BUILD_LIVE_WITH_CUDA ON)
endif ()

if (BUILD_LIVE_WITH_CUDA)
    project(R4 LANGUAGES C CXX CUDA)
else ()
    project(R4 LANGUAGES C CXX)
endif ()

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src/ray4)
add_subdirectory(src/r4display)

if (BUILD_LIVE_WITH_CUDA)
    add_subdirectory(src/r4live)
endif ()

foreach (target ${TARGETS})
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src)
endforeach ()

set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/bin)
set_target_properties(${TARGETS} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR})
