set(TARGET_NAME ray4)

add_executable(${TARGET_NAME}
        r4_globals.h
        r4_image.h
        ray4.h
        vector.h
        r4_main.c
        r4_hit.c
        r4_io.c
        r4_parse.c
        r4_trace.c
        r4_vec.c)

target_link_libraries(${TARGET_NAME} m)

set(TARGETS ${TARGETS} ${TARGET_NAME} PARENT_SCOPE)
