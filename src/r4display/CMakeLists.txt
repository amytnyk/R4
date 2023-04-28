set(TARGET_NAME r4display)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(fmt REQUIRED)

add_executable(${TARGET_NAME} main.cpp ../ray4/r4_image.h)

target_include_directories(${TARGET_NAME} PRIVATE ${fmt_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} fmt::fmt)

target_include_directories(${TARGET_NAME} PRIVATE ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} OpenGL::GL glfw)

set(TARGETS ${TARGETS} ${TARGET_NAME} PARENT_SCOPE)
