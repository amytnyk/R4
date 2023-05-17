set(TARGET_NAME r4live)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(fmt REQUIRED)

add_executable(${TARGET_NAME}
#        main.cu
        camera.hpp
        objects/entity.hpp
        objects/triangle.hpp
        objects/entity_list.hpp
        objects/sphere.hpp
        ray.hpp
        vec.hpp
        vector.hpp
        array.hpp
        array2d.hpp
        matrix.hpp
        scene.hpp
        color.hpp
        material.hpp
        objects/triangle.hpp
        view.hpp
        light.hpp
        objects/parallelepiped.hpp
        objects/tetrahedron.hpp
        world.hpp
        new_main.cu)

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_include_directories(${TARGET_NAME} PRIVATE ${fmt_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} fmt::fmt)

target_include_directories(${TARGET_NAME} PRIVATE ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} OpenGL::GL glfw)

set(TARGETS ${TARGETS} ${TARGET_NAME} PARENT_SCOPE)
