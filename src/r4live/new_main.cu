#include "scene.hpp"
#include "vec.hpp"
#include <iostream>
#include <GLFW/glfw3.h>

int main() {
    auto path = std::string("inputs.r4/sphere.r4");
    Scene<Vec4d> scene{path};
    int width = 108;
    int height = 72;
    int depth = 72;
    double res[3] = {static_cast<double>(width),
                     static_cast<double>(height),
                     static_cast<double>(depth)};
    Camera<Vec4d> camera{scene.world.view,
                         res};

    auto* picture = new GLubyte[width * height * 3];
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            double a[] = {camera.size_x / static_cast<double>(width) * static_cast<double>(x),
                          camera.size_y / static_cast<double>(height) * static_cast<double>( y),
                          5};
            Color color = scene.rayTrace(Vec3d{a}, camera);
            picture[(y * width + x) * 3] = std::clamp((int) (256 * color.x()), 0, 255);
            picture[(y * width + x) * 3 + 1] = std::clamp((int) (256 * color.y()), 0, 255);
            picture[(y * width + x) * 3 + 2] = std::clamp((int) (256 * color.z()), 0, 255);
        }
    }

    glfwInit();
    GLFWwindow *window = glfwCreateWindow(width, height, "Render", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, picture);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}
