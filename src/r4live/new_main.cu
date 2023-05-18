#include "scene.hpp"
#include "vec.hpp"
#include <iostream>
#include <GLFW/glfw3.h>

int main(int argc, char **argv) {
    auto path = std::string(argv[1]);
    Scene<Vec4d> scene{path};
    int width = 1080 / 10;
    int height = 720 / 10;
    int depth = 1;
    double res[3] = {static_cast<double>(width),
                     static_cast<double>(height),
                     static_cast<double>(depth)};

    auto *picture = new GLubyte[width * height * 3];


    glfwInit();
    GLFWwindow *window = glfwCreateWindow(width, height, "Render", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    int g = 0;
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        scene.world.view.direction.origin().x() += 0.01;
        Camera<Vec4d> camera{scene.world.view,
                             res};
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                double a[] = {static_cast<double>(x),
                              static_cast<double>(y),
                              0};

                if (x == 540 && y == 360)
                    a[0] = a[0];
                Color color = scene.rayTrace(Vec3d{a}, camera);
                picture[(y * width + x) * 3] = std::clamp((int) (256 * color.x()), 0, 255);
                picture[(y * width + x) * 3 + 1] = std::clamp((int) (256 * color.y()), 0, 255);
                picture[(y * width + x) * 3 + 2] = std::clamp((int) (256 * color.z()), 0, 255);
            }
        }

        glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, picture);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}
