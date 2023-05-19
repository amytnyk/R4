#include "scene.hpp"
#include "vec.hpp"
#include <iostream>
#include <GLFW/glfw3.h>

int cur_plane = 0;
double y_pos;
double x_pos;
double z_pos;

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(window, true);
        else {
            if (key == GLFW_KEY_LEFT)
                --cur_plane;
            else if (key == GLFW_KEY_RIGHT)
                ++cur_plane;
            else if (key == GLFW_KEY_Q)
                ++y_pos;
            else if (key == GLFW_KEY_E)
                --y_pos;
            else if (key == GLFW_KEY_W)
                ++x_pos;
            else if (key == GLFW_KEY_S)
                --x_pos;
            else if (key == GLFW_KEY_A)
                ++z_pos;
            else if (key == GLFW_KEY_D)
                --z_pos;
        }
    }
}

int main(int argc, char **argv) {
    auto path = std::string(argv[1]);
    Scene<Vec4d> scene{path};
    x_pos = scene.world.view.direction.origin().x();
    y_pos = scene.world.view.direction.origin().y();
    z_pos = scene.world.view.direction.origin().z();
    int width = 700;
    int height = 700;
    int depth = 1;
    double res[3] = {static_cast<double>(width),
                     static_cast<double>(height),
                     static_cast<double>(depth)};

    auto *picture = new GLubyte[width * height * 3];


    glfwInit();
    GLFWwindow *window = glfwCreateWindow(width, height, "Render", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        scene.world.view.direction.origin().x() = x_pos;
        scene.world.view.direction.origin().y() = y_pos;
        scene.world.view.direction.origin().z() = z_pos;
        Camera<Vec4d> camera{scene.world.view,
                             res};
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                double a[] = {static_cast<double>(x),
                              static_cast<double>(y),
                              static_cast<double>(cur_plane)};

                if (x == 180 && y == 150)
                    a[0] = a[0];
                Color color = scene.rayTrace(Vec3d{a}, camera);
                picture[(y * width + x) * 3] = std::clamp((int) (256 * color.x()), 0, 255);
                picture[(y * width + x) * 3 + 1] = std::clamp((int) (256 * color.y()), 0, 255);
                picture[(y * width + x) * 3 + 2] = std::clamp((int) (256 * color.z()), 0, 255);
            }
        }

        glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, picture);
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}
