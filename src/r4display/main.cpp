#include <iostream>
#include <string>
#include <fmt/format.h>
#include "Image.hpp"

using namespace std::string_literals;

int current_plane = 0;
int depth;

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(window, true);
        else {
            if (key == GLFW_KEY_LEFT)
                --current_plane;
            else if (key == GLFW_KEY_RIGHT)
                ++current_plane;
            current_plane = std::min(current_plane, depth - 1);
            current_plane = std::max(0, current_plane);
        }
    }
}



int main(int argc, char *argv[]) {
    try {
        if (argc != 2 && argc != 3)
            throw std::runtime_error(fmt::format("usage: r4display <input_file> <--lazy?> (1 or 2 arguments expected, {} given)",
                                                 argc - 1));
        if (argc == 3 && argv[2] != "--lazy"s)
            throw std::runtime_error(fmt::format("unknown argument {} (--lazy expected)", argv[2]));
        Image image{argv[1], argc == 3};
        depth = image.getDepth();

        glfwInit();
        GLFWwindow *window = glfwCreateWindow(image.getWidth(), image.getHeight(), "Render", nullptr, nullptr);
        glfwMakeContextCurrent(window);
        glfwSetKeyCallback(window, key_callback);

        while (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT);
            std::cout << "\r" << current_plane << std::flush;
//            double xpos, ypos;
//            glfwGetCursorPos(window, &xpos, &ypos);
//            std::cout << xpos << " " << image.getHeight() - ypos << std::endl;
            glDrawPixels(image.getWidth(), image.getHeight(), GL_RGB, GL_UNSIGNED_BYTE, image.slice(current_plane));

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
        std::cout << std::endl;

        glfwTerminate();
    } catch (const std::runtime_error& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }
    return 0;
}
