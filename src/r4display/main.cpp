#include <GLFW/glfw3.h>

#include <fstream>
#include <iostream>
#include <string>
#include <fmt/format.h>
#include <ray4/r4_image.h>

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

struct Image {
    explicit Image(const std::string &path, bool lazy_loading = false) : in{path}, lazy_loading{lazy_loading} {
        if (!in.is_open())
            throw std::runtime_error(fmt::format("cannot open input file \"{}\"", path));
        in.read(reinterpret_cast<char *>(&image_header), sizeof(image_header));
        if (image_header.magic != R4_IMAGE_ID)
            throw std::runtime_error("given input file is not in the r4 format");
        if (image_header.version != 1)
            throw std::runtime_error(fmt::format("only version 1 is supported (given {})",
                                                 image_header.version));
        if (image_header.bitsperpixel != 24)
            throw std::runtime_error(fmt::format("only 24 bits per pixel are supported (given {})",
                                                 image_header.bitsperpixel));
        if (image_header.aspect[0] != 1 || image_header.aspect[1] != 1 || image_header.aspect[2] != 1)
            throw std::runtime_error(fmt::format("aspect size (1, 1, 1) is required (given {}, {}, {})",
                                                 image_header.aspect[0],
                                                 image_header.aspect[1],
                                                 image_header.aspect[2]));
        width = image_header.last[0] - image_header.first[0] + 1;
        height = image_header.last[1] - image_header.first[1] + 1;
        depth = image_header.last[2] - image_header.first[2] + 1;

        if (lazy_loading)
            buffer = nullptr;
        else {
            buffer = new GLubyte[width * height * depth * 3];
            in.read(reinterpret_cast<char *>(buffer), width * height * depth * 3);
        }
    }

    [[nodiscard]] int getWidth() const {
        return width;
    }

    [[nodiscard]] int getHeight() const {
        return height;
    }

    [[nodiscard]] int getDepth() const {
        return depth;
    }

    [[nodiscard]] const GLubyte *slice(unsigned plane) {
        if (plane >= depth)
            throw std::runtime_error(fmt::format("slice index error ({} >= {})", plane, depth));
        if (lazy_loading) {
            delete[] buffer;
            buffer = new GLubyte[width * height * 3];
            in.seekg(32 + width * height * plane * 3, std::ios_base::beg);
            in.read(reinterpret_cast<char *>(buffer), width * height * 3);
            return buffer;
        } else
            return buffer + width * height * plane * 3;
    }

    ~Image() {
        delete[] buffer;
    }

private:
    std::ifstream in;
    int width, height, depth;
    ImageHdr image_header{};
    GLubyte *buffer;
    bool lazy_loading;
};

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
        GLFWwindow *window = glfwCreateWindow(1024, 1024, "Render", nullptr, nullptr);
        glfwMakeContextCurrent(window);
        glfwSetKeyCallback(window, key_callback);

        while (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT);
            std::cout << "\r" << current_plane << std::flush;

            int width, height;
            glfwGetWindowSize(window, &width, &height);
            glPixelZoom((float) width / (float) image.getWidth(), (float) height / (float) image.getHeight());
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
