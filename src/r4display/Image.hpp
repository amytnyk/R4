#ifndef R4_IMAGE_HPP
#define R4_IMAGE_HPP

#include <GLFW/glfw3.h>
#include <fstream>
#include <string>
#include <fmt/format.h>
#include <ray4/r4_image.h>

struct Image {
    explicit Image(const std::string &path, bool lazy_loading = false);

    [[nodiscard]] int getWidth() const;

    [[nodiscard]] int getHeight() const;

    [[nodiscard]] int getDepth() const;

    [[nodiscard]] const GLubyte *slice(unsigned plane);

    ~Image();

private:
    std::ifstream in;
    int width, height, depth;
    ImageHdr image_header{};
    GLubyte *buffer;
    bool lazy_loading;
};

#endif //R4_IMAGE_HPP
