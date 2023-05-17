#include "Image.hpp"

Image::Image(const std::string &path, bool lazy_loading) : in{path}, lazy_loading{lazy_loading} {
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

[[nodiscard]] int Image::getWidth() const {
    return width;
}

[[nodiscard]] int Image::getHeight() const {
    return height;
}

[[nodiscard]] int Image::getDepth() const {
    return depth;
}

[[nodiscard]] const GLubyte *Image::slice(unsigned plane) {
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

Image::~Image() {
    delete[] buffer;
}