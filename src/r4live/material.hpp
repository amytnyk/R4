#ifndef R4_MATERIAL_HPP
#define R4_MATERIAL_HPP

#include "color.hpp"

struct Material {
    Color ambient{};
    Color diffuse{};
    Color specular{};
    Color transparent{};
    double shine{};
    bool reflect{};
};

#endif //R4_MATERIAL_HPP
