#ifndef R4_LIGHT_HPP
#define R4_LIGHT_HPP

#include "color.hpp"

template<typename VecType>
struct Light {
    enum Type {
        DIRECTIONAL,
        POSITIONAL
    } type;

    Color color;
    union {
        VecType direction;
        VecType position;
    };
};

#endif //R4_LIGHT_HPP
