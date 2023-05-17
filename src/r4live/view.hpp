#ifndef R4_VIEW_HPP
#define R4_VIEW_HPP

#include "ray.hpp"

template<typename VecType>
struct View {
    Ray<VecType> direction;
    VecType up;
    VecType over;
    VecType::value_type angle;
};

#endif //R4_VIEW_HPP
