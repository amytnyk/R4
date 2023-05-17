#ifndef R4_COLOR_HPP
#define R4_COLOR_HPP

#include "vec.hpp"

class Color : public Vec3d {
public:
    using Vec3d::Vec;

    [[nodiscard]] __host__ __device__ inline value_type r() const {
        return x();
    }

    __host__ __device__ inline value_type &r() {
        return x();
    }

    [[nodiscard]]__host__ __device__ inline value_type g() const {
        return y();
    }

    __host__ __device__ inline value_type &g() {
        return y();
    }

    [[nodiscard]]__host__ __device__ inline value_type b() const {
        return z();
    }

    __host__ __device__ inline value_type &b() {
        return z();
    }

    __host__ __device__ inline bool present() const {
        return !(-FLT_EPSILON < x() && x() < FLT_EPSILON &&
                 -FLT_EPSILON < y() && y() < FLT_EPSILON &&
                 -FLT_EPSILON < z() && z() < FLT_EPSILON);
    }
};


#endif // R4_COLOR_HPP
