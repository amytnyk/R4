#ifndef R4_RAY_HPP
#define R4_RAY_HPP

#include "vec.hpp"

template<typename VecType>
class Ray {
public:
    using vec_type = VecType;

    Ray() = default;

    __host__ __device__ Ray(const VecType &origin, const VecType &direction) : origin_m{origin_m},
                                                                               direction_m{direction_m} {}

    __host__ __device__ inline auto origin() const {
        return origin_m;
    }

    __host__ __device__ inline auto direction() const {
        return direction_m;
    }

    __host__ __device__ inline auto at(VecType::value_type t) const {
        return origin_m + t * direction_m;
    }

    __host__ __device__ inline auto forward() const {
        return Ray<VecType>{at(1e-10), direction_m};
    }

private:
    VecType origin_m;
    VecType direction_m;
};

#endif // R4_RAY_HPP
