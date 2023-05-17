#ifndef R4_CAMERA_HPP
#define R4_CAMERA_HPP

#include "ray.hpp"

#define PI 3.14159265359

template<typename VecType>
class Camera {
public:
    using vec_type = VecType;

    __host__ __device__ Camera(const View<VecType>& view, double aspect_ratio) {
        float theta = view.fov * ((float) PI) / 180.0f;

        left_top = view.origin();
    }

    __host__ __device__ inline auto ray_to(size_t x, size_t y, size_t z) const {
        return Ray<VecType>{left_top,
                            horizontal * static_cast<VecType::value_type>(x) +
                            vertical * static_cast<VecType::value_type>(y)};
    }

private:
    VecType left_top;
    VecType horizontal;
    VecType vertical;
};

#endif // R4_CAMERA_HPP
