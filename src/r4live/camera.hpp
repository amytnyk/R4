#ifndef R4_CAMERA_HPP
#define R4_CAMERA_HPP

#include "ray.hpp"

#define PI 3.14159265359

template<typename VecType>
class Camera {
public:
    using vec_type = VecType;

    __host__ __device__ explicit Camera(const View<VecType>& view, const RayGrid<VecType>& ray_grid): view{view} {
        VecType los = view.direction.direction().normalized();

        VecType gz = cross(Array<VecType>{view.over, view.up, los}).normalized();
        VecType gy = cross(gz, los, view.over).normalized();

        gnx = 2 * view.direction.direction().norm() * std::tan(view.angle / 2 * PI / 180);

        gorigin = to - (gx + gy + gz) / 2 + (gx / res.x() + gy / res.y() + gz / res.z()) / 2;
    }

    __host__ __device__ inline auto ray_to(const VecType& relative_direction) const {
        return Ray<VecType>{};
    }

private:
    View<VecType> view;
};

#endif // R4_CAMERA_HPP
