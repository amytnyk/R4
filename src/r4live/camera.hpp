#ifndef R4_CAMERA_HPP
#define R4_CAMERA_HPP

#include "ray.hpp"
#include "view.hpp"

#define PI 3.14159265359

template<typename VecType>
class Camera {
public:
    using value = VecType::value_type;
    using vec_type = VecType;

    value size_y{};
    value size_x{};
    value size_z{};

    __host__ __device__ Camera(const View<VecType> &view,
                               double resolution[3]) : view{view} {
        VecType los = view.direction.direction().unit();

        VecType temp1[3] = {view.over, view.up, los};
        GridX = cross(Array<VecType, 3>{temp1}).unit();
        VecType temp2[3] = {GridX, los, view.over};
        GridY = cross(Array<VecType, 3>{temp2}).unit();
        VecType temp3[3] = {GridX, GridY, los};
        GridZ = cross(Array<VecType, 3>{temp3}).unit();

        size_x = 2 * view.direction.direction().norm() *
                 std::tan(view.angle / 2. * PI / 180);
        size_y = size_x * resolution[1] / resolution[0];
        size_z = size_x * resolution[2] / resolution[0];

        GridX *= size_x;
        GridY *= size_y;
        GridZ *= size_z;
        GridO = view.direction.at(1) - (GridX + GridY + GridZ) / 2.;

        GridX /= resolution[0];
        GridY /= resolution[1];
        GridZ /= resolution[2];
        GridO += (GridX + GridY + GridZ) / 2.;
    }

    __host__ __device__ inline auto ray_to(const Vec3d &relative_direction) const {
        auto origin = view.direction.origin();
        auto direction = GridO +
                         GridX * relative_direction.x() +
                         GridY * relative_direction.y() +
                         GridZ * relative_direction.z() -
                         origin;
        return Ray<Vec4d>{origin, direction};
    }

//private:
    View<VecType> view;
    VecType GridX;
    VecType GridY;
    VecType GridZ;
    VecType GridO;
};

#endif // R4_CAMERA_HPP
