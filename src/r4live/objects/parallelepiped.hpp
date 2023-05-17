#ifndef R4_PARALLELEPIPED_HPP
#define R4_PARALLELEPIPED_HPP

#include "../array.hpp"
#include "../vec.hpp"

template<typename VecType>
class Parallelepiped : public Entity<VecType> {
    using Entity<VecType>::m_material;
public:
    __host__ __device__ explicit Parallelepiped(const Array<Vec4d, 4> &points,
                                                const Material &material)
            : points{points}, Entity<VecType>(material) {}

    __host__ __device__ bool hit(const Ray<VecType> &ray,
                                 Hit<VecType> &hit_record) const override {
        Vec4d temp[3] = {points[0] - points[1],
                         points[0] - points[2],
                         points[0] - points[3]};
        Array<Vec4d, 3> vectors(temp);
        auto space_norm = cross(vectors);
        space_norm = space_norm.unit();
        auto dist = space_norm.dot(points[0]);
        if (space_norm.dot(ray.direction()) == 0) {
            return false;
        }
        auto time = -(space_norm.dot(ray.origin()) + dist) /
                    space_norm.dot(ray.direction());
        if (time < 0)
            return false;
        auto intersect = ray.origin() + ray.direction() * time;
        auto dominant1 = (std::abs(space_norm[0]) > std::abs(space_norm[1])) ? 0 : 1;
        auto dominant2 = (std::abs(space_norm[2]) > std::abs(space_norm[3])) ? 2 : 3;
        size_t ax1, ax2, ax3;
        if (std::abs(space_norm[dominant1]) > std::abs(space_norm[dominant2])) {
            ax1 = (dominant1 == 0) ? 1 : 0;
            ax2 = 2;
            ax3 = 3;
        } else {
            ax1 = 0;
            ax2 = 1;
            ax3 = (dominant2 == 2) ? 3 : 2;
        }
        auto A11 = temp[0][ax1];
        auto A12 = temp[0][ax2];
        auto A13 = temp[0][ax3];

        auto A21 = temp[1][ax1];
        auto A22 = temp[1][ax2];
        auto A23 = temp[1][ax3];

        auto A31 = temp[2][ax1];
        auto A32 = temp[2][ax2];
        auto A33 = temp[2][ax3];

        auto b1 = intersect[ax1] - points[0][ax1];
        auto b2 = intersect[ax2] - points[0][ax2];
        auto b3 = intersect[ax3] - points[0][ax3];

        auto determinant_2233_2332 = (A22 * A33) - (A23 * A32);
        auto determinant_b233_b332 = (b2 * A33) - (b3 * A32);
        auto determinant_12b3_13b2 = (A12 * b3) - (A13 * b2);
        auto determinant_1233_1332 = (A12 * A33) - (A13 * A32);
        auto determinant_1223_1322 = (A12 * A23) - (A13 * A22);
        auto determinant_b223_b322 = (b2 * A23) - (b3 * A22);

        auto cramer_div = A11 * (A22*A33 - A23*A32)
                    - A21 * (A12*A33 - A13*A32)
                    + A31 * (A12*A23 - A13*A22);

        auto x = ((b1 * determinant_2233_2332)
               - (A21 * determinant_b233_b332)
               + (A31 * determinant_b223_b322)) / cramer_div;

        auto y = ((A11 * determinant_b233_b332 )
               - (b1 * determinant_1233_1332)
               + (A31 * determinant_12b3_13b2)) / cramer_div;

        auto z = (-(A11 * determinant_b223_b322)
               - (A21 * determinant_12b3_13b2)
               + (b1 * determinant_1223_1322)) / cramer_div;

        if ((x < 0) || (x > 1) || (y < 0) || (y > 1) || (z < 0) || (z > 1))
            return false;

        hit_record.normal = space_norm;
        hit_record.material = &m_material;
        hit_record.t = time;
        hit_record.point = intersect;
        return true;
    }

private:
    Array<Vec4d, 4> points;
};

#endif //R4_PARALLELEPIPED_HPP
