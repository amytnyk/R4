#ifndef R4_TETRAHEDRON_HPP
#define R4_TETRAHEDRON_HPP

#include "entity.hpp"
#include "../array.hpp"
#include "../vec.hpp"

template<typename VecType>
class Tetrahedron : public Entity<VecType> {
    using Entity<VecType>::m_material;
public:
    __host__ __device__ explicit Tetrahedron (const Array<Vec4d, 4> &points,
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
        auto P = ray.at(time);

        auto is_inside =
                same_sides(points[0], points[1], points[2], points[3], P, space_norm) &&
                same_sides(points[1], points[2], points[3], points[0], P, space_norm) &&
                same_sides(points[2], points[3], points[0], points[1], P, space_norm) &&
                same_sides(points[3], points[0], points[1], points[2], P, space_norm);
        if (is_inside) {
            hit_record.normal = space_norm;
            hit_record.material = &m_material;
            hit_record.t = time;
            hit_record.point = P;
            return true;
        }
        return false;
    }

private:
    Array<Vec4d, 4> points;

    __host__ __device__ bool same_sides(const Vec4d &A,
                                        const Vec4d &B,
                                        const Vec4d &C,
                                        const Vec4d &P1,
                                        const Vec4d &P2,
                                        const Vec4d &space_norm) const {
        Vec4d temp[3] = {A - B, A - C, space_norm};
        Array<Vec4d, 3> vectors{temp};
        auto norm = cross(vectors);
        auto dotP1 = norm.dot(A - P1);
        auto dotP2 = norm.dot(A - P2);
        return signbit(dotP1) == signbit(dotP2);
    }
};

#endif //R4_TETRAHEDRON_HPP
