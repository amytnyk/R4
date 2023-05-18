#include "entity.hpp"

#ifndef R4_SPHERE_HPP
#define R4_SPHERE_HPP

template<typename VecType>
class Sphere : public Entity<VecType> {
    using Entity<VecType>::m_material;
public:

    __host__ __device__ Sphere(const VecType &center,
                               VecType::value_type radius,
                               const Material &material)
        : center{center}, radius{radius}, Entity<VecType>(material) {}

    __host__ __device__ bool hit(const Ray<VecType> &ray,
                                  Hit<VecType> &hit_record) const override {
        auto cdir = center - ray.origin();
        auto bb = cdir.dot(ray.direction().unit());
        auto rad = bb * bb - cdir.dot(cdir) + radius * radius;

        if (rad <= 0.0)
            return false;

        rad = std::sqrt(rad);
        auto t1 = bb + rad;
        auto t2 = bb - rad;

        if ((t1 < 0.0) || ((t2 > 0.0) && (t2 < t1)))
            t1 = t2;

        if (t1 <= 0.0)
            return false;

        hit_record.t = t1 / ray.direction().norm();
        hit_record.point = ray.at(hit_record.t);
        hit_record.normal = (hit_record.point - center) / radius;
        hit_record.material = &m_material;

        return true;
    }

private:
    VecType center;
    VecType::value_type radius;
};

#endif //R4_SPHERE_HPP
