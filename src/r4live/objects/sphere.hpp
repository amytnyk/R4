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
        auto OC = ray.origin() - center;
        auto dir = ray.direction();
        auto a = dir.dot(dir);
        auto b = dir.dot(OC);
        auto c = OC.dot(OC) - radius*radius;
        auto discriminant = b*b - a*c;
        if (discriminant > 0) {
            auto tmp = (-b - sqrt(discriminant)) / a;
            if (tmp > 0) {
                hit_record.t = tmp;
                hit_record.point = ray.at(tmp);
                hit_record.normal = (hit_record.point - center) / radius;
                hit_record.material = &m_material;
                return true;
            }
            tmp = (-b + sqrt(discriminant)) / a;
            if (tmp > 0) {
                hit_record.t = tmp;
                hit_record.point = ray.at(tmp);
                hit_record.normal = (hit_record.point - center) / radius;
                hit_record.material = &m_material;
                return true;
            }
        }
        return false;
    }

private:
    VecType center;
    VecType::value_type radius;
};

#endif //R4_SPHERE_HPP
