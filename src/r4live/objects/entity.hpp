#ifndef R4_ENTITY_HPP
#define R4_ENTITY_HPP

#include "../ray.hpp"
#include "../material.hpp"
#include <nvfunctional>

class material;

template<typename VecType>
struct Hit {
    using vec_type = VecType;

    Material *material{};

    VecType::value_type t{};
    VecType point{};
    VecType normal{};
};

template<typename VecType>
class Entity {
public:
    using vec_type = VecType;

    __host__ __device__ virtual bool hit(const Ray<VecType> &ray,
                                         Hit<VecType> &hit_record) const = 0;

    __host__ __device__ virtual bool call_for_hits(
            const Ray<VecType> &ray,
            const nvstd::function<void(Entity<VecType>&)> &lambda) const {
        if (hit(ray, {}))
            lambda(this);
    }

    __host__ __device__ Material& material() {
        return m_material;
    }

    __host__ __device__ Material material() const {
        return m_material;
    }

protected:
    Material m_material{};
};

#endif // R4_ENTITY_HPP
