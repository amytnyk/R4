#ifndef R4_ENTITY_HPP
#define R4_ENTITY_HPP

#include "../ray.hpp"
#include "../material.hpp"
#include <nvfunctional>
#include <utility>
#include <iostream>

class material;

template<typename VecType>
struct Hit {
    using vec_type = VecType;

    const Material *material{};

    VecType::value_type t{std::numeric_limits<typename VecType::value_type>::max()};
    VecType point{};
    VecType normal{};
};

template<typename VecType>
class Entity {
public:
    using vec_type = VecType;

    Entity() = default;

    __host__ __device__ explicit Entity(Material material)
            : m_material(std::move(material)) {}

    __host__ __device__ Entity(const Entity &entity) = delete;
    __host__ __device__ Entity& operator=(const Entity &entity) = delete;

    __host__ __device__ virtual bool hit(const Ray<VecType> &ray,
                                         Hit<VecType> &hit_record) const = 0;

    __host__ __device__ virtual void call_for_hits(
            const Ray<VecType> &ray,
            const nvstd::function<void(const Entity<VecType> &)> &lambda) const {
        Hit<VecType> temp_hit{};
        if (hit(ray, temp_hit))
            lambda(*this);
    }

    __host__ __device__ Material &material() {
        return m_material;
    }

    __host__ __device__ Material material() const {
        return m_material;
    }

protected:
    Material m_material{};
};

#endif // R4_ENTITY_HPP
