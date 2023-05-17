#ifndef R4_ENTITY_LIST_HPP
#define R4_ENTITY_LIST_HPP

#include "entity.hpp"
#include "../ray.hpp"

template<typename VecType>
class EntityList : public Entity<VecType> {
public:
    __host__ __device__ bool hit(const Ray<VecType> &ray,
                                   Hit<VecType> &hit_record) const override {
        bool hit_something = false;
        for (auto &child : children) {
            Hit<VecType> possible_hit_record{};
            if (child.hit(ray, possible_hit_record) &&
                (!hit_something || possible_hit_record.t < hit_record.t)) {
                hit_something = true;
                hit_record = possible_hit_record;
            }
        }
        return hit_something;
    }

    __host__ __device__ void call_for_hits(
            const Ray<VecType> &ray,
            const nvstd::function<void(Entity<VecType>&)> &lambda) const override {
        for (const auto &child : children)
            child.call_for_hits(ray, lambda);
    }

    __host__ __device__ void addChild(Entity<VecType> *child) {
        children.push(child);
    }

private:
    Vector<Entity<VecType> *> children;
};

#endif