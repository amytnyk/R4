#ifndef R4_PARALLELPIPED_HPP
#define R4_PARALLELPIPED_HPP

#include "../array.hpp"
#include "../vec.hpp"

template<typename VecType>
class Parallelpiped : public Entity<VecType> {
    __host__ __device__ Parallelpiped (const Array<Vec4d, 4> &points)
        : points{points} {}

    __host__ __device__ bool hit(const Ray<VecType> &ray,
                                 Hit<VecType> &hit_record) const override {
        
    }

private:
    Array<Vec4d, 4> points;
};

#endif //R4_PARALLELPIPED_HPP
