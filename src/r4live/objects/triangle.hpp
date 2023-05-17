#ifndef R4_TRIANGLE_HPP
#define R4_TRIANGLE_HPP

#include "entity.hpp"
#include "../array.hpp"

template<typename VecType>
class Triangle : public Entity<VecType> {

    __host__ __device__ Triangle(const VecType &A,
                                 const VecType &B,
                                 const VecType &C) : A{A}, B{B}, C{C} {}

private:
    VecType A;
    VecType B;
    VecType C;
};

#endif //R4_TRIANGLE_HPP
