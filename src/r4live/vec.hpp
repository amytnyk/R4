#ifndef R4_VEC_HPP
#define R4_VEC_HPP

#include <cstddef>
#include <type_traits>
#include <cfloat>
#include <initializer_list>
#include <cmath>
#include "array.hpp"

template<typename T, size_t N>
class Vec : public Array<T, N> {
public:
    static constexpr size_t dim = N;

    using Array<T, N>::Array;

    __host__ __device__ inline auto norm_squared() const {
        T sum{};
        for (auto &value: Array<T, N>::data)
            sum += value * value;
        return sum;
    }

    __host__ __device__ inline auto norm() const {
        return std::sqrt(norm_squared());
    }

    __host__ __device__ inline auto unit() const {
        auto length = norm();

        Vec<T, N> vec;
        for (size_t i = 0; i < N; ++i)
            vec[i] = Array<T, N>::data[i] / length;
        return vec;
    }

    __host__ __device__ inline auto &operator-=(const Vec<T, N> &rhs) {
        return *this += -rhs;
    }

    __host__ __device__ inline auto &operator+=(const Vec<T, N> &rhs) {
        for (size_t i = 0; i < N; ++i)
            Array<T, N>::data[i] += rhs[i];
        return *this;
    }

    __host__ __device__ inline auto &operator*=(T k) {
        for (auto &value: Array<T, N>::data)
            value *= k;
        return *this;
    }

    __host__ __device__ inline auto &operator*=(const Vec<T, N> &rhs) {
        for (size_t i = 0; i < N; ++i)
            Array<T, N>::data[i] *= rhs[i];
        return *this;
    }

    __host__ __device__ inline auto &operator/=(T k) {
        return *this *= T{1} / k;
    }

    __host__ __device__ inline auto operator-() const {
        Vec<T, N> arr;
        for (size_t i = 0; i < N; ++i)
            arr[i] = -Array<T, N>::data[i];
        return arr;
    }

    __host__ __device__ inline auto dot(const Vec<T, N> &rhs) const {
        T cross{};
        for (size_t i = 0; i < N; ++i)
            cross += Array<T, N>::data[i] * rhs[i];
        return cross;
    }

    __host__ __device__ inline auto x() const {
        return Array<T, N>::template get<0>();
    }

    __host__ __device__ inline auto &x() {
        return Array<T, N>::template get<0>();
    }

    __host__ __device__ inline auto y() const {
        return Array<T, N>::template get<1>();
    }

    __host__ __device__ inline auto &y() {
        return Array<T, N>::template get<1>();
    }

    __host__ __device__ inline auto z() const {
        return Array<T, N>::template get<2>();
    }

    __host__ __device__ inline auto &z() {
        return Array<T, N>::template get<2>();
    }

    __host__ __device__ inline auto w() const {
        return Array<T, N>::template get<3>();
    }

    __host__ __device__ inline auto &w() {
        return Array<T, N>::template get<3>();
    }

    __host__ __device__ inline bool present() const {
        return FLT_EPSILON <= norm_squared();
    }
};

template<typename T, size_t N>
__host__ std::istream &operator>>(std::istream &stream, Vec<T, N> &vec) {
    for (size_t i = 0; i < N; ++i)
        stream >> vec[i];
    return stream;
}

template<typename T, size_t N>
__host__ __device__ inline auto calculate_determinant(const Array<Vec<T, N>, N - 1> &vectors,
                                                      Array<bool, N> &used,
                                                      size_t size = N - 1) {
    if (size == 0)
        return T{1};
    T result{};
    size_t count{};
    for (size_t cur_pos = 0; cur_pos < N; ++cur_pos) {
        if (!used[cur_pos]) {
            used[cur_pos] = true;
            T cur_value{vectors[N - 1 - size][cur_pos]};
            cur_value *= calculate_determinant(vectors, used, size - 1);
            used[cur_pos] = false;
            if (count % 2 == 0)
                result += cur_value;
            else
                result -= cur_value;
            ++count;
        }
    }
    return result;
}

template<typename VecType>
__host__ __device__ inline auto cross(const Array<VecType, VecType::dim - 1> &vectors) {
    VecType result;
    Vec<bool, VecType::dim> used(false);
    for (size_t cur_pos = 0; cur_pos < VecType::dim; ++cur_pos) {
        used[cur_pos] = true;
        auto minor = calculate_determinant(vectors, used);
        if (cur_pos % 2 == 0)
            result[cur_pos] = minor;
        else
            result[cur_pos] = -minor;
        used[cur_pos] = false;
    }
    return result;
}

template<typename T, size_t N>
__host__ __device__ inline auto operator*(int k, Vec<T, N> rhs) {
    return rhs *= k;
}

template<typename T, size_t N>
__host__ __device__ inline auto operator*(Vec<T, N> lhs, T k) {
    return lhs *= k;
}

template<typename T, size_t N>
__host__ __device__ inline auto operator/(Vec<T, N> lhs, T k) {
    return lhs /= k;
}

template<typename T, size_t N>
__host__ __device__ inline auto operator*(Vec<T, N> lhs, const Vec<T, N> &rhs) {
    return lhs *= rhs;
}

template<typename T, size_t N>
__host__ __device__ inline auto operator+(Vec<T, N> lhs, const Vec<T, N> &rhs) {
    return lhs += rhs;
}

template<typename T, size_t N>
__host__ __device__ inline auto operator-(Vec<T, N> lhs, const Vec<T, N> &rhs) {
    return lhs -= rhs;
}

using Vec2f = Vec<float, 2>;
using Vec2d = Vec<double, 2>;
using Vec2i = Vec<int, 2>;
using Vec2 = Vec2d;

using Vec3f = Vec<float, 3>;
using Vec3d = Vec<double, 3>;
using Vec3i = Vec<int, 3>;
using Vec3 = Vec3d;

using Vec4f = Vec<float, 4>;
using Vec4d = Vec<double, 4>;
using Vec4i = Vec<int, 4>;
using Vec4 = Vec4d;

#endif // R4_VEC_HPP
