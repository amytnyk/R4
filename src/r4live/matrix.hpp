#ifndef R4_MATRIX_HPP
#define R4_MATRIX_HPP

#include "array2d.hpp"

template<typename T, size_t N, size_t M>
class Mat {
public:
    using value_type = T;
    static constexpr size_t rows = N;
    static constexpr size_t cols = M;

    Mat() = default;

    explicit Mat(const Array2D<T, N, M> &container) : values{container} {}

    explicit Mat(std::array<std::array<T, M>, N> &&container) : values{std::move(container)} {}

    auto &operator+=(const Mat<T, N, M> &rhs) {
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < M; ++j)
                values[i][j] += rhs[i][j];
        return *this;
    }

    auto &operator-=(const Mat<T, N, M> &rhs) {
        return *this += -rhs;
    }

    auto operator-() const {
        return *this *= -1;
    }

    auto &operator*=(T k) {
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < M; ++j)
                values[i][j] *= k;
        return *this;
    }

    auto &operator/=(T k) {
        return *this *= T{1} / k;
    }

    auto &operator*=(const Mat<T, M, M> &rhs) {
        Mat<T, N, M> old{*this};
        values.fill(std::array<T, M>{});
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < M; ++j)
                for (size_t k = 0; k < M; ++k)
                    values[i][j] += old[i][j + k] * rhs[i + k][j];
        return *this;
    }

    auto &transpose() requires (N == M) {
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < i; ++j)
                std::swap(values[i][j], values[j][i]);
        return *this;
    }

    auto transposed() const {
        Mat<T, M, N> mat;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                mat[i][j] = values[j][i];
        return mat;
    }

    static auto diagonal(const std::array<T, N> &arr) requires (N == M) {
        Mat<T, N, M> mat;
        for (size_t i = 0; i < N; ++i)
            mat.get(i, i) = arr[i];
        return mat;
    }

    static auto diagonal(T value) {
        std::array<T, N> arr;
        arr.fill(value);
        return diagonal(arr);
    }

    static auto identity() {
        return diagonal(T{1});
    }

    auto &get(size_t row, size_t col) {
        return values[row][col];
    }

    auto get(size_t row, size_t col) const {
        return values[row][col];
    }

private:
    std::array<std::array<T, M>, N> values;
};


#endif //R4_MATRIX_HPP
