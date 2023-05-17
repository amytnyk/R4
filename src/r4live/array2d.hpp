#ifndef R4_ARRAY2D_HPP
#define R4_ARRAY2D_HPP

template<typename T, size_t N, size_t M>
class Array2D {
public:
    using value_type = T;
    static constexpr size_t rows = N;
    static constexpr size_t cols = M;

    Array2D() = default;

    __host__ __device__ Array2D(const Array2D<T, N, M> &array) : Array2D{array.data} {}

    __host__ __device__ explicit Array2D(const T array[N][M]) {
        memcpy_s(data, sizeof(T) * N * M, array, sizeof(T) * N * M);
    }

    template<size_t I, size_t J>
    __host__ __device__ inline auto get() const requires(I < N && J < M) {
        return data[I][J];
    }

    __host__ __device__ inline auto get(size_t row, size_t col) const {
        return data[row][col];
    }

    template<size_t I, size_t J>
    __host__ __device__ inline auto &get() requires(I < N && J < M) {
        return data[I][J];
    }

    __host__ __device__ inline auto &get(size_t row, size_t col) {
        return data[row][col];
    }

protected:
    T data[N][M];
};

#endif // R4_ARRAY2D_HPP
