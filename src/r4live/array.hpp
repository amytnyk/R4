#ifndef R4_ARRAY_HPP
#define R4_ARRAY_HPP

template<typename T, size_t N>
class Array {
public:
    using value_type = T;
    static constexpr size_t size = N;

    Array() = default;

    __host__ __device__ Array(const Array<T, N> &array) : Array{array.data} {}

    __host__ __device__ explicit Array(T fill_value) requires (sizeof(T) == 1) {
        memset(data, fill_value, N);
    }

    __host__ __device__ explicit Array(const T array[N]) {
        memcpy(data, sizeof(T) * N, array, sizeof(T) * N);
    }

    template<size_t I>
    __host__ __device__ inline auto get() const requires(I < N) {
        return data[I];
    }

    __host__ __device__ inline auto get(size_t idx) const {
        return data[idx];
    }

    __host__ __device__ inline auto operator[](size_t idx) const {
        return data[idx];
    }

    template<size_t I>
    __host__ __device__ inline auto &get() requires(I < N) {
        return data[I];
    }

    __host__ __device__ inline auto &get(size_t idx) {
        return data[idx];
    }

    __host__ __device__ inline auto &operator[](size_t idx) {
        return data[idx];
    }

protected:
    T data[N];
};

#endif // R4_ARRAY_HPP
