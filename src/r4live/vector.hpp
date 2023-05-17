#ifndef R4_VECTOR_HPP
#define R4_VECTOR_HPP

#include <utility>

template<typename T>
class Vector {
public:
    using value_type = T;

    __host__ __device__ explicit Vector(size_t capacity = 4) : size_m{0}, capacity{capacity} {
        data = new T[capacity];
    }

    __host__ __device__ auto &operator[](size_t idx) {
        return data[idx];
    }

    __host__ __device__ auto &operator[](size_t idx) const {
        return data[idx];
    }

    __host__ __device__ inline auto size() const {
        return size_m;
    }

    __host__ __device__ inline void push(const T &item) {
        reserve(size_m + 1);

        data[size_m++] = item;
    }

    __host__ __device__ inline void push(T &&item) {
        reserve(size_m + 1);

        data[size_m++] = std::move(item);
    }

    __host__ __device__ void reserve(size_t new_capacity) {
        if (new_capacity <= capacity)
            return;

        while (capacity < new_capacity)
            capacity *= 2;

        auto old_data = data;
        data = new T[capacity];

        memcpy_s(data, size_m * sizeof(T), old_data, size_m * sizeof(T));

        delete[] old_data;
    }

    __host__ __device__ ~Vector() {
        delete[] data;
    }

private:
    T *data;

    size_t size_m;
    size_t capacity;
};

#endif // R4_VECTOR_HPP
