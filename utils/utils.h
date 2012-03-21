// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cutl/blob/master/LICENSE

#ifndef CUTL_UTILS_H
#define CUTL_UTILS_H

#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <boost/shared_ptr.hpp>
#include <vector_types.h>
#include <iostream>
#include <string>
#include <vector_types.h>
#include <cutil_math.h>

// defined outside of namespace, __globla__ functions don't scope
// correctly to global namespaces
typedef unsigned long long UInt64;
struct SFloat4 {
    float3 vec;
    float w;
};

typedef union {
    float4 vec;
    float component[4];
    SFloat4 f3;
} UFloat4;

__host__ __device__
inline UFloat4
make_ufloat4(const float4 & vec) {
    UFloat4 res;
    res.vec = vec;
    return res;
}

__host__ __device__
inline UFloat4
make_ufloat4(float x, float y, float z, float w) {
    UFloat4 res;
    res.vec = make_float4(x,y,z,w);
    return res;
}

__host__ __device__
inline UFloat4
inv_ufloat4(const UFloat4 & vec) {
    return make_ufloat4(1.f/vec.vec.x, 1.f/vec.vec.y, 1.f/vec.vec.z, 1.f/vec.vec.w);
}

__host__ __device__
inline UFloat4
inv_ufloat3(const UFloat4 & vec) {
    return make_ufloat4(1.f/vec.vec.x, 1.f/vec.vec.y, 1.f/vec.vec.z, 0);
}

__host__ __device__
inline UFloat4
finite_ufloat4(const UFloat4 & vec) {
    UFloat4 res;
    for(int i = 0; i < 4; ++i) {
        if(vec.component[i]*vec.component[i] < 1e-16f)
            res.component[i] = 1e-8f;
        else
            res.component[i] = vec.component[i];
    }
    return res;
}

__host__ __device__
inline float3
diff_ufloat3(const UFloat4 & v1, const UFloat4 & v2) {
    return v1.f3.vec - v2.f3.vec;
}

struct UAABB {
    UFloat4 minimum;
    UFloat4 maximum;
};

/**********************************************************************************
 *
 * DevVector
 *
 * Wrapper around thrust::device_vector<T>. A fixed size is allocated
 * during construction, trying to minimize calls to cudaMalloc.
 * Everytime the vector is resized above the available allocated
 * memory, the allocated memory is doubled.
 *
 * TODO: thrust::device_vector<T>::reserve() seems not to call malloc,
 *       otherwise we could simplify this quite a bit
 *
 **********************************************************************************/

template<typename T>
class DevVector {
    public:
        typedef thrust::device_vector<T> type;
        typedef typename type::iterator iterator;

        DevVector(int alloc_size = 1024);

        // Default copy constructor and assignment are ok here. Will
        // copy the device pointers, not the contents
        // DevVector(const DevVector<T> & vec);
        // DevVector<T> operator=(const DevVector<T> & vec);

        int alloc_size() const;
        int size() const;

        // Explicit copy, allocates a new device vector
        DevVector<T> copy();

        iterator begin();
        iterator end();

        // Sets the accessible vector length to zero without
        // deallocating memory or clearing values. Warning: Raw pointer still
        // has access to elements.
        void clear();

        // Clears the allocated memory.
        void free();

        // Returns a raw pointer to device memory to be used in custom
        // kernel calls
        T* pointer();

        // Resize the vector to a given size. If needed, more memory
        // is allocated by doubling the reserved memory. Supports
        // shrinking, however, raw pointer still has access to the
        // elements.
        void resize(int size);

        // Same as above, but initialize new elements to the given
        // value. Shrinking here is illegal.
        void resize(int size, T value);

        // Copy contents into host_vector
        void get(thrust::host_vector<T> & vec) const;
        // Get a single element at index. Should be used sparingly,
        // performs copy.
        T get_at(int index) const;
        // Set a single element at index. Should be used sparingly,
        // performs copy
        void set(int index, const T & value);

        void populate(const thrust::host_vector<T> & hvec);
        void append(const thrust::host_vector<T> & hvec);

        void print(std::string prefix) const;

    private:
        thrust::device_vector<T> _dev_vec;
        int _alloc_size;
        int _current_size;
};

/**********************************************************************************
 *
 * DevVariable
 *
 * Wrapper around a device pointer to simplify usage.
 *
 **********************************************************************************/

template<typename T>
class DevVariable {
    public:
        DevVariable();
        DevVariable(T value);
        // Copy constructor and assignment only copy the pointers, not
        // the content
        DevVariable(const DevVariable & var);
        DevVariable<T> & operator=(const DevVariable & var);

        // Explicit copy, creates a new device vector
        DevVariable<T> copy();

        T get() const;
        void set(T value);
        // Raw pointer to device memory
        T* pointer();

    private:
        static void shared_ptr_deleter(T* ptr);

    private:
        boost::shared_ptr<T> _var;
};

/**********************************************************************************
 *
 * Timer
 *
 * Simple cumulative timer using cudaEvent
 *
 **********************************************************************************/

class Timer {
    public:
        Timer(std::string name) : _name(name), time(0) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
        }

        void start() {
            cudaEventRecord(_start, 0);
        }

        void stop() {
            float temp;
            cudaEventRecord(_stop, 0);
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(&temp, _start, _stop);
            time += temp;
        }

        void print() {
            std::cout << "Elapsed: " << _name << "\t" << time << " ms" << std::endl;
        }

        int get_ms() { return time; };

    private:
        cudaEvent_t _start, _stop;
        std::string _name;
        float time;
};

/**********************************************************************************
 *
 * custom functors for thrust::functional
 *
 **********************************************************************************/

struct IntegerDivide {
    IntegerDivide(int div) : divisor(div) {};
    __inline__ __device__ __host__
    int
    operator()(const int x) {
        int rest = ((x % divisor) == 0) ? 0 : 1;
        return x/divisor + rest;
    }
    const int divisor;
};

struct Float3Minimum {
    __device__
    float3 operator()(const float3 & f1, const float3 & f2) {
        return make_float3(fminf(f1.x, f2.x), fminf(f1.y, f2.y), fminf(f1.z, f2.z));
    }
};

struct Float3Maximum {
    __device__
    float3 operator()(const float3 & f1, const float3 & f2) {
        return make_float3(fmaxf(f1.x, f2.x), fmaxf(f1.y, f2.y), fmaxf(f1.z, f2.z));
    }
};


struct Float4Minimum {
    __device__
    UFloat4 operator()(const UFloat4 & f1, const UFloat4 & f2) {
        return make_ufloat4(fminf(f1.vec.x, f2.vec.x), fminf(f1.vec.y, f2.vec.y),
                            fminf(f1.vec.z, f2.vec.z), fminf(f1.vec.w, f2.vec.w));
    }
};

struct Float4Maximum {
    __device__
    UFloat4 operator()(const UFloat4 & f1, const UFloat4 & f2) {
        return make_ufloat4(fmaxf(f1.vec.x, f2.vec.x), fmaxf(f1.vec.y, f2.vec.y),
                            fmaxf(f1.vec.z, f2.vec.z), fmaxf(f1.vec.w, f2.vec.w));
    }
};

struct CountBitsFunctor : public thrust::unary_function<int,int> {
    template<typename T>
    __device__ __host__
    int operator()(T value) {
      unsigned int c;
      for (c = 0; value; c++)
        value &= value - 1;
      return c;
    }
};

struct FillLowestBitsFunctor : public thrust::unary_function<int,int> {
    __device__
    UInt64 operator()(int value) {
        UInt64 res = (UInt64) 1 << value;
        return res - 1;
    }
};

template<typename Tuple, typename T>
struct FirstIsValue {
    FirstIsValue(T val) : _val(val) {};

    __host__ __device__ __inline__
    bool operator()(Tuple x) {
        return (thrust::get<0>(x) == _val);
    }
    T _val;
};

template<typename T>
struct IsNonZero {
    __host__ __device__ __inline__
    bool operator()(T x) {
        return (x != 0);
    }
};

struct GreaterThanZero {
    __device__
    bool operator()(int val) {
        return val > 0;
    }
};

template<typename Pair>
struct AddPair {
    __host__ __device__ __inline__
    Pair operator()(const Pair & x, const Pair & y) {
        return thrust::make_tuple(thrust::get<0>(x) + thrust::get<0>(y),
                                  thrust::get<1>(x) + thrust::get<1>(y));
    }
};

/**********************************************************************************
 *
 * helper functions
 *
 **********************************************************************************/

float4 from_float3(float3 val);

template<typename T>
__device__
T min_three(const T & v1, const T & v2, const T & v3);

template<typename T>
__device__
T max_three(const T & v1, const T & v2, const T & v3);

/**********************************************************************************
 *
 * overloaded operator<<()
 *
 **********************************************************************************/

inline std::ostream & operator<<(std::ostream & str, float4 value);
inline std::ostream & operator<<(std::ostream & str, float3 value);

#include "utils-inl.h"

#endif  // CUTL_UTILS_H
