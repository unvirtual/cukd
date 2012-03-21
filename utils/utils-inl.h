// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cutl/blob/master/LICENSE

#include <cutil_inline.h>

/**********************************************************************************
 *
 * DevVector implementation
 *
 **********************************************************************************/

template<typename T>
DevVector<T>::DevVector(int alloc_size)
    : _alloc_size(alloc_size), _current_size(0) {
        _dev_vec.resize(_alloc_size);
}

template<typename T>
inline void
DevVector<T>::clear() {
    _current_size = 0;
}

template<typename T>
inline void
DevVector<T>::free() {
    _current_size = 0;
    _dev_vec.clear();
}

template<typename T>
DevVector<T>
DevVector<T>::copy() {
    DevVector<T> vec(_alloc_size);
    vec.resize(_current_size);
    thrust::copy(begin(), end(), vec.begin());
    return vec;
}

template<typename T>
void
DevVector<T>::populate(const thrust::host_vector<T> & hvec) {
    if(hvec.size() > _current_size)
        resize(hvec.size());
    thrust::copy(hvec.begin(), hvec.end(), _dev_vec.begin());
}

template<typename T>
void
DevVector<T>::append(const thrust::host_vector<T> & hvec) {
    int last_size = _current_size;
    resize(hvec.size() + _current_size);
    thrust::copy(hvec.begin(), hvec.end(), _dev_vec.begin() + last_size);
}

template<typename T>
void
DevVector<T>::resize(int size) {
    if(size <= _alloc_size) {
        _current_size = size;
    } else {
        bool resize = false;
        while(size > _alloc_size) {
            _alloc_size *= 2;
            resize = true;
        }
        if(resize) {
            _dev_vec.resize(_alloc_size);
        }
        _current_size = size;
    }
}

template<typename T>
void
DevVector<T>::resize(int size, T value) {
    if(size < _current_size) {
        std::cerr << "Error: DevVector::resize: tried to shrink with value"
                  << std::endl;
        exit(1);
    }
    bool resize = false;
    while(size > _alloc_size) {
        _alloc_size *= 2;
        resize = true;
    }
    if(resize)
        _dev_vec.resize(_alloc_size);
    thrust::fill(_dev_vec.begin() + _current_size, _dev_vec.begin() + size, value);
    _current_size = size;
}

template<typename T>
void
DevVector<T>::get(thrust::host_vector<T> & vec) const {
    vec.resize(_current_size);
    thrust::copy(_dev_vec.begin(), _dev_vec.begin() + _current_size,
            vec.begin());
}

template<typename T>
void
DevVector<T>::print(std::string prefix="") const {
    thrust::host_vector<T> hvec = _dev_vec;
    if(_current_size == 0) {
        std::cout << "empty vector: " << prefix << std::endl;
    }
    for(int i = 0; i < _current_size; ++i)
        std::cout << prefix << "[" << i << "] = " << hvec[i] << std::endl;
}

template<typename T>
inline
DevVector<T>::iterator
DevVector<T>::begin() {
    return _dev_vec.begin();
}

template<typename T>
inline
DevVector<T>::iterator
DevVector<T>::end() {
    return _dev_vec.begin() + _current_size;
}

template<typename T>
inline
T*
DevVector<T>::pointer() {
    return thrust::raw_pointer_cast(&_dev_vec[0]);
}

template<typename T>
inline
int
DevVector<T>::alloc_size() const {
    return _alloc_size;
}

template<typename T>
inline
int
DevVector<T>::size() const {
    return _current_size;
}

template<typename T>
T
DevVector<T>::get_at(int index) const {
    if(index > _current_size) {
        std::cerr << "Error: illegal access to DevVector" << std::endl;
        exit(1);
    }
    return _dev_vec[index];
}

template<typename T>
void
DevVector<T>::set(int index, const T & value) {
    _dev_vec[index] = value;
}

/**********************************************************************************
 *
 * DevVariable implementation
 *
 **********************************************************************************/

template<typename T>
void
DevVariable<T>::shared_ptr_deleter(T* ptr) {
    cudaFree(ptr);
}

template<typename T>
DevVariable<T>::DevVariable() {
    T value = (T) 0;
    T * ptr;
    cutilSafeCall(cudaMalloc((void**) &ptr, sizeof(T)));
    cutilSafeCall(cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyHostToDevice));
    _var = boost::shared_ptr<T>(ptr, DevVariable<T>::shared_ptr_deleter);
}

template<typename T>
DevVariable<T>::DevVariable(T value) {
    T * ptr;
    cutilSafeCall(cudaMalloc((void**) &ptr, sizeof(T)));
    cutilSafeCall(cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyHostToDevice));
    _var = boost::shared_ptr<T>((T*) ptr, DevVariable<T>::shared_ptr_deleter);
}

template<typename T>
DevVariable<T>::DevVariable(const DevVariable & var) {
    _var = var._var;
}

template<typename T>
DevVariable<T> &
DevVariable<T>::operator=(const DevVariable & var) {
    if(&var != this) {
        _var = var._var;
    }
    return *this;
}

template<typename T>
DevVariable<T>
DevVariable<T>::copy() {
    return DevVariable<T>(get());
}

template<typename T>
T
DevVariable<T>::get() const {
    T value;
    cudaMemcpy(&value, _var.get(), sizeof(T), cudaMemcpyDeviceToHost);
    return value;
}

template<typename T>
void
DevVariable<T>::set(T value) {
    cudaMemcpy(_var.get(), &value, sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
T*
DevVariable<T>::pointer() {
    return _var.get();
}

/**********************************************************************************
 *
 * helper functions
 *
 **********************************************************************************/

template<typename T>
__device__
T min_three(const T & v1, const T & v2, const T & v3) {
    T val = min(v1,v2);
    return min(val,v3);
}

template<typename T>
__device__
T max_three(const T & v1, const T & v2, const T & v3) {
    T val = max(v1,v2);
    return max(val,v3);
}

inline std::ostream & operator<<(std::ostream & str, float4 value) {
    str << value.x << " " << value.y << " " << value.z << " " << value.w;
    return str;
}
inline std::ostream & operator<<(std::ostream & str, float3 value) {
    str << value.x << " " << value.y << " " << value.z;
    return str;
}
inline std::ostream & operator<<(std::ostream & str, UFloat4 value) {
    str << value.vec;
    return str;
}

inline
float4 from_float3(float3 val) {
    return make_float4(val.x, val.y, val.z, 0);
}
