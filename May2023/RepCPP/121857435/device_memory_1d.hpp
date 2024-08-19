#pragma once

#include "defs.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace bnmf_algs {
namespace cuda {


template <typename T> class DeviceMemory1D {
public:

using value_type = T;


explicit DeviceMemory1D(size_t num_elems)
: m_dims(shape<1>{num_elems}), m_data(nullptr) {
size_t alloc_size = num_elems * sizeof(T);
auto err = cudaMalloc((void**)(&m_data), alloc_size);
BNMF_ASSERT(
err == cudaSuccess,
"Error allocating memory in cuda::DeviceMemory1D::DeviceMemory1D");
};


DeviceMemory1D(const DeviceMemory1D&) = delete;


DeviceMemory1D& operator=(const DeviceMemory1D&) = delete;


DeviceMemory1D(DeviceMemory1D&& other)
: m_dims(other.m_dims), m_data(other.m_data) {
other.reset_members();
}


DeviceMemory1D& operator=(DeviceMemory1D&& other) {
this->free_cuda_mem();
this->m_dims = other.m_dims;
this->m_data = other.m_data;

other.reset_members();

return *this;
}


~DeviceMemory1D() { free_cuda_mem(); }


T* data() const { return m_data; }


size_t bytes() const { return m_dims[0] * sizeof(T); }


shape<1> dims() const { return m_dims; }

private:

void free_cuda_mem() {
auto err = cudaFree(m_data);
BNMF_ASSERT(
err == cudaSuccess,
"Error deallocating memory in cuda::DeviceMemory1D::free_cuda_mem");
}


void reset_members() {
m_dims[0] = 0;
m_data = nullptr;
}

private:

shape<1> m_dims;


T* m_data;
};
} 
} 
