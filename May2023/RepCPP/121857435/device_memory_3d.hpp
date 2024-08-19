#pragma once

#include "defs.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace bnmf_algs {
namespace cuda {

template <typename T> class DeviceMemory3D {
public:

using value_type = T;


explicit DeviceMemory3D(size_t first_dim, size_t second_dim,
size_t third_dim)
: m_dims(shape<3>{first_dim, second_dim, third_dim}),
m_extent(
make_cudaExtent(third_dim * sizeof(T), second_dim, first_dim)),
m_ptr() {
auto err = cudaMalloc3D(&m_ptr, m_extent);
BNMF_ASSERT(
err == cudaSuccess,
"Error allocating memory in cuda::DeviceMemory3D::DeviceMemory3D");
}


DeviceMemory3D(const DeviceMemory3D&) = delete;


DeviceMemory3D& operator=(const DeviceMemory3D&) = delete;


DeviceMemory3D(DeviceMemory3D&& other)
: m_dims(other.m_dims), m_extent(other.m_extent), m_ptr(other.m_ptr) {
other.reset_members();
}


DeviceMemory3D& operator=(DeviceMemory3D&& other) {
this->free_cuda_mem();

this->m_dims = other.m_dims;
this->m_extent = other.m_extent;
this->m_ptr = other.m_ptr;

other.reset_members();

return *this;
}


~DeviceMemory3D() { free_cuda_mem(); }


cudaPitchedPtr pitched_ptr() const { return m_ptr; }


cudaExtent extent() const { return m_extent; }


shape<3> dims() const { return m_dims; }

private:

void free_cuda_mem() {
auto err = cudaFree(m_ptr.ptr);
BNMF_ASSERT(
err == cudaSuccess,
"Error deallocating memory in cuda::DeviceMemory3D::free_cuda_mem");
}


void reset_members() {
this->m_dims = {0, 0, 0};
this->m_extent = {0, 0, 0};
this->m_ptr = {nullptr, 0, 0, 0};
}

private:

shape<3> m_dims;


cudaExtent m_extent;


cudaPitchedPtr m_ptr;
};
} 
} 
