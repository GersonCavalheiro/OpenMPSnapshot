#pragma once

#include "defs.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace bnmf_algs {
namespace cuda {

template <typename T> class DeviceMemory2D {
public:

using value_type = T;


explicit DeviceMemory2D(size_t rows, size_t cols)
: m_data(nullptr), m_pitch(), m_dims(shape<2>{rows, cols}) {
auto err = cudaMallocPitch((void**)(&m_data), &m_pitch,
cols * sizeof(T), rows);
BNMF_ASSERT(
err == cudaSuccess,
"Error allocating memory in cuda::DeviceMemory2D::DeviceMemory2D");
};


DeviceMemory2D(const DeviceMemory2D&) = delete;


DeviceMemory2D& operator=(const DeviceMemory2D&) = delete;


DeviceMemory2D(DeviceMemory2D&& other)
: m_data(other.m_data), m_pitch(other.m_pitch), m_dims(other.m_dims) {
other.reset_members();
}


DeviceMemory2D& operator=(DeviceMemory2D&& other) {
this->free_cuda_mem();

this->m_data = other.m_data;
this->m_pitch = other.m_pitch;
this->m_dims = other.m_dims;

other.reset_members();

return *this;
}


~DeviceMemory2D() { free_cuda_mem(); }


T* data() const { return m_data; }


size_t pitch() const { return m_pitch; }


size_t width() const { return m_dims[1] * sizeof(T); }


size_t height() const { return m_dims[0]; }


shape<2> dims() const { return m_dims; }

private:

void free_cuda_mem() {
auto err = cudaFree(m_data);
BNMF_ASSERT(
err == cudaSuccess,
"Error deallocating memory in cuda::DeviceMemory2D::free_cuda_mem");
}


void reset_members() {
this->m_data = nullptr;
this->m_pitch = 0;
this->m_dims = {0, 0};
}

private:

T* m_data;


size_t m_pitch;


shape<2> m_dims;
};
} 
} 
