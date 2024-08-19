#pragma once

#include "defs.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace bnmf_algs {
namespace cuda {

template <typename T> class HostMemory3D {
public:

using value_type = T;


explicit HostMemory3D(T* data, size_t first_dim, size_t second_dim,
size_t third_dim)
: m_dims(shape<3>{first_dim, second_dim, third_dim}),
m_extent(
make_cudaExtent(third_dim * sizeof(T), second_dim, first_dim)),
m_ptr() {
m_ptr.pitch = third_dim * sizeof(T);
m_ptr.xsize = third_dim;
m_ptr.ysize = second_dim;
m_ptr.ptr = (void*)(data);
}


cudaPitchedPtr pitched_ptr() const { return m_ptr; }


cudaExtent extent() const { return m_extent; }


shape<3> dims() const { return m_dims; }

private:

shape<3> m_dims;


cudaExtent m_extent;


cudaPitchedPtr m_ptr;
};
} 
} 
