#pragma once

#include "defs.hpp"
#include <cstddef>

namespace bnmf_algs {
namespace cuda {

template <typename T> class HostMemory2D {
public:

using value_type = T;


explicit HostMemory2D(T* data, size_t rows, size_t cols)
: m_data(data), m_pitch(cols * sizeof(T)),
m_dims(shape<2>{rows, cols}){};


T* data() const { return m_data; }


size_t pitch() const { return m_pitch; }


size_t width() const { return m_dims[1] * sizeof(T); }


size_t height() const { return m_dims[0]; }


shape<2> dims() const { return m_dims; }

private:

T* m_data;


size_t m_pitch;


shape<2> m_dims;
};
} 
} 
