#pragma once

#include "defs.hpp"
#include <cstddef>

namespace bnmf_algs {
namespace cuda {

template <typename T> class HostMemory1D {
public:

using value_type = T;


HostMemory1D(T* data, size_t num_elems)
: m_dims(shape<1>{num_elems}), m_data(data){};


T* data() const { return m_data; }


size_t bytes() const { return m_dims[0] * sizeof(T); }


shape<1> dims() const { return m_dims; }

private:

shape<1> m_dims;

T* m_data;
};
} 
} 
