#pragma once

#include "cuda/memory.hpp"
#include "defs.hpp"
#include <array>

namespace bnmf_algs {

namespace cuda {


template <typename T>
void tensor_sums(const DeviceMemory3D<T>& tensor,
std::array<DeviceMemory2D<T>, 3>& result_arr);

} 
} 
