#pragma once

#include "defs.hpp"
#include <cuda.h>
#include <device_launch_parameters.h>

namespace bnmf_algs {
namespace cuda {

namespace kernel {

template <typename Scalar>
__global__ void sum_tensor3D(cudaPitchedPtr tensor, Scalar* out,
size_t out_pitch, size_t axis, size_t n_rows,
size_t n_cols, size_t n_layers);
} 
} 
} 
