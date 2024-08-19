

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include <alpaka/core/ApiCudaRt.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>

namespace alpaka
{
using DevCudaRt = DevUniformCudaHipRt<ApiCudaRt>;
} 

#endif 
