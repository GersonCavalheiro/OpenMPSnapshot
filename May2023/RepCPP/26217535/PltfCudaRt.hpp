

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include <alpaka/core/ApiCudaRt.hpp>
#    include <alpaka/pltf/PltfUniformCudaHipRt.hpp>

namespace alpaka
{
using PltfCudaRt = PltfUniformCudaHipRt<ApiCudaRt>;
} 

#endif 
