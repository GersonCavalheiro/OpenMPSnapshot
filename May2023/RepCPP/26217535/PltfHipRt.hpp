

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <alpaka/core/ApiHipRt.hpp>
#    include <alpaka/pltf/PltfUniformCudaHipRt.hpp>

namespace alpaka
{
using PltfHipRt = PltfUniformCudaHipRt<ApiHipRt>;
} 

#endif 
