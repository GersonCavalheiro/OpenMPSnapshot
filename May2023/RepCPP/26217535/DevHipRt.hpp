

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <alpaka/core/ApiHipRt.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>

namespace alpaka
{
using DevHipRt = DevUniformCudaHipRt<ApiHipRt>;
} 

#endif 
