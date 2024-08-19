

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include <alpaka/core/ApiCudaRt.hpp>
#    include <alpaka/event/EventUniformCudaHipRt.hpp>

namespace alpaka
{
using EventCudaRt = EventUniformCudaHipRt<ApiCudaRt>;
} 

#endif 
