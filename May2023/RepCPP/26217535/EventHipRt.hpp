

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <alpaka/core/ApiHipRt.hpp>
#    include <alpaka/event/EventUniformCudaHipRt.hpp>

namespace alpaka
{
using EventHipRt = EventUniformCudaHipRt<ApiHipRt>;
} 

#endif 
