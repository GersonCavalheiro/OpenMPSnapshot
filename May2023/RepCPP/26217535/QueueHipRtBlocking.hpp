

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <alpaka/core/ApiHipRt.hpp>
#    include <alpaka/queue/QueueUniformCudaHipRtBlocking.hpp>

namespace alpaka
{
using QueueHipRtBlocking = QueueUniformCudaHipRtBlocking<ApiHipRt>;
} 

#endif 
