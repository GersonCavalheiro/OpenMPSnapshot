

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <alpaka/core/ApiHipRt.hpp>
#    include <alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp>

namespace alpaka
{
using QueueHipRtNonBlocking = QueueUniformCudaHipRtNonBlocking<ApiHipRt>;
} 

#endif 
