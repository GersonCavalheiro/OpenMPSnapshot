

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include <alpaka/core/ApiCudaRt.hpp>
#    include <alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp>

namespace alpaka
{
using QueueCudaRtNonBlocking = QueueUniformCudaHipRtNonBlocking<ApiCudaRt>;
} 

#endif 
