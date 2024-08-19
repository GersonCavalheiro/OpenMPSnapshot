

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include <alpaka/core/ApiCudaRt.hpp>
#    include <alpaka/queue/QueueUniformCudaHipRtBlocking.hpp>

namespace alpaka
{
using QueueCudaRtBlocking = QueueUniformCudaHipRtBlocking<ApiCudaRt>;
} 

#endif 
