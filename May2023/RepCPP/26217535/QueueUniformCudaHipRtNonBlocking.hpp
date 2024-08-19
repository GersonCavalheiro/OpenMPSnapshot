

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/queue/cuda_hip/QueueUniformCudaHipRt.hpp>

namespace alpaka
{
template<typename TApi>
using QueueUniformCudaHipRtNonBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, false>;

} 

#endif
