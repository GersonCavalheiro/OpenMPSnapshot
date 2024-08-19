

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <alpaka/core/ApiHipRt.hpp>
#    include <alpaka/kernel/TaskKernelGpuUniformCudaHipRt.hpp>

namespace alpaka
{
template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
using TaskKernelGpuHipRt = TaskKernelGpuUniformCudaHipRt<ApiHipRt, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>;
}

#endif 
