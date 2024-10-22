


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_GPU)

#    include <alpaka/kernel/TaskKernelGenericSycl.hpp>

namespace alpaka
{
template<typename TDim, typename TIdx>
class AccGpuSyclIntel;

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
using TaskKernelGpuSyclIntel
= TaskKernelGenericSycl<AccGpuSyclIntel<TDim, TIdx>, TDim, TIdx, TKernelFnObj, TArgs...>;
} 

#endif
