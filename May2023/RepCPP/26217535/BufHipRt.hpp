

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <alpaka/core/ApiHipRt.hpp>
#    include <alpaka/mem/buf/BufUniformCudaHipRt.hpp>

namespace alpaka
{
template<typename TElem, typename TDim, typename TIdx>
using BufHipRt = BufUniformCudaHipRt<ApiHipRt, TElem, TDim, TIdx>;
}

#endif 
