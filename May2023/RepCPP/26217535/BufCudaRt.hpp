

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include <alpaka/core/ApiCudaRt.hpp>
#    include <alpaka/mem/buf/BufUniformCudaHipRt.hpp>

namespace alpaka
{
template<typename TElem, typename TDim, typename TIdx>
using BufCudaRt = BufUniformCudaHipRt<ApiCudaRt, TElem, TDim, TIdx>;
}

#endif 