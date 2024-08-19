

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>

#include <tuple>

namespace alpaka::test
{
using TestDims = std::tuple<
DimInt<0u>,
DimInt<1u>,
DimInt<2u>,
DimInt<3u>
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(ALPAKA_ACC_SYCL_ENABLED)
,
DimInt<4u>
#endif
>;
} 
