

#pragma once

#include <alpaka/meta/CudaVectorArrayWrapper.hpp>

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

namespace alpaka::rand::engine
{
namespace trait
{
template<typename TScalar>
struct PhiloxResultContainerTraits;

template<>
struct PhiloxResultContainerTraits<float>
{
using type = meta::CudaVectorArrayWrapper<float, 4>;
};

template<>
struct PhiloxResultContainerTraits<double>
{
using type = meta::CudaVectorArrayWrapper<double, 4>;
};

template<>
struct PhiloxResultContainerTraits<int>
{
using type = meta::CudaVectorArrayWrapper<int, 4>;
};

template<>
struct PhiloxResultContainerTraits<unsigned>
{
using type = meta::CudaVectorArrayWrapper<unsigned, 4>;
};

template<typename TScalar>
using PhiloxResultContainer = typename PhiloxResultContainerTraits<TScalar>::type;
} 


template<typename TParams>
class PhiloxBaseCudaArray
{
static_assert(TParams::counterSize == 4, "GPU Philox implemented only for counters of width == 4");

public:
using Counter
= meta::CudaVectorArrayWrapper<unsigned, 4>; 
using Key = meta::CudaVectorArrayWrapper<unsigned, 2>; 
template<typename TDistributionResultScalar>
using ResultContainer = trait::PhiloxResultContainer<TDistributionResultScalar>; 
};
} 

#endif
