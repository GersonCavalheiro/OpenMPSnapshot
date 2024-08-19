

#pragma once

#include <alpaka/rand/Philox/PhiloxBaseCommon.hpp>
#include <alpaka/rand/Philox/PhiloxBaseStdArray.hpp>
#include <alpaka/rand/Philox/PhiloxStateless.hpp>
#include <alpaka/rand/Philox/PhiloxStatelessKeyedBase.hpp>
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#    include <alpaka/rand/Philox/PhiloxBaseCudaArray.hpp>
#endif

namespace alpaka::rand::engine::trait
{
template<typename TAcc>
constexpr inline bool isGPU = false;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
template<typename TApi, typename TDim, typename TIdx>
constexpr inline bool isGPU<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> = true;
#endif


template<typename TAcc, typename TParams, typename TSfinae = void>
struct PhiloxStatelessBaseTraits
{
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
using Backend = std::conditional_t<isGPU<TAcc>, PhiloxBaseCudaArray<TParams>, PhiloxBaseStdArray<TParams>>;
#else
using Backend = PhiloxBaseStdArray<TParams>;
#endif
using Counter = typename Backend::Counter; 
using Key = typename Backend::Key; 
template<typename TDistributionResultScalar>
using ResultContainer =
typename Backend::template ResultContainer<TDistributionResultScalar>; 
using Base = PhiloxStateless<Backend, TParams>;
};


template<typename TAcc, typename TParams, typename TSfinae = void>
struct PhiloxStatelessKeyedBaseTraits : public PhiloxStatelessBaseTraits<TAcc, TParams>
{
using Backend = typename PhiloxStatelessBaseTraits<TAcc, TParams>::Backend;
using Base = PhiloxStatelessKeyedBase<Backend, TParams>;
};


template<typename TAcc, typename TParams, typename TImpl, typename TSfinae = void>
struct PhiloxBaseTraits : public PhiloxStatelessBaseTraits<TAcc, TParams>
{
using Backend = typename PhiloxStatelessBaseTraits<TAcc, TParams>::Backend;
using Base = PhiloxBaseCommon<Backend, TParams, TImpl>;
};
} 
