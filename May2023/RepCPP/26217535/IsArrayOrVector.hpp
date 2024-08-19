

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    include <alpaka/meta/CudaVectorArrayWrapper.hpp>
#endif

#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

namespace alpaka::meta
{

template<typename T>
struct IsArrayOrVector : std::false_type
{
};


template<typename T, typename A>
struct IsArrayOrVector<std::vector<T, A>> : std::true_type
{
};


template<typename T, std::size_t N>
struct IsArrayOrVector<T[N]> : std::true_type
{
};


template<typename T, std::size_t N>
struct IsArrayOrVector<std::array<T, N>> : std::true_type
{
};

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
template<typename T, unsigned N>
struct IsArrayOrVector<CudaVectorArrayWrapper<T, N>> : std::true_type
{
};
#endif
} 
