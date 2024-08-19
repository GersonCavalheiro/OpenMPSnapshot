

#pragma once

#include "iterator.hpp"

#include <alpaka/alpaka.hpp>

using Dim = alpaka::DimInt<1u>;
using Idx = uint64_t;
using Extent = uint64_t;
using WorkDiv = alpaka::WorkDivMembers<Dim, Extent>;

template<typename TAcc, uint64_t TSize>
static constexpr auto getMaxBlockSize() -> uint64_t
{
return (TAcc::MaxBlockSize::value > TSize) ? TSize : TAcc::MaxBlockSize::value;
}

template<typename T, typename TBuf, typename TAcc>
struct GetIterator
{
using Iterator = IteratorCpu<TAcc, T, TBuf>;
};


#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
struct CpuOmp2Blocks
{
using Host = alpaka::AccCpuOmp2Blocks<Dim, Extent>;
using Acc = alpaka::AccCpuOmp2Blocks<Dim, Extent>;
using SmCount = alpaka::DimInt<1u>;
using MaxBlockSize = alpaka::DimInt<1u>;
};

template<typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccCpuOmp2Blocks<TArgs...>>
{
using Iterator = IteratorCpu<alpaka::AccCpuOmp2Blocks<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
struct CpuSerial
{
using Host = alpaka::AccCpuSerial<Dim, Extent>;
using Acc = alpaka::AccCpuSerial<Dim, Extent>;
using MaxBlockSize = alpaka::DimInt<1u>;
};

template<typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccCpuSerial<TArgs...>>
{
using Iterator = IteratorCpu<alpaka::AccCpuSerial<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
struct CpuThreads
{
using Host = alpaka::AccCpuThreads<Dim, Extent>;
using Acc = alpaka::AccCpuThreads<Dim, Extent>;
using MaxBlockSize = alpaka::DimInt<1u>;
};

template<typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccCpuThreads<TArgs...>>
{
using Iterator = IteratorCpu<alpaka::AccCpuThreads<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#    ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
struct GpuCudaRt
{
using Host = alpaka::AccCpuSerial<Dim, Extent>;
using Acc = alpaka::AccGpuCudaRt<Dim, Extent>;
using MaxBlockSize = alpaka::DimInt<1024u>;
};

template<typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccGpuUniformCudaHipRt<TArgs...>>
{
using Iterator = IteratorGpu<alpaka::AccGpuUniformCudaHipRt<TArgs...>, T, TBuf>;
};
#    endif
#endif
