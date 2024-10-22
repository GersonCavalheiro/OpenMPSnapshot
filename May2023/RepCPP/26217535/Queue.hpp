

#pragma once

#include <alpaka/alpaka.hpp>

namespace alpaka::test
{
namespace trait
{
template<typename TDev, typename TSfinae = void>
struct DefaultQueueType;

template<>
struct DefaultQueueType<DevCpu>
{
#if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
using type = QueueCpuBlocking;
#else
using type = QueueCpuNonBlocking;
#endif
};

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

template<typename TApi>
struct DefaultQueueType<DevUniformCudaHipRt<TApi>>
{
#    if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
using type = QueueUniformCudaHipRtBlocking<TApi>;
#    else
using type = QueueUniformCudaHipRtNonBlocking<TApi>;
#    endif
};
#endif
} 

template<typename TDev>
using DefaultQueue = typename trait::DefaultQueueType<TDev>::type;

namespace trait
{
template<typename TQueue, typename TSfinae = void>
struct IsBlockingQueue;

template<typename TDev>
struct IsBlockingQueue<QueueGenericThreadsBlocking<TDev>>
{
static constexpr bool value = true;
};

template<typename TDev>
struct IsBlockingQueue<QueueGenericThreadsNonBlocking<TDev>>
{
static constexpr bool value = false;
};

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

template<typename TApi>
struct IsBlockingQueue<QueueUniformCudaHipRtBlocking<TApi>>
{
static constexpr bool value = true;
};

template<typename TApi>
struct IsBlockingQueue<QueueUniformCudaHipRtNonBlocking<TApi>>
{
static constexpr bool value = false;
};
#endif

#ifdef ALPAKA_ACC_SYCL_ENABLED
#    ifdef ALPAKA_SYCL_BACKEND_ONEAPI
#        ifdef ALPAKA_SYCL_ONEAPI_CPU
template<>
struct DefaultQueueType<alpaka::DevCpuSyclIntel>
{
#            if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
using type = alpaka::QueueCpuSyclIntelBlocking;
#            else
using type = alpaka::QueueCpuSyclIntelNonBlocking;
#            endif
};

template<>
struct IsBlockingQueue<alpaka::QueueCpuSyclIntelBlocking>
{
static constexpr auto value = true;
};

template<>
struct IsBlockingQueue<alpaka::QueueCpuSyclIntelNonBlocking>
{
static constexpr auto value = false;
};
#        endif
#        ifdef ALPAKA_SYCL_ONEAPI_FPGA
template<>
struct DefaultQueueType<alpaka::DevFpgaSyclIntel>
{
#            if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
using type = alpaka::QueueFpgaSyclIntelBlocking;
#            else
using type = alpaka::QueueFpgaSyclIntelNonBlocking;
#            endif
};

template<>
struct IsBlockingQueue<alpaka::QueueFpgaSyclIntelBlocking>
{
static constexpr auto value = true;
};

template<>
struct IsBlockingQueue<alpaka::QueueFpgaSyclIntelNonBlocking>
{
static constexpr auto value = false;
};
#        endif
#        ifdef ALPAKA_SYCL_ONEAPI_GPU
template<>
struct DefaultQueueType<alpaka::DevGpuSyclIntel>
{
#            if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
using type = alpaka::QueueGpuSyclIntelBlocking;
#            else
using type = alpaka::QueueGpuSyclIntelNonBlocking;
#            endif
};

template<>
struct IsBlockingQueue<alpaka::QueueGpuSyclIntelBlocking>
{
static constexpr auto value = true;
};

template<>
struct IsBlockingQueue<alpaka::QueueGpuSyclIntelNonBlocking>
{
static constexpr auto value = false;
};
#        endif
#    endif
#    ifdef ALPAKA_SYCL_BACKEND_XILINX
template<>
struct DefaultQueueType<alpaka::DevFpgaSyclXilinx>
{
#        if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
using type = alpaka::QueueFpgaSyclXilinxBlocking;
#        else
using type = alpaka::QueueFpgaSyclXilinxNonBlocking;
#        endif
};

template<>
struct IsBlockingQueue<alpaka::QueueFpgaSyclXilinxBlocking>
{
static constexpr auto value = true;
};

template<>
struct IsBlockingQueue<alpaka::QueueFpgaSyclXilinxNonBlocking>
{
static constexpr auto value = false;
};
#    endif
#endif
} 
template<typename TQueue>
using IsBlockingQueue = trait::IsBlockingQueue<TQueue>;

using TestQueues = std::tuple<
std::tuple<DevCpu, QueueCpuBlocking>,
std::tuple<DevCpu, QueueCpuNonBlocking>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
,
std::tuple<DevCudaRt, QueueCudaRtBlocking>,
std::tuple<DevCudaRt, QueueCudaRtNonBlocking>
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
,
std::tuple<DevHipRt, QueueHipRtBlocking>,
std::tuple<DevHipRt, QueueHipRtNonBlocking>
#endif
#ifdef ALPAKA_ACC_SYCL_ENABLED
#    ifdef ALPAKA_SYCL_BACKEND_ONEAPI
#        ifdef ALPAKA_SYCL_ONEAPI_CPU
,
std::tuple<alpaka::DevCpuSyclIntel, alpaka::QueueCpuSyclIntelBlocking>,
std::tuple<alpaka::DevCpuSyclIntel, alpaka::QueueCpuSyclIntelNonBlocking>
#        endif
#        ifdef ALPAKA_SYCL_ONEAPI_FPGA
,
std::tuple<alpaka::DevFpgaSyclIntel, alpaka::QueueFpgaSyclIntelBlocking>,
std::tuple<alpaka::DevFpgaSyclIntel, alpaka::QueueFpgaSyclIntelNonBlocking>
#        endif
#        ifdef ALPAKA_SYCL_ONEAPI_GPU
,
std::tuple<alpaka::DevGpuSyclIntel, alpaka::QueueGpuSyclIntelBlocking>,
std::tuple<alpaka::DevGpuSyclIntel, alpaka::QueueGpuSyclIntelNonBlocking>
#        endif
#    endif
#    if defined(ALPAKA_SYCL_BACKEND_XILINX)
,
std::tuple<alpaka::DevFpgaSyclXilinx, alpaka::QueueFpgaSyclXilinxBlocking>,
std::tuple<alpaka::DevFpgaSyclXilinx, alpaka::QueueFpgaSyclXilinxNonBlocking>
#    endif
#endif
>;
} 
