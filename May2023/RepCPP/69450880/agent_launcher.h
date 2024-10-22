
#pragma once

#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC
#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/core/util.h>
#include <cassert>

#if 0
#define __HYDRA_THRUST__TEMPLATE_DEBUG
#endif

#if __HYDRA_THRUST__TEMPLATE_DEBUG
template<int...> class ID_impl;
template<int... I> class Foo { ID_impl<I...> t;};
#endif

HYDRA_THRUST_BEGIN_NS
namespace cuda_cub {
namespace core {


#ifdef __CUDA_ARCH__
#if 0
template <class Agent, class... Args>
void __global__
__launch_bounds__(Agent::ptx_plan::BLOCK_THREADS,Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(Args... args)
{
extern __shared__ char shmem[];
Agent::entry(args..., shmem);
}
#else
template <class Agent, class _0>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0)
{
extern __shared__ char shmem[];
Agent::entry(x0, shmem);
}
template <class Agent, class _0, class _1>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, shmem);
}
template <class Agent, class _0, class _1, class _2>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, shmem);
}
template <class Agent, class _0, class _1, class _2, class _3>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, x3, shmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, x3, x4, shmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, x3, x4, x5, shmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, x3, x4, x5, x6, shmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, shmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, shmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, shmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, shmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, shmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, shmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, shmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD, _xE xE)
{
extern __shared__ char shmem[];
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE, shmem);
}
#endif



#if 0
template <class Agent, class... Args>
void __global__
__launch_bounds__(Agent::ptx_plan::BLOCK_THREADS,Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, Args... args)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(args..., vshmem);
}
#else
template <class Agent, class _0>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, vshmem);
}
template <class Agent, class _0, class _1>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, vshmem);
}
template <class Agent, class _0, class _1, class _2>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, vshmem);
}
template <class Agent, class _0, class _1, class _2, class _3>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, x3, vshmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, x3, x4, vshmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, x3, x4, x5, vshmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, x3, x4, x5, x6, vshmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, vshmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, vshmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, vshmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, vshmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, vshmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, vshmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, vshmem);
}
template <class Agent, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
void __global__ __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS, Agent::ptx_plan::MIN_BLOCKS)
_kernel_agent_vshmem(char* vshmem, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD, _xE xE)
{
extern __shared__ char shmem[];
vshmem = vshmem == NULL ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
Agent::entry(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE, vshmem);
}
#endif
#else
#if 0
template <class , class... Args >
void __global__  _kernel_agent(Args... args) {}
template <class , class... Args >
void __global__  _kernel_agent_vshmem(char*, Args... args) {}
#else
template <class, class _0>
void __global__ _kernel_agent(_0) {}
template <class, class _0, class _1>
void __global__ _kernel_agent(_0,_1) {}
template <class, class _0, class _1, class _2>
void __global__ _kernel_agent(_0,_1,_2) {}
template <class, class _0, class _1, class _2, class _3>
void __global__ _kernel_agent(_0,_1,_2,_3) {}
template <class, class _0, class _1, class _2, class _3, class _4>
void __global__ _kernel_agent(_0,_1,_2,_3, _4) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5>
void __global__ _kernel_agent(_0,_1,_2,_3, _4, _5) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
void __global__ _kernel_agent(_0,_1,_2,_3, _4, _5, _6) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
void __global__ _kernel_agent(_0,_1,_2,_3, _4, _5, _6, _7) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
void __global__ _kernel_agent(_0,_1,_2,_3, _4, _5, _6, _7, _8) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
void __global__ _kernel_agent(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
void __global__ _kernel_agent(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
void __global__ _kernel_agent(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
void __global__ _kernel_agent(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB,_xC) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
void __global__ _kernel_agent(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB,_xC, _xD) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
void __global__ _kernel_agent(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB,_xC, _xD, _xE) {}
template <class, class _0>
void __global__ _kernel_agent_vshmem(char*,_0) {}
template <class, class _0, class _1>
void __global__ _kernel_agent_vshmem(char*,_0,_1) {}
template <class, class _0, class _1, class _2>
void __global__ _kernel_agent_vshmem(char*,_0,_1,_2) {}
template <class, class _0, class _1, class _2, class _3>
void __global__ _kernel_agent_vshmem(char*,_0,_1,_2,_3) {}
template <class, class _0, class _1, class _2, class _3, class _4>
void __global__ _kernel_agent_vshmem(char*,_0,_1,_2,_3, _4) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5>
void __global__ _kernel_agent_vshmem(char*,_0,_1,_2,_3, _4, _5) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
void __global__ _kernel_agent_vshmem(char*,_0,_1,_2,_3, _4, _5, _6) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
void __global__ _kernel_agent_vshmem(char*,_0,_1,_2,_3, _4, _5, _6, _7) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
void __global__ _kernel_agent_vshmem(char*,_0,_1,_2,_3, _4, _5, _6, _7, _8) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
void __global__ _kernel_agent_vshmem(char*,_0, _1, _2, _3, _4, _5, _6, _7, _8, _9) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
void __global__ _kernel_agent_vshmem(char*,_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
void __global__ _kernel_agent_vshmem(char*,_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
void __global__ _kernel_agent_vshmem(char*,_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB, _xC) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
void __global__ _kernel_agent_vshmem(char*,_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB, _xC, _xD) {}
template <class, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
void __global__ _kernel_agent_vshmem(char*,_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB, _xC, _xD, _xE) {}
#endif
#endif


template<class Agent>
struct AgentLauncher : Agent
{
core::AgentPlan plan;
size_t          count;
cudaStream_t    stream;
char const*     name;
bool            debug_sync;
unsigned int    grid;
char*           vshmem;
bool            has_shmem;
size_t          shmem_size;

enum
{
MAX_SHMEM_PER_BLOCK = 48 * 1024,
};
typedef
typename has_enough_shmem<Agent,
MAX_SHMEM_PER_BLOCK>::type has_enough_shmem_t;
typedef
has_enough_shmem<Agent,
MAX_SHMEM_PER_BLOCK> shm1;

template <class Size>
HYDRA_THRUST_RUNTIME_FUNCTION
AgentLauncher(AgentPlan    plan_,
Size         count_,
cudaStream_t stream_,
char const*  name_,
bool         debug_sync_)
: plan(plan_),
count((size_t)count_),
stream(stream_),
name(name_),
debug_sync(debug_sync_),
grid(static_cast<unsigned int>((count + plan.items_per_tile - 1) / plan.items_per_tile)),
vshmem(NULL),
has_shmem((size_t)core::get_max_shared_memory_per_block() >= (size_t)plan.shared_memory_size),
shmem_size(has_shmem ? plan.shared_memory_size : 0)
{
assert(count > 0);
}

template <class Size>
HYDRA_THRUST_RUNTIME_FUNCTION
AgentLauncher(AgentPlan    plan_,
Size         count_,
cudaStream_t stream_,
char*        vshmem,
char const*  name_,
bool         debug_sync_)
: plan(plan_),
count((size_t)count_),
stream(stream_),
name(name_),
debug_sync(debug_sync_),
grid(static_cast<unsigned int>((count + plan.items_per_tile - 1) / plan.items_per_tile)),
vshmem(vshmem),
has_shmem((size_t)core::get_max_shared_memory_per_block() >= (size_t)plan.shared_memory_size),
shmem_size(has_shmem ? plan.shared_memory_size : 0)
{
assert(count > 0);
}

HYDRA_THRUST_RUNTIME_FUNCTION
AgentLauncher(AgentPlan    plan_,
cudaStream_t stream_,
char const*  name_,
bool         debug_sync_)
: plan(plan_),
count(0),
stream(stream_),
name(name_),
debug_sync(debug_sync_),
grid(plan.grid_size),
vshmem(NULL),
has_shmem((size_t)core::get_max_shared_memory_per_block() >= (size_t)plan.shared_memory_size),
shmem_size(has_shmem ? plan.shared_memory_size : 0)
{
assert(plan.grid_size > 0);
}

HYDRA_THRUST_RUNTIME_FUNCTION
AgentLauncher(AgentPlan    plan_,
cudaStream_t stream_,
char*        vshmem,
char const*  name_,
bool         debug_sync_)
: plan(plan_),
count(0),
stream(stream_),
name(name_),
debug_sync(debug_sync_),
grid(plan.grid_size),
vshmem(vshmem),
has_shmem((size_t)core::get_max_shared_memory_per_block() >= (size_t)plan.shared_memory_size),
shmem_size(has_shmem ? plan.shared_memory_size : 0)
{
assert(plan.grid_size > 0);
}

#if 0
HYDRA_THRUST_RUNTIME_FUNCTION
AgentPlan static get_plan(cudaStream_t s, void* d_ptr = 0)
{
#ifdef __CUDACC_RDC__
return core::get_agent_plan<Agent>(s, d_ptr);
#else
core::cuda_optional<int> ptx_version = core::get_ptx_version();
return get_agent_plan<Agent>(ptx_version);
#endif
}
HYDRA_THRUST_RUNTIME_FUNCTION
AgentPlan static get_plan_default()
{
return get_agent_plan<Agent>(sm_arch<0>::type::ver);
}
#endif

HYDRA_THRUST_RUNTIME_FUNCTION
typename core::get_plan<Agent>::type static get_plan(cudaStream_t , void* d_ptr = 0)
{
HYDRA_THRUST_UNUSED_VAR(d_ptr);
core::cuda_optional<int> ptx_version = core::get_ptx_version();
return get_agent_plan<Agent>(ptx_version);
}

HYDRA_THRUST_RUNTIME_FUNCTION
typename core::get_plan<Agent>::type static get_plan()
{
return get_agent_plan<Agent>(lowest_supported_sm_arch::ver);
}

HYDRA_THRUST_RUNTIME_FUNCTION void sync() const
{
if (debug_sync)
{
#ifdef __CUDA_ARCH__
cudaDeviceSynchronize();
#else
cudaStreamSynchronize(stream);
#endif
}
}

template<class K>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
max_blocks_per_sm_impl(K k, int block_threads)
{
int occ;
cudaError_t status = cub::MaxSmOccupancy(occ, k, block_threads);
return cuda_optional<int>(status == cudaSuccess ? occ : -1, status);
}

template <class K>
cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
max_sm_occupancy(K k) const
{
return max_blocks_per_sm_impl(k, plan.block_threads);
}



template<class K>
HYDRA_THRUST_RUNTIME_FUNCTION
void print_info(K k) const
{
if (debug_sync)
{
cuda_optional<int> occ = max_sm_occupancy(k);
core::cuda_optional<int> ptx_version = core::get_ptx_version();
if (count > 0)
{
_CubLog("Invoking %s<<<%u, %d, %d, %lld>>>(), %llu items total, %d items per thread, %d SM occupancy, %d vshmem size, %d ptx_version \n",
name,
grid,
plan.block_threads,
(has_shmem ? (int)plan.shared_memory_size : 0),
(long long)stream,
(long long)count,
plan.items_per_thread,
(int)occ,
(!has_shmem ? (int)plan.shared_memory_size : 0),
(int)ptx_version);
}
else
{
_CubLog("Invoking %s<<<%u, %d, %d, %lld>>>(), %d items per thread, %d SM occupancy, %d vshmem size, %d ptx_version\n",
name,
grid,
plan.block_threads,
(has_shmem ? (int)plan.shared_memory_size : 0),
(long long)stream,
plan.items_per_thread,
(int)occ,
(!has_shmem ? (int)plan.shared_memory_size : 0),
(int)ptx_version);
}
}
}


#if 0
template<class... Args>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
return max_blocks_per_sm_impl(_kernel_agent<Agent, Args...>, plan.block_threads);
}
#else
template<class _0>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0) = _kernel_agent<Agent, _0>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0, _1) = _kernel_agent<Agent, _0, _1>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2) = _kernel_agent<Agent, _0, _1, _2>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2, class _3>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2,_3) = _kernel_agent<Agent, _0, _1, _2,_3>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2, class _3, class _4>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2,_3,_4) = _kernel_agent<Agent, _0, _1, _2,_3,_4>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2, class _3, class _4, class _5>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2,_3,_4,_5) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2, class _3, class _4, class _5, class _6>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2,_3,_4,_5,_6) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_xA) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_xA>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB,_xC) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB,_xC>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB,_xC,_xD) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB,_xC,_xD>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
template<class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
static cuda_optional<int> HYDRA_THRUST_RUNTIME_FUNCTION
get_max_blocks_per_sm(AgentPlan plan)
{
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB,_xC,_xD,_xE) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB,_xC,_xD,_xE>;
return max_blocks_per_sm_impl(ptr, plan.block_threads);
}
#endif



#if 0

template <class... Args>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, Args... args) const
{
assert(has_shmem && vshmem == NULL);
print_info(_kernel_agent<Agent, Args...>);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(_kernel_agent<Agent, Args...>, args...);
}

template <class... Args>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, Args... args) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
print_info(_kernel_agent_vshmem<Agent, Args...>);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(_kernel_agent_vshmem<Agent, Args...>, vshmem, args...);
}

template <class... Args>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(Args... args) const
{
#if __HYDRA_THRUST__TEMPLATE_DEBUG
#ifdef __CUDA_ARCH__
typedef typename Foo<
shm1::v1,
shm1::v2,
shm1::v3,
shm1::v4,
shm1::v5>::t tt;
#endif
#endif
launch_impl(has_enough_shmem_t(),args...);
sync();
}
#else
template <class _0>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0) = _kernel_agent_vshmem<Agent, _0>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0);
}
template <class _0, class _1>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1) = _kernel_agent_vshmem<Agent, _0, _1>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1);
}
template <class _0, class _1, class _2>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2) = _kernel_agent_vshmem<Agent, _0, _1, _2>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2);
}
template <class _0, class _1, class _2, class _3>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2, _3) = _kernel_agent_vshmem<Agent, _0, _1, _2, _3>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2, x3);
}
template <class _0, class _1, class _2, class _3, class _4>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2, _3, _4) = _kernel_agent_vshmem<Agent, _0, _1, _2, _3, _4>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2, x3, x4);
}
template <class _0, class _1, class _2, class _3, class _4, class _5>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2, _3, _4, _5) = _kernel_agent_vshmem<Agent, _0, _1, _2, _3, _4, _5>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2, x3, x4, x5);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2, _3, _4, _5, _6) = _kernel_agent_vshmem<Agent, _0, _1, _2, _3, _4, _5, _6>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2, _3, _4, _5, _6, _7) = _kernel_agent_vshmem<Agent, _0, _1, _2, _3, _4, _5, _6, _7>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2, _3, _4, _5, _6, _7, _8) = _kernel_agent_vshmem<Agent, _0, _1, _2, _3, _4, _5, _6, _7, _8>;
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9) = _kernel_agent_vshmem<Agent, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9,_xA xA) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA) = _kernel_agent_vshmem<Agent, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9,_xA xA,_xB xB) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB) = _kernel_agent_vshmem<Agent, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9,_xA xA,_xB xB,_xC xC) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB, _xC) = _kernel_agent_vshmem<Agent, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB, _xC>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9,_xA xA,_xB xB,_xC xC,_xD xD) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB, _xC, _xD) = _kernel_agent_vshmem<Agent, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB, _xC, _xD>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::false_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9,_xA xA,_xB xB,_xC xC,_xD xD,_xE xE) const
{
assert((has_shmem && vshmem == NULL) || (!has_shmem && vshmem != NULL && shmem_size == 0));
void (*ptr)(char*, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB, _xC, _xD, _xE) = _kernel_agent_vshmem<Agent, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _xA, _xB, _xC, _xD, _xE>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, shmem_size, stream)
.doit(ptr, vshmem, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE);
}


template <class _0>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0) = _kernel_agent<Agent, _0>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0);
}
template <class _0, class _1>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0, _1) = _kernel_agent<Agent, _0, _1>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1);
}
template <class _0, class _1, class _2>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2) = _kernel_agent<Agent, _0, _1, _2>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1, x2);
}
template <class _0, class _1, class _2, class _3>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2,_3) = _kernel_agent<Agent, _0, _1, _2,_3>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1, x2, x3);
}
template <class _0, class _1, class _2, class _3, class _4>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2,_3,_4) = _kernel_agent<Agent, _0, _1, _2,_3,_4>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1, x2, x3, x4);
}
template <class _0, class _1, class _2, class _3, class _4, class _5>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2,_3,_4,_5) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1, x2, x3, x4, x5);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2,_3,_4,_5,_6) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1, x2, x3, x4, x5, x6);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7, x8);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_xA) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_xA>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB,_xC) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB,_xC>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB,_xC,_xD) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB,_xC,_xD>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD);
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch_impl(hydra_thrust::detail::true_type, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD, _xE xE) const
{
assert(has_shmem && vshmem == NULL);
void (*ptr)(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB,_xC,_xD,_xE) = _kernel_agent<Agent, _0, _1, _2,_3,_4,_5,_6,_7,_8,_9,_xA,_xB,_xC,_xD,_xE>;
print_info(ptr);
launcher::triple_chevron(grid, plan.block_threads, plan.shared_memory_size, stream)
.doit(ptr,x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE);
}


template <class _0>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0) const
{
launch_impl(has_enough_shmem_t(), x0);
sync();
}
template <class _0, class _1>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1) const
{
launch_impl(has_enough_shmem_t(), x0, x1);
sync();
}
template <class _0, class _1, class _2>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2);
sync();
}
template <class _0, class _1, class _2, class _3>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2, _3 x3) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2, x3);
sync();
}
template <class _0, class _1, class _2, class _3, class _4>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4);
sync();
}
template <class _0, class _1, class _2, class _3, class _4, class _5>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5);
sync();
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6);
sync();
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7);
sync();
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8);
sync();
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9);
sync();
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA);
sync();
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB);
sync();
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC);
sync();
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD);
sync();
}
template <class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
void HYDRA_THRUST_RUNTIME_FUNCTION
launch(_0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD, _xE xE) const
{
launch_impl(has_enough_shmem_t(), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE);
sync();
}
#endif


};

}    
}
HYDRA_THRUST_END_NS
#endif
