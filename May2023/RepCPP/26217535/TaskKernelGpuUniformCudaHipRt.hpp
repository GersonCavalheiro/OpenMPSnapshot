

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    if !defined(ALPAKA_HOST_ONLY)

#        include <alpaka/core/BoostPredef.hpp>

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

#        include <alpaka/acc/Traits.hpp>
#        include <alpaka/dev/Traits.hpp>
#        include <alpaka/dim/Traits.hpp>
#        include <alpaka/idx/Traits.hpp>
#        include <alpaka/pltf/Traits.hpp>
#        include <alpaka/queue/Traits.hpp>

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#            include <alpaka/core/Cuda.hpp>
#        else
#            include <alpaka/core/Hip.hpp>
#        endif

#        include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#        include <alpaka/core/Decay.hpp>
#        include <alpaka/core/DemangleTypeNames.hpp>
#        include <alpaka/core/RemoveRestrict.hpp>
#        include <alpaka/dev/DevUniformCudaHipRt.hpp>
#        include <alpaka/kernel/Traits.hpp>
#        include <alpaka/queue/QueueUniformCudaHipRtBlocking.hpp>
#        include <alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp>
#        include <alpaka/workdiv/WorkDivMembers.hpp>

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#            include <alpaka/acc/Traits.hpp>
#            include <alpaka/dev/Traits.hpp>
#            include <alpaka/workdiv/WorkDivHelpers.hpp>
#        endif

#        include <alpaka/core/BoostPredef.hpp>

#        include <stdexcept>
#        include <tuple>
#        include <type_traits>
#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#            include <iostream>
#        endif

namespace alpaka
{
namespace detail
{
template<typename TKernelFnObj, typename TApi, typename TAcc, typename TDim, typename TIdx, typename... TArgs>
__global__ void gpuKernel(
Vec<TDim, TIdx> const threadElemExtent,
TKernelFnObj const kernelFnObj,
TArgs... args)
{
#        if BOOST_ARCH_PTX && (BOOST_ARCH_PTX < BOOST_VERSION_NUMBER(2, 0, 0))
#            error "Device capability >= 2.0 is required!"
#        endif

const TAcc acc(threadElemExtent);

#        if !(BOOST_COMP_CLANG_CUDA && BOOST_COMP_CLANG)
static_assert(
std::is_same_v<decltype(kernelFnObj(const_cast<TAcc const&>(acc), args...)), void>,
"The TKernelFnObj is required to return void!");
#        endif
kernelFnObj(const_cast<TAcc const&>(acc), args...);
}
} 

namespace uniform_cuda_hip
{
namespace detail
{
template<typename TDim, typename TIdx>
ALPAKA_FN_HOST auto checkVecOnly3Dim(Vec<TDim, TIdx> const& vec) -> void
{
if constexpr(TDim::value > 0)
{
for(auto i = std::min(typename TDim::value_type{3}, TDim::value); i < TDim::value; ++i)
{
if(vec[TDim::value - 1u - i] != 1)
{
throw std::runtime_error(
"The CUDA/HIP accelerator supports a maximum of 3 dimensions. All "
"work division extents of the dimensions higher 3 have to be 1!");
}
}
}
}

template<typename TDim, typename TIdx>
ALPAKA_FN_HOST auto convertVecToUniformCudaHipDim(Vec<TDim, TIdx> const& vec) -> dim3
{
dim3 dim(1, 1, 1);
if constexpr(TDim::value >= 1)
dim.x = static_cast<unsigned>(vec[TDim::value - 1u]);
if constexpr(TDim::value >= 2)
dim.y = static_cast<unsigned>(vec[TDim::value - 2u]);
if constexpr(TDim::value >= 3)
dim.z = static_cast<unsigned>(vec[TDim::value - 3u]);
checkVecOnly3Dim(vec);
return dim;
}
} 
} 

template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
class TaskKernelGpuUniformCudaHipRt final : public WorkDivMembers<TDim, TIdx>
{
public:
template<typename TWorkDiv>
ALPAKA_FN_HOST TaskKernelGpuUniformCudaHipRt(
TWorkDiv&& workDiv,
TKernelFnObj const& kernelFnObj,
TArgs&&... args)
: WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
, m_kernelFnObj(kernelFnObj)
, m_args(std::forward<TArgs>(args)...)
{
static_assert(
Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
"The work division and the execution task have to be of the same dimensionality!");
}

TKernelFnObj m_kernelFnObj;
std::tuple<remove_restrict_t<std::decay_t<TArgs>>...> m_args;
};

namespace trait
{
template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct AccType<TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = AccGpuUniformCudaHipRt<TApi, TDim, TIdx>;
};

template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct DevType<TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = DevUniformCudaHipRt<TApi>;
};

template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct DimType<TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = TDim;
};

template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct PltfType<TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = PltfUniformCudaHipRt<TApi>;
};

template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct IdxType<TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = TIdx;
};

template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct Enqueue<
QueueUniformCudaHipRtNonBlocking<TApi>,
TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
{
ALPAKA_FN_HOST static auto enqueue(
QueueUniformCudaHipRtNonBlocking<TApi>& queue,
TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
#        endif
auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(task);
auto const blockThreadExtent = getWorkDiv<Block, Threads>(task);
auto const threadElemExtent = getWorkDiv<Thread, Elems>(task);

dim3 const gridDim = uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(gridBlockExtent);
dim3 const blockDim = uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(blockThreadExtent);
uniform_cuda_hip::detail::checkVecOnly3Dim(threadElemExtent);

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
std::cout << __func__ << " gridDim: " << gridDim.z << " " << gridDim.y << " " << gridDim.x
<< " blockDim: " << blockDim.z << " " << blockDim.y << " " << blockDim.x << std::endl;
#        endif

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
if(!isValidWorkDiv<TAcc>(getDev(queue), task))
{
throw std::runtime_error(
"The given work division is not valid or not supported by the device of type "
+ getAccName<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>() + "!");
}
#        endif

auto const blockSharedMemDynSizeBytes = std::apply(
[&](remove_restrict_t<ALPAKA_DECAY_T(TArgs)> const&... args) {
return getBlockSharedMemDynSizeBytes<TAcc>(
task.m_kernelFnObj,
blockThreadExtent,
threadElemExtent,
args...);
},
task.m_args);

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
std::cout << __func__ << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
<< std::endl;
#        endif
auto kernelName = alpaka::detail::
gpuKernel<TKernelFnObj, TApi, TAcc, TDim, TIdx, remove_restrict_t<std::decay_t<TArgs>>...>;

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
typename TApi::FuncAttributes_t funcAttrs;
TApi::funcGetAttributes(&funcAttrs, kernelName);
std::cout << __func__ << " binaryVersion: " << funcAttrs.binaryVersion
<< " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
<< " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
<< " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
<< " numRegs: " << funcAttrs.numRegs << " ptxVersion: " << funcAttrs.ptxVersion
<< " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B" << std::endl;
#        endif

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(queue.m_spQueueImpl->m_dev.getNativeHandle()));
std::apply(
[&](remove_restrict_t<ALPAKA_DECAY_T(TArgs)> const&... args)
{
kernelName<<<
gridDim,
blockDim,
static_cast<std::size_t>(blockSharedMemDynSizeBytes),
queue.getNativeHandle()>>>(threadElemExtent, task.m_kernelFnObj, args...);
},
task.m_args);

if constexpr(ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL)
{
std::ignore = TApi::streamSynchronize(queue.getNativeHandle());
auto const msg = std::string{
"'execution of kernel: '" + std::string{core::demangled<TKernelFnObj>} + "' failed with"};
::alpaka::uniform_cuda_hip::detail::rtCheckLastError<TApi, true>(msg.c_str(), __FILE__, __LINE__);
}
}
};

template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct Enqueue<
QueueUniformCudaHipRtBlocking<TApi>,
TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
{
ALPAKA_FN_HOST static auto enqueue(
QueueUniformCudaHipRtBlocking<TApi>& queue,
TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
#        endif
auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(task);
auto const blockThreadExtent = getWorkDiv<Block, Threads>(task);
auto const threadElemExtent = getWorkDiv<Thread, Elems>(task);

dim3 const gridDim = uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(gridBlockExtent);
dim3 const blockDim = uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(blockThreadExtent);
uniform_cuda_hip::detail::checkVecOnly3Dim(threadElemExtent);

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
std::cout << __func__ << "gridDim: " << gridDim.z << " " << gridDim.y << " " << gridDim.x << std::endl;
std::cout << __func__ << "blockDim: " << blockDim.z << " " << blockDim.y << " " << blockDim.x
<< std::endl;
#        endif

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
if(!isValidWorkDiv<TAcc>(getDev(queue), task))
{
throw std::runtime_error(
"The given work division is not valid or not supported by the device of type "
+ getAccName<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>() + "!");
}
#        endif

auto const blockSharedMemDynSizeBytes = std::apply(
[&](remove_restrict_t<ALPAKA_DECAY_T(TArgs)> const&... args) {
return getBlockSharedMemDynSizeBytes<TAcc>(
task.m_kernelFnObj,
blockThreadExtent,
threadElemExtent,
args...);
},
task.m_args);

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
std::cout << __func__ << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
<< std::endl;
#        endif

auto kernelName = alpaka::detail::
gpuKernel<TKernelFnObj, TApi, TAcc, TDim, TIdx, remove_restrict_t<std::decay_t<TArgs>>...>;
#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
typename TApi::FuncAttributes_t funcAttrs;
TApi::funcGetAttributes(&funcAttrs, kernelName);
std::cout << __func__ << " binaryVersion: " << funcAttrs.binaryVersion
<< " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
<< " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
<< " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
<< " numRegs: " << funcAttrs.numRegs << " ptxVersion: " << funcAttrs.ptxVersion
<< " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B" << std::endl;
#        endif

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(queue.m_spQueueImpl->m_dev.getNativeHandle()));

std::apply(
[&](remove_restrict_t<ALPAKA_DECAY_T(TArgs)> const&... args)
{
kernelName<<<
gridDim,
blockDim,
static_cast<std::size_t>(blockSharedMemDynSizeBytes),
queue.getNativeHandle()>>>(threadElemExtent, task.m_kernelFnObj, args...);
},
task.m_args);

std::ignore = TApi::streamSynchronize(queue.getNativeHandle());
if constexpr(ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL)
{
auto const msg
= std::string{"'execution of kernel: '" + core::demangled<TKernelFnObj> + "' failed with"};
::alpaka::uniform_cuda_hip::detail::rtCheckLastError<TApi, true>(msg.c_str(), __FILE__, __LINE__);
}
}
};
} 
} 

#    endif

#endif
