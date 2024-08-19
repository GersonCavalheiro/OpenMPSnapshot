

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/atomic/AtomicHierarchy.hpp>
#    include <alpaka/atomic/AtomicUniformCudaHipBuiltIn.hpp>
#    include <alpaka/block/shared/dyn/BlockSharedMemDynUniformCudaHipBuiltIn.hpp>
#    include <alpaka/block/shared/st/BlockSharedMemStUniformCudaHipBuiltIn.hpp>
#    include <alpaka/block/sync/BlockSyncUniformCudaHipBuiltIn.hpp>
#    include <alpaka/core/DemangleTypeNames.hpp>
#    include <alpaka/idx/bt/IdxBtUniformCudaHipBuiltIn.hpp>
#    include <alpaka/idx/gb/IdxGbUniformCudaHipBuiltIn.hpp>
#    include <alpaka/intrinsic/IntrinsicUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/MathUniformCudaHipBuiltIn.hpp>
#    include <alpaka/mem/fence/MemFenceUniformCudaHipBuiltIn.hpp>
#    include <alpaka/rand/RandUniformCudaHipRand.hpp>
#    include <alpaka/warp/WarpUniformCudaHipBuiltIn.hpp>
#    include <alpaka/workdiv/WorkDivUniformCudaHipBuiltIn.hpp>

#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

#    include <alpaka/core/ClipCast.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/Cuda.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>

#    include <typeinfo>

namespace alpaka
{
template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
class TaskKernelGpuUniformCudaHipRt;

template<
typename TApi,
typename TDim,
typename TIdx>
class AccGpuUniformCudaHipRt final :
public WorkDivUniformCudaHipBuiltIn<TDim, TIdx>,
public gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>,
public bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx>,
public AtomicHierarchy<
AtomicUniformCudaHipBuiltIn, 
AtomicUniformCudaHipBuiltIn, 
AtomicUniformCudaHipBuiltIn  
>,
public math::MathUniformCudaHipBuiltIn,
public BlockSharedMemDynUniformCudaHipBuiltIn,
public BlockSharedMemStUniformCudaHipBuiltIn,
public BlockSyncUniformCudaHipBuiltIn,
public IntrinsicUniformCudaHipBuiltIn,
public MemFenceUniformCudaHipBuiltIn,
public rand::RandUniformCudaHipRand<TApi>,
public warp::WarpUniformCudaHipBuiltIn,
public concepts::Implements<ConceptAcc, AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
{
static_assert(
sizeof(TIdx) >= sizeof(int),
"Index type is not supported, consider using int or a larger type.");

public:
AccGpuUniformCudaHipRt(AccGpuUniformCudaHipRt const&) = delete;
AccGpuUniformCudaHipRt(AccGpuUniformCudaHipRt&&) = delete;
auto operator=(AccGpuUniformCudaHipRt const&) -> AccGpuUniformCudaHipRt& = delete;
auto operator=(AccGpuUniformCudaHipRt&&) -> AccGpuUniformCudaHipRt& = delete;

ALPAKA_FN_HOST_ACC AccGpuUniformCudaHipRt(Vec<TDim, TIdx> const& threadElemExtent)
: WorkDivUniformCudaHipBuiltIn<TDim, TIdx>(threadElemExtent)
, gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>()
, bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx>()
, AtomicHierarchy<
AtomicUniformCudaHipBuiltIn, 
AtomicUniformCudaHipBuiltIn, 
AtomicUniformCudaHipBuiltIn 
>()
, math::MathUniformCudaHipBuiltIn()
, BlockSharedMemDynUniformCudaHipBuiltIn()
, BlockSharedMemStUniformCudaHipBuiltIn()
, BlockSyncUniformCudaHipBuiltIn()
, MemFenceUniformCudaHipBuiltIn()
, rand::RandUniformCudaHipRand<TApi>()
{
}
};

namespace trait
{
template<typename TApi, typename TDim, typename TIdx>
struct AccType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
{
using type = AccGpuUniformCudaHipRt<TApi, TDim, TIdx>;
};

template<typename TApi, typename TDim, typename TIdx>
struct GetAccDevProps<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getAccDevProps(DevUniformCudaHipRt<TApi> const& dev) -> AccDevProps<TDim, TIdx>
{
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
int multiProcessorCount = {};
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
&multiProcessorCount,
TApi::deviceAttributeMultiprocessorCount,
dev.getNativeHandle()));

int maxGridSize[3] = {};
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
&maxGridSize[0],
TApi::deviceAttributeMaxGridDimX,
dev.getNativeHandle()));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
&maxGridSize[1],
TApi::deviceAttributeMaxGridDimY,
dev.getNativeHandle()));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
&maxGridSize[2],
TApi::deviceAttributeMaxGridDimZ,
dev.getNativeHandle()));

int maxBlockDim[3] = {};
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
&maxBlockDim[0],
TApi::deviceAttributeMaxBlockDimX,
dev.getNativeHandle()));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
&maxBlockDim[1],
TApi::deviceAttributeMaxBlockDimY,
dev.getNativeHandle()));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
&maxBlockDim[2],
TApi::deviceAttributeMaxBlockDimZ,
dev.getNativeHandle()));

int maxThreadsPerBlock = {};
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
&maxThreadsPerBlock,
TApi::deviceAttributeMaxThreadsPerBlock,
dev.getNativeHandle()));

int sharedMemSizeBytes = {};
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
&sharedMemSizeBytes,
TApi::deviceAttributeMaxSharedMemoryPerBlock,
dev.getNativeHandle()));

return {
alpaka::core::clipCast<TIdx>(multiProcessorCount),
getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
alpaka::core::clipCast<TIdx>(maxGridSize[2u]),
alpaka::core::clipCast<TIdx>(maxGridSize[1u]),
alpaka::core::clipCast<TIdx>(maxGridSize[0u]))),
std::numeric_limits<TIdx>::max(),
getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
alpaka::core::clipCast<TIdx>(maxBlockDim[2u]),
alpaka::core::clipCast<TIdx>(maxBlockDim[1u]),
alpaka::core::clipCast<TIdx>(maxBlockDim[0u]))),
alpaka::core::clipCast<TIdx>(maxThreadsPerBlock),
Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
std::numeric_limits<TIdx>::max(),
static_cast<size_t>(sharedMemSizeBytes)};

#    else
typename TApi::DeviceProp_t properties;
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getDeviceProperties(&properties, dev.getNativeHandle()));

return {
alpaka::core::clipCast<TIdx>(properties.multiProcessorCount),
getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
alpaka::core::clipCast<TIdx>(properties.maxGridSize[2u]),
alpaka::core::clipCast<TIdx>(properties.maxGridSize[1u]),
alpaka::core::clipCast<TIdx>(properties.maxGridSize[0u]))),
std::numeric_limits<TIdx>::max(),
getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
alpaka::core::clipCast<TIdx>(properties.maxThreadsDim[2u]),
alpaka::core::clipCast<TIdx>(properties.maxThreadsDim[1u]),
alpaka::core::clipCast<TIdx>(properties.maxThreadsDim[0u]))),
alpaka::core::clipCast<TIdx>(properties.maxThreadsPerBlock),
Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
std::numeric_limits<TIdx>::max(),
static_cast<size_t>(properties.sharedMemPerBlock)};
#    endif
}
};

template<typename TApi, typename TDim, typename TIdx>
struct GetAccName<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getAccName() -> std::string
{
return std::string("AccGpu") + TApi::name + "Rt<" + std::to_string(TDim::value) + ","
+ core::demangled<TIdx> + ">";
}
};

template<typename TApi, typename TDim, typename TIdx>
struct DevType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
{
using type = DevUniformCudaHipRt<TApi>;
};

template<typename TApi, typename TDim, typename TIdx>
struct DimType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
{
using type = TDim;
};
} 

namespace detail
{
template<typename TApi, typename TDim, typename TIdx>
struct CheckFnReturnType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
{
template<typename TKernelFnObj, typename... TArgs>
void operator()(TKernelFnObj const&, TArgs const&...)
{
}
};
} 

namespace trait
{
template<
typename TApi,
typename TDim,
typename TIdx,
typename TWorkDiv,
typename TKernelFnObj,
typename... TArgs>
struct CreateTaskKernel<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
{
ALPAKA_FN_HOST static auto createTaskKernel(
TWorkDiv const& workDiv,
TKernelFnObj const& kernelFnObj,
TArgs&&... args)
{
return TaskKernelGpuUniformCudaHipRt<
TApi,
AccGpuUniformCudaHipRt<TApi, TDim, TIdx>,
TDim,
TIdx,
TKernelFnObj,
TArgs...>(workDiv, kernelFnObj, std::forward<TArgs>(args)...);
}
};

template<typename TApi, typename TDim, typename TIdx>
struct PltfType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
{
using type = PltfUniformCudaHipRt<TApi>;
};

template<typename TApi, typename TDim, typename TIdx>
struct IdxType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
{
using type = TIdx;
};
} 
} 

#endif
