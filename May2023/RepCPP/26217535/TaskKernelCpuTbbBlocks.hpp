

#pragma once

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

#    include <alpaka/acc/AccCpuTbbBlocks.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/idx/MapIdx.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/meta/NdLoop.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <functional>
#    include <stdexcept>
#    include <tuple>
#    include <type_traits>
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#        include <iostream>
#    endif

#    include <tbb/blocked_range.h>
#    include <tbb/parallel_for.h>
#    include <tbb/task_group.h>

namespace alpaka
{
template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
class TaskKernelCpuTbbBlocks final : public WorkDivMembers<TDim, TIdx>
{
public:
template<typename TWorkDiv>
ALPAKA_FN_HOST TaskKernelCpuTbbBlocks(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
: WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
, m_kernelFnObj(kernelFnObj)
, m_args(std::forward<TArgs>(args)...)
{
static_assert(
Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
"The work division and the execution task have to be of the same dimensionality!");
}

ALPAKA_FN_HOST auto operator()() const -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(*this);
auto const blockThreadExtent = getWorkDiv<Block, Threads>(*this);
auto const threadElemExtent = getWorkDiv<Thread, Elems>(*this);

auto const blockSharedMemDynSizeBytes = std::apply(
[&](ALPAKA_DECAY_T(TArgs) const&... args)
{
return getBlockSharedMemDynSizeBytes<AccCpuTbbBlocks<TDim, TIdx>>(
m_kernelFnObj,
blockThreadExtent,
threadElemExtent,
args...);
},
m_args);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
std::cout << __func__ << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
<< std::endl;
#    endif

TIdx const numBlocksInGrid = gridBlockExtent.prod();

if(blockThreadExtent.prod() != static_cast<TIdx>(1u))
{
throw std::runtime_error("A block for the TBB accelerator can only ever have one single thread!");
}

tbb::this_task_arena::isolate(
[&]
{
tbb::parallel_for(
static_cast<TIdx>(0),
static_cast<TIdx>(numBlocksInGrid),
[&](TIdx i)
{
AccCpuTbbBlocks<TDim, TIdx> acc(
*static_cast<WorkDivMembers<TDim, TIdx> const*>(this),
blockSharedMemDynSizeBytes);

acc.m_gridBlockIdx
= mapIdx<TDim::value>(Vec<DimInt<1u>, TIdx>(static_cast<TIdx>(i)), gridBlockExtent);

std::apply(m_kernelFnObj, std::tuple_cat(std::tie(acc), m_args));

freeSharedVars(acc);
});
});
}

private:
TKernelFnObj m_kernelFnObj;
std::tuple<std::decay_t<TArgs>...> m_args;
};

namespace trait
{
template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct AccType<TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = AccCpuTbbBlocks<TDim, TIdx>;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct DevType<TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = DevCpu;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct DimType<TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = TDim;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct PltfType<TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = PltfCpu;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct IdxType<TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = TIdx;
};
} 
} 

#endif
