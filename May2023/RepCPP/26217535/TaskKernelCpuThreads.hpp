

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

#    include <alpaka/acc/AccCpuThreads.hpp>
#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/ConcurrentExecPool.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/meta/NdLoop.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <algorithm>
#    include <functional>
#    include <future>
#    include <thread>
#    include <tuple>
#    include <type_traits>
#    include <vector>
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#        include <iostream>
#    endif

namespace alpaka
{
template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
class TaskKernelCpuThreads final : public WorkDivMembers<TDim, TIdx>
{
private:
struct ThreadPoolYield
{
ALPAKA_FN_HOST static auto yield() -> void
{
std::this_thread::yield();
}
};
using ThreadPool = alpaka::core::detail::ConcurrentExecPool<
TIdx,
std::thread, 
std::promise, 
ThreadPoolYield>; 

public:
template<typename TWorkDiv>
ALPAKA_FN_HOST TaskKernelCpuThreads(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
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
return getBlockSharedMemDynSizeBytes<AccCpuThreads<TDim, TIdx>>(
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
AccCpuThreads<TDim, TIdx> acc(
*static_cast<WorkDivMembers<TDim, TIdx> const*>(this),
blockSharedMemDynSizeBytes);

auto const blockThreadCount = blockThreadExtent.prod();
ThreadPool threadPool(blockThreadCount);

auto const boundGridBlockExecHost = std::apply(
[this, &acc, &blockThreadExtent, &threadPool](ALPAKA_DECAY_T(TArgs) const&... args)
{
return std::bind(
&TaskKernelCpuThreads::gridBlockExecHost,
std::ref(acc),
std::placeholders::_1,
std::ref(blockThreadExtent),
std::ref(threadPool),
std::ref(m_kernelFnObj),
std::ref(args)...);
},
m_args);

meta::ndLoopIncIdx(gridBlockExtent, boundGridBlockExecHost);
}

private:
ALPAKA_FN_HOST static auto gridBlockExecHost(
AccCpuThreads<TDim, TIdx>& acc,
Vec<TDim, TIdx> const& gridBlockIdx,
Vec<TDim, TIdx> const& blockThreadExtent,
ThreadPool& threadPool,
TKernelFnObj const& kernelFnObj,
std::decay_t<TArgs> const&... args) -> void
{
std::vector<std::future<void>> futuresInBlock;

acc.m_gridBlockIdx = gridBlockIdx;

auto boundBlockThreadExecHost = std::bind(
&TaskKernelCpuThreads::blockThreadExecHost,
std::ref(acc),
std::ref(futuresInBlock),
std::placeholders::_1,
std::ref(threadPool),
std::ref(kernelFnObj),
std::ref(args)...);
meta::ndLoopIncIdx(blockThreadExtent, boundBlockThreadExecHost);
for(auto& t : futuresInBlock)
t.wait();
futuresInBlock.clear();

acc.m_threadToIndexMap.clear();

freeSharedVars(acc);
}
ALPAKA_FN_HOST static auto blockThreadExecHost(
AccCpuThreads<TDim, TIdx>& acc,
std::vector<std::future<void>>& futuresInBlock,
Vec<TDim, TIdx> const& blockThreadIdx,
ThreadPool& threadPool,
TKernelFnObj const& kernelFnObj,
std::decay_t<TArgs> const&... args) -> void
{
auto boundBlockThreadExecAcc
= [&, blockThreadIdx]() { blockThreadExecAcc(acc, blockThreadIdx, kernelFnObj, args...); };
futuresInBlock.emplace_back(threadPool.enqueueTask(boundBlockThreadExecAcc));
}
ALPAKA_FN_HOST static auto blockThreadExecAcc(
AccCpuThreads<TDim, TIdx>& acc,
Vec<TDim, TIdx> const& blockThreadIdx,
TKernelFnObj const& kernelFnObj,
std::decay_t<TArgs> const&... args) -> void
{
auto const threadId(std::this_thread::get_id());

if(blockThreadIdx.sum() == 0)
{
acc.m_idMasterThread = threadId;
}

{
std::lock_guard<std::mutex> lock(acc.m_mtxMapInsert);

acc.m_threadToIndexMap.emplace(threadId, blockThreadIdx);
}

syncBlockThreads(acc);

kernelFnObj(const_cast<AccCpuThreads<TDim, TIdx> const&>(acc), args...);

syncBlockThreads(acc);
}

TKernelFnObj m_kernelFnObj;
std::tuple<std::decay_t<TArgs>...> m_args;
};

namespace trait
{
template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct AccType<TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = AccCpuThreads<TDim, TIdx>;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct DevType<TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = DevCpu;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct DimType<TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = TDim;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct PltfType<TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = PltfCpu;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct IdxType<TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = TIdx;
};
} 
} 

#endif
