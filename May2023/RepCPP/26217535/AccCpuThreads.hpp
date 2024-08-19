

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

#    include <alpaka/atomic/AtomicCpu.hpp>
#    include <alpaka/atomic/AtomicHierarchy.hpp>
#    include <alpaka/block/shared/dyn/BlockSharedMemDynMember.hpp>
#    include <alpaka/block/shared/st/BlockSharedMemStMemberMasterSync.hpp>
#    include <alpaka/block/sync/BlockSyncBarrierThread.hpp>
#    include <alpaka/core/DemangleTypeNames.hpp>
#    include <alpaka/idx/bt/IdxBtRefThreadIdMap.hpp>
#    include <alpaka/idx/gb/IdxGbRef.hpp>
#    include <alpaka/intrinsic/IntrinsicCpu.hpp>
#    include <alpaka/math/MathStdLib.hpp>
#    include <alpaka/mem/fence/MemFenceCpu.hpp>
#    include <alpaka/rand/RandStdLib.hpp>
#    include <alpaka/warp/WarpSingleThread.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

#    include <alpaka/acc/Tag.hpp>
#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/ClipCast.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevCpu.hpp>

#    include <memory>
#    include <thread>
#    include <typeinfo>

namespace alpaka
{
template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
class TaskKernelCpuThreads;

template<
typename TDim,
typename TIdx>
class AccCpuThreads final :
public WorkDivMembers<TDim, TIdx>,
public gb::IdxGbRef<TDim, TIdx>,
public bt::IdxBtRefThreadIdMap<TDim, TIdx>,
public AtomicHierarchy<
AtomicCpu, 
AtomicCpu, 
AtomicCpu  
>,
public math::MathStdLib,
public BlockSharedMemDynMember<>,
public BlockSharedMemStMemberMasterSync<>,
public BlockSyncBarrierThread<TIdx>,
public IntrinsicCpu,
public MemFenceCpu,
public rand::RandStdLib,
public warp::WarpSingleThread,
public concepts::Implements<ConceptAcc, AccCpuThreads<TDim, TIdx>>
{
static_assert(
sizeof(TIdx) >= sizeof(int),
"Index type is not supported, consider using int or a larger type.");

public:
template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
friend class ::alpaka::TaskKernelCpuThreads;

AccCpuThreads(AccCpuThreads const&) = delete;
AccCpuThreads(AccCpuThreads&&) = delete;
auto operator=(AccCpuThreads const&) -> AccCpuThreads& = delete;
auto operator=(AccCpuThreads&&) -> AccCpuThreads& = delete;

private:
template<typename TWorkDiv>
ALPAKA_FN_HOST AccCpuThreads(TWorkDiv const& workDiv, std::size_t const& blockSharedMemDynSizeBytes)
: WorkDivMembers<TDim, TIdx>(workDiv)
, gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx)
, bt::IdxBtRefThreadIdMap<TDim, TIdx>(m_threadToIndexMap)
, AtomicHierarchy<
AtomicCpu, 
AtomicCpu, 
AtomicCpu 
>()
, math::MathStdLib()
, BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes)
, BlockSharedMemStMemberMasterSync<>(
staticMemBegin(),
staticMemCapacity(),
[this]() { syncBlockThreads(*this); },
[this]() noexcept { return (m_idMasterThread == std::this_thread::get_id()); })
, BlockSyncBarrierThread<TIdx>(getWorkDiv<Block, Threads>(workDiv).prod())
, MemFenceCpu()
, rand::RandStdLib()
, m_gridBlockIdx(Vec<TDim, TIdx>::zeros())
{
}

private:
std::mutex mutable m_mtxMapInsert; 
typename bt::IdxBtRefThreadIdMap<TDim, TIdx>::
ThreadIdToIdxMap mutable m_threadToIndexMap; 
Vec<TDim, TIdx> mutable m_gridBlockIdx; 

std::thread::id mutable m_idMasterThread; 
};

namespace trait
{
template<typename TDim, typename TIdx>
struct AccType<AccCpuThreads<TDim, TIdx>>
{
using type = AccCpuThreads<TDim, TIdx>;
};
template<typename TDim, typename TIdx>
struct GetAccDevProps<AccCpuThreads<TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getAccDevProps(DevCpu const& dev) -> AccDevProps<TDim, TIdx>
{
#    ifdef ALPAKA_CI
auto const blockThreadCountMax(static_cast<TIdx>(8));
#    else
auto const blockThreadCountMax = std::max(
static_cast<TIdx>(1),
alpaka::core::clipCast<TIdx>(std::thread::hardware_concurrency() * 8));
#    endif
return {
static_cast<TIdx>(1),
Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
std::numeric_limits<TIdx>::max(),
Vec<TDim, TIdx>::all(blockThreadCountMax),
blockThreadCountMax,
Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
std::numeric_limits<TIdx>::max(),
getMemBytes(dev)};
}
};
template<typename TDim, typename TIdx>
struct GetAccName<AccCpuThreads<TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getAccName() -> std::string
{
return "AccCpuThreads<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
}
};

template<typename TDim, typename TIdx>
struct DevType<AccCpuThreads<TDim, TIdx>>
{
using type = DevCpu;
};

template<typename TDim, typename TIdx>
struct DimType<AccCpuThreads<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
struct CreateTaskKernel<AccCpuThreads<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
{
ALPAKA_FN_HOST static auto createTaskKernel(
TWorkDiv const& workDiv,
TKernelFnObj const& kernelFnObj,
TArgs&&... args)
{
return TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>(
workDiv,
kernelFnObj,
std::forward<TArgs>(args)...);
}
};

template<typename TDim, typename TIdx>
struct PltfType<AccCpuThreads<TDim, TIdx>>
{
using type = PltfCpu;
};

template<typename TDim, typename TIdx>
struct IdxType<AccCpuThreads<TDim, TIdx>>
{
using type = TIdx;
};

template<typename TDim, typename TIdx>
struct AccToTag<alpaka::AccCpuThreads<TDim, TIdx>>
{
using type = alpaka::TagCpuThreads;
};

template<typename TDim, typename TIdx>
struct TagToAcc<alpaka::TagCpuThreads, TDim, TIdx>
{
using type = alpaka::AccCpuThreads<TDim, TIdx>;
};
} 
} 

#endif
