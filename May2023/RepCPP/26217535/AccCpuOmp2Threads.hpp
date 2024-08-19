

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

#    include <alpaka/atomic/AtomicCpu.hpp>
#    include <alpaka/atomic/AtomicHierarchy.hpp>
#    include <alpaka/atomic/AtomicOmpBuiltIn.hpp>
#    include <alpaka/block/shared/dyn/BlockSharedMemDynMember.hpp>
#    include <alpaka/block/shared/st/BlockSharedMemStMemberMasterSync.hpp>
#    include <alpaka/block/sync/BlockSyncBarrierOmp.hpp>
#    include <alpaka/core/DemangleTypeNames.hpp>
#    include <alpaka/idx/bt/IdxBtOmp.hpp>
#    include <alpaka/idx/gb/IdxGbRef.hpp>
#    include <alpaka/intrinsic/IntrinsicCpu.hpp>
#    include <alpaka/math/MathStdLib.hpp>
#    include <alpaka/mem/fence/MemFenceOmp2Threads.hpp>
#    include <alpaka/rand/RandStdLib.hpp>
#    include <alpaka/warp/WarpSingleThread.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

#    include <alpaka/acc/Tag.hpp>
#    include <alpaka/core/ClipCast.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevCpu.hpp>

#    include <omp.h>

#    include <limits>
#    include <typeinfo>

namespace alpaka
{
template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
class TaskKernelCpuOmp2Threads;

template<
typename TDim,
typename TIdx>
class AccCpuOmp2Threads final :
public WorkDivMembers<TDim, TIdx>,
public gb::IdxGbRef<TDim, TIdx>,
public bt::IdxBtOmp<TDim, TIdx>,
public AtomicHierarchy<
AtomicCpu,   
AtomicOmpBuiltIn,    
AtomicOmpBuiltIn     
>,
public math::MathStdLib,
public BlockSharedMemDynMember<>,
public BlockSharedMemStMemberMasterSync<>,
public BlockSyncBarrierOmp,
public IntrinsicCpu,
public MemFenceOmp2Threads,
public rand::RandStdLib,
public warp::WarpSingleThread,
public concepts::Implements<ConceptAcc, AccCpuOmp2Threads<TDim, TIdx>>
{
static_assert(
sizeof(TIdx) >= sizeof(int),
"Index type is not supported, consider using int or a larger type.");

public:
template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
friend class ::alpaka::TaskKernelCpuOmp2Threads;

AccCpuOmp2Threads(AccCpuOmp2Threads const&) = delete;
AccCpuOmp2Threads(AccCpuOmp2Threads&&) = delete;
auto operator=(AccCpuOmp2Threads const&) -> AccCpuOmp2Threads& = delete;
auto operator=(AccCpuOmp2Threads&&) -> AccCpuOmp2Threads& = delete;

private:
template<typename TWorkDiv>
ALPAKA_FN_HOST AccCpuOmp2Threads(TWorkDiv const& workDiv, std::size_t const& blockSharedMemDynSizeBytes)
: WorkDivMembers<TDim, TIdx>(workDiv)
, gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx)
, bt::IdxBtOmp<TDim, TIdx>()
, AtomicHierarchy<
AtomicCpu, 
AtomicOmpBuiltIn, 
AtomicOmpBuiltIn 
>()
, math::MathStdLib()
, BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes)
, BlockSharedMemStMemberMasterSync<>(
staticMemBegin(),
staticMemCapacity(),
[this]() { syncBlockThreads(*this); },
[]() noexcept { return (::omp_get_thread_num() == 0); })
, BlockSyncBarrierOmp()
, MemFenceOmp2Threads()
, rand::RandStdLib()
, m_gridBlockIdx(Vec<TDim, TIdx>::zeros())
{
}

private:
Vec<TDim, TIdx> mutable m_gridBlockIdx; 
};

namespace trait
{
template<typename TDim, typename TIdx>
struct AccType<AccCpuOmp2Threads<TDim, TIdx>>
{
using type = AccCpuOmp2Threads<TDim, TIdx>;
};
template<typename TDim, typename TIdx>
struct GetAccDevProps<AccCpuOmp2Threads<TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getAccDevProps(DevCpu const& dev) -> alpaka::AccDevProps<TDim, TIdx>
{
#    ifdef ALPAKA_CI
auto const blockThreadCountMax = alpaka::core::clipCast<TIdx>(std::min(4, ::omp_get_max_threads()));
#    else
auto const blockThreadCountMax = alpaka::core::clipCast<TIdx>(::omp_get_max_threads());
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
struct GetAccName<AccCpuOmp2Threads<TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getAccName() -> std::string
{
return "AccCpuOmp2Threads<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
}
};

template<typename TDim, typename TIdx>
struct DevType<AccCpuOmp2Threads<TDim, TIdx>>
{
using type = DevCpu;
};

template<typename TDim, typename TIdx>
struct DimType<AccCpuOmp2Threads<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
struct CreateTaskKernel<AccCpuOmp2Threads<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
{
ALPAKA_FN_HOST static auto createTaskKernel(
TWorkDiv const& workDiv,
TKernelFnObj const& kernelFnObj,
TArgs&&... args)
{
return TaskKernelCpuOmp2Threads<TDim, TIdx, TKernelFnObj, TArgs...>(
workDiv,
kernelFnObj,
std::forward<TArgs>(args)...);
}
};

template<typename TDim, typename TIdx>
struct PltfType<AccCpuOmp2Threads<TDim, TIdx>>
{
using type = PltfCpu;
};

template<typename TDim, typename TIdx>
struct IdxType<AccCpuOmp2Threads<TDim, TIdx>>
{
using type = TIdx;
};

template<typename TDim, typename TIdx>
struct AccToTag<alpaka::AccCpuOmp2Threads<TDim, TIdx>>
{
using type = alpaka::TagCpuOmp2Threads;
};

template<typename TDim, typename TIdx>
struct TagToAcc<alpaka::TagCpuOmp2Threads, TDim, TIdx>
{
using type = alpaka::AccCpuOmp2Threads<TDim, TIdx>;
};
} 
} 

#endif
