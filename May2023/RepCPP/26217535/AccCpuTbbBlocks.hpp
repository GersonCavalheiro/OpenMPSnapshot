

#pragma once

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#    include <alpaka/atomic/AtomicCpu.hpp>
#    include <alpaka/atomic/AtomicHierarchy.hpp>
#    include <alpaka/atomic/AtomicNoOp.hpp>
#    include <alpaka/block/shared/dyn/BlockSharedMemDynMember.hpp>
#    include <alpaka/block/shared/st/BlockSharedMemStMember.hpp>
#    include <alpaka/block/sync/BlockSyncNoOp.hpp>
#    include <alpaka/core/DemangleTypeNames.hpp>
#    include <alpaka/idx/bt/IdxBtZero.hpp>
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
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevCpu.hpp>

#    include <memory>
#    include <typeinfo>

namespace alpaka
{
template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
class TaskKernelCpuTbbBlocks;

template<
typename TDim,
typename TIdx>
class AccCpuTbbBlocks final :
public WorkDivMembers<TDim, TIdx>,
public gb::IdxGbRef<TDim, TIdx>,
public bt::IdxBtZero<TDim, TIdx>,
public AtomicHierarchy<
AtomicCpu, 
AtomicCpu, 
AtomicNoOp         
>,
public math::MathStdLib,
public BlockSharedMemDynMember<>,
public BlockSharedMemStMember<>,
public BlockSyncNoOp,
public IntrinsicCpu,
public MemFenceCpu,
public rand::RandStdLib,
public warp::WarpSingleThread,
public concepts::Implements<ConceptAcc, AccCpuTbbBlocks<TDim, TIdx>>
{
static_assert(
sizeof(TIdx) >= sizeof(int),
"Index type is not supported, consider using int or a larger type.");

public:
template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
friend class ::alpaka::TaskKernelCpuTbbBlocks;

AccCpuTbbBlocks(AccCpuTbbBlocks const&) = delete;
AccCpuTbbBlocks(AccCpuTbbBlocks&&) = delete;
auto operator=(AccCpuTbbBlocks const&) -> AccCpuTbbBlocks& = delete;
auto operator=(AccCpuTbbBlocks&&) -> AccCpuTbbBlocks& = delete;

private:
template<typename TWorkDiv>
ALPAKA_FN_HOST AccCpuTbbBlocks(TWorkDiv const& workDiv, std::size_t const& blockSharedMemDynSizeBytes)
: WorkDivMembers<TDim, TIdx>(workDiv)
, gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx)
, bt::IdxBtZero<TDim, TIdx>()
, AtomicHierarchy<
AtomicCpu, 
AtomicCpu, 
AtomicNoOp 
>()
, math::MathStdLib()
, BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes)
, BlockSharedMemStMember<>(staticMemBegin(), staticMemCapacity())
, BlockSyncNoOp()
, MemFenceCpu()
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
struct AccType<AccCpuTbbBlocks<TDim, TIdx>>
{
using type = AccCpuTbbBlocks<TDim, TIdx>;
};
template<typename TDim, typename TIdx>
struct GetAccDevProps<AccCpuTbbBlocks<TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getAccDevProps(DevCpu const& ) -> AccDevProps<TDim, TIdx>
{
return {
static_cast<TIdx>(1),
Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
std::numeric_limits<TIdx>::max(),
Vec<TDim, TIdx>::ones(),
static_cast<TIdx>(1),
Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
std::numeric_limits<TIdx>::max(),
static_cast<size_t>(AccCpuTbbBlocks<TDim, TIdx>::staticAllocBytes())};
}
};
template<typename TDim, typename TIdx>
struct GetAccName<AccCpuTbbBlocks<TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getAccName() -> std::string
{
return "AccCpuTbbBlocks<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
}
};

template<typename TDim, typename TIdx>
struct DevType<AccCpuTbbBlocks<TDim, TIdx>>
{
using type = DevCpu;
};

template<typename TDim, typename TIdx>
struct DimType<AccCpuTbbBlocks<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
struct CreateTaskKernel<AccCpuTbbBlocks<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
{
ALPAKA_FN_HOST static auto createTaskKernel(
TWorkDiv const& workDiv,
TKernelFnObj const& kernelFnObj,
TArgs&&... args)
{
return TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>(
workDiv,
kernelFnObj,
std::forward<TArgs>(args)...);
}
};

template<typename TDim, typename TIdx>
struct PltfType<AccCpuTbbBlocks<TDim, TIdx>>
{
using type = PltfCpu;
};

template<typename TDim, typename TIdx>
struct IdxType<AccCpuTbbBlocks<TDim, TIdx>>
{
using type = TIdx;
};

template<typename TDim, typename TIdx>
struct AccToTag<alpaka::AccCpuTbbBlocks<TDim, TIdx>>
{
using type = alpaka::TagCpuTbbBlocks;
};

template<typename TDim, typename TIdx>
struct TagToAcc<alpaka::TagCpuTbbBlocks, TDim, TIdx>
{
using type = alpaka::AccCpuTbbBlocks<TDim, TIdx>;
};
} 
} 

#endif
