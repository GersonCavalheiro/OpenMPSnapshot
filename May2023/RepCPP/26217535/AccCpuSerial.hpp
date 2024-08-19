

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

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
#    include <alpaka/mem/fence/MemFenceCpuSerial.hpp>
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
class TaskKernelCpuSerial;

template<
typename TDim,
typename TIdx>
class AccCpuSerial final :
public WorkDivMembers<TDim, TIdx>,
public gb::IdxGbRef<TDim, TIdx>,
public bt::IdxBtZero<TDim, TIdx>,
public AtomicHierarchy<
AtomicCpu, 
AtomicNoOp,        
AtomicNoOp         
>,
public math::MathStdLib,
public BlockSharedMemDynMember<>,
public BlockSharedMemStMember<>,
public BlockSyncNoOp,
public IntrinsicCpu,
public MemFenceCpuSerial,
public rand::RandStdLib,
public warp::WarpSingleThread,
public concepts::Implements<ConceptAcc, AccCpuSerial<TDim, TIdx>>
{
static_assert(
sizeof(TIdx) >= sizeof(int),
"Index type is not supported, consider using int or a larger type.");

public:
template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
friend class ::alpaka::TaskKernelCpuSerial;

AccCpuSerial(AccCpuSerial const&) = delete;
AccCpuSerial(AccCpuSerial&&) = delete;
auto operator=(AccCpuSerial const&) -> AccCpuSerial& = delete;
auto operator=(AccCpuSerial&&) -> AccCpuSerial& = delete;

private:
template<typename TWorkDiv>
ALPAKA_FN_HOST AccCpuSerial(TWorkDiv const& workDiv, size_t const& blockSharedMemDynSizeBytes)
: WorkDivMembers<TDim, TIdx>(workDiv)
, gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx)
, bt::IdxBtZero<TDim, TIdx>()
, AtomicHierarchy<
AtomicCpu, 
AtomicNoOp, 
AtomicNoOp 
>()
, math::MathStdLib()
, BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes)
, BlockSharedMemStMember<>(staticMemBegin(), staticMemCapacity())
, BlockSyncNoOp()
, MemFenceCpuSerial()
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
struct AccType<AccCpuSerial<TDim, TIdx>>
{
using type = AccCpuSerial<TDim, TIdx>;
};
template<typename TDim, typename TIdx>
struct GetAccDevProps<AccCpuSerial<TDim, TIdx>>
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
static_cast<size_t>(AccCpuSerial<TDim, TIdx>::staticAllocBytes())};
}
};
template<typename TDim, typename TIdx>
struct GetAccName<AccCpuSerial<TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getAccName() -> std::string
{
return "AccCpuSerial<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
}
};

template<typename TDim, typename TIdx>
struct DevType<AccCpuSerial<TDim, TIdx>>
{
using type = DevCpu;
};

template<typename TDim, typename TIdx>
struct DimType<AccCpuSerial<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
struct CreateTaskKernel<AccCpuSerial<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
{
ALPAKA_FN_HOST static auto createTaskKernel(
TWorkDiv const& workDiv,
TKernelFnObj const& kernelFnObj,
TArgs&&... args)
{
return TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>(
workDiv,
kernelFnObj,
std::forward<TArgs>(args)...);
}
};

template<typename TDim, typename TIdx>
struct PltfType<AccCpuSerial<TDim, TIdx>>
{
using type = PltfCpu;
};

template<typename TDim, typename TIdx>
struct IdxType<AccCpuSerial<TDim, TIdx>>
{
using type = TIdx;
};

template<typename TDim, typename TIdx>
struct AccToTag<alpaka::AccCpuSerial<TDim, TIdx>>
{
using type = alpaka::TagCpuSerial;
};

template<typename TDim, typename TIdx>
struct TagToAcc<alpaka::TagCpuSerial, TDim, TIdx>
{
using type = alpaka::AccCpuSerial<TDim, TIdx>;
};
} 
} 

#endif
