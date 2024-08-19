

#pragma once

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

#    include <alpaka/acc/AccCpuOmp2Blocks.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/core/OmpSchedule.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/idx/MapIdx.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <omp.h>

#    include <functional>
#    include <stdexcept>
#    include <tuple>
#    include <type_traits>
#    include <utility>
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#        include <iostream>
#    endif

namespace alpaka
{
namespace detail
{
template<typename TKernel, typename TSchedule, omp::Schedule::Kind TScheduleKind>
struct ParallelForImpl;

template<typename TKernel, typename TSchedule>
struct ParallelForImpl<TKernel, TSchedule, omp::Schedule::NoSchedule>
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const&,
TLoopBody&& loopBody,
TIdx const numIterations,
TSchedule const&)
{
#    if _OPENMP < 200805 
std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
std::intmax_t i;
#        pragma omp for nowait
for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait
for(TIdx i = 0; i < numIterations; ++i)
#    endif
{
auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
wrappedLoopBody(i);
}
}
};



template<typename TKernel>
struct ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Static>
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const&,
TLoopBody&& loopBody,
TIdx const numIterations,
omp::Schedule const& schedule)
{
#    if _OPENMP < 200805 
std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
std::intmax_t i;
#        pragma omp for nowait schedule(static, schedule.chunkSize)
for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(static, schedule.chunkSize)
for(TIdx i = 0; i < numIterations; ++i)
#    endif
{
auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
wrappedLoopBody(i);
}
}
};

template<typename TKernel, typename TSchedule, typename TSfinae = void>
struct ParallelForStaticImpl
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const&,
TLoopBody&& loopBody,
TIdx const numIterations,
TSchedule const&)
{
#    if _OPENMP < 200805 
std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
std::intmax_t i;
#        pragma omp for nowait schedule(static)
for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(static)
for(TIdx i = 0; i < numIterations; ++i)
#    endif
{
auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
wrappedLoopBody(i);
}
}
};

template<typename TKernel>
using HasScheduleChunkSize = std::void_t<decltype(TKernel::ompScheduleChunkSize)>;

template<typename TKernel, typename TSchedule>
struct ParallelForStaticImpl<TKernel, TSchedule, HasScheduleChunkSize<TKernel>>
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const& kernel,
TLoopBody&& loopBody,
TIdx const numIterations,
TSchedule const&)
{
#    if _OPENMP < 200805 
std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
std::intmax_t i;
#        pragma omp for nowait schedule(static, kernel.ompScheduleChunkSize)
for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(static, kernel.ompScheduleChunkSize)
for(TIdx i = 0; i < numIterations; ++i)
#    endif
{
auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
wrappedLoopBody(i);
}
}
};

template<typename TKernel, typename TSchedule>
struct ParallelForImpl<TKernel, TSchedule, omp::Schedule::Static> : ParallelForStaticImpl<TKernel, TSchedule>
{
};

template<typename TKernel>
struct ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Dynamic>
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const&,
TLoopBody&& loopBody,
TIdx const numIterations,
omp::Schedule const& schedule)
{
#    if _OPENMP < 200805 
std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
std::intmax_t i;
#        pragma omp for nowait schedule(dynamic, schedule.chunkSize)
for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(dynamic, schedule.chunkSize)
for(TIdx i = 0; i < numIterations; ++i)
#    endif
{
auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
wrappedLoopBody(i);
}
}
};

template<typename TKernel, typename TSchedule, typename TSfinae = void>
struct ParallelForDynamicImpl
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const&,
TLoopBody&& loopBody,
TIdx const numIterations,
TSchedule const&)
{
#    if _OPENMP < 200805 
std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
std::intmax_t i;
#        pragma omp for nowait schedule(dynamic)
for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(dynamic)
for(TIdx i = 0; i < numIterations; ++i)
#    endif
{
auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
wrappedLoopBody(i);
}
}
};

template<typename TKernel, typename TSchedule>
struct ParallelForDynamicImpl<TKernel, TSchedule, HasScheduleChunkSize<TKernel>>
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const& kernel,
TLoopBody&& loopBody,
TIdx const numIterations,
TSchedule const&)
{
#    if _OPENMP < 200805 
std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
std::intmax_t i;
#        pragma omp for nowait schedule(dynamic, kernel.ompScheduleChunkSize)
for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(dynamic, kernel.ompScheduleChunkSize)
for(TIdx i = 0; i < numIterations; ++i)
#    endif
{
auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
wrappedLoopBody(i);
}
}
};

template<typename TKernel, typename TSchedule>
struct ParallelForImpl<TKernel, TSchedule, omp::Schedule::Dynamic> : ParallelForDynamicImpl<TKernel, TSchedule>
{
};

template<typename TKernel>
struct ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Guided>
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const&,
TLoopBody&& loopBody,
TIdx const numIterations,
omp::Schedule const& schedule)
{
#    if _OPENMP < 200805 
std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
std::intmax_t i;
#        pragma omp for nowait schedule(guided, schedule.chunkSize)
for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(guided, schedule.chunkSize)
for(TIdx i = 0; i < numIterations; ++i)
#    endif
{
auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
wrappedLoopBody(i);
}
}
};

template<typename TKernel, typename TSchedule, typename TSfinae = void>
struct ParallelForGuidedImpl
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const&,
TLoopBody&& loopBody,
TIdx const numIterations,
TSchedule const&)
{
#    if _OPENMP < 200805 
std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
std::intmax_t i;
#        pragma omp for nowait schedule(guided)
for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(guided)
for(TIdx i = 0; i < numIterations; ++i)
#    endif
{
auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
wrappedLoopBody(i);
}
}
};

template<typename TKernel, typename TSchedule>
struct ParallelForGuidedImpl<TKernel, TSchedule, HasScheduleChunkSize<TKernel>>
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const& kernel,
TLoopBody&& loopBody,
TIdx const numIterations,
TSchedule const&)
{
#    if _OPENMP < 200805 
std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
std::intmax_t i;
#        pragma omp for nowait schedule(guided, kernel.ompScheduleChunkSize)
for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(guided, kernel.ompScheduleChunkSize)
for(TIdx i = 0; i < numIterations; ++i)
#    endif
{
auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
wrappedLoopBody(i);
}
}
};

template<typename TKernel, typename TSchedule>
struct ParallelForImpl<TKernel, TSchedule, omp::Schedule::Guided> : ParallelForGuidedImpl<TKernel, TSchedule>
{
};

#    if _OPENMP >= 200805
template<typename TKernel, typename TSchedule>
struct ParallelForImpl<TKernel, TSchedule, omp::Schedule::Auto>
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const&,
TLoopBody&& loopBody,
TIdx const numIterations,
TSchedule const&)
{
#        pragma omp for nowait schedule(auto)
for(TIdx i = 0; i < numIterations; ++i)
{
auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
wrappedLoopBody(i);
}
}
};
#    endif

template<typename TKernel, typename TSchedule>
struct ParallelForImpl<TKernel, TSchedule, omp::Schedule::Runtime>
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const&,
TLoopBody&& loopBody,
TIdx const numIterations,
TSchedule const&)
{
#    if _OPENMP < 200805 
std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numIterations));
std::intmax_t i;
#        pragma omp for nowait schedule(runtime)
for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(runtime)
for(TIdx i = 0; i < numIterations; ++i)
#    endif
{
auto wrappedLoopBody = [&loopBody](auto idx) { loopBody(idx); };
wrappedLoopBody(i);
}
}
};

template<typename TKernel, typename TSchedule, typename TSfinae = void>
struct ParallelFor
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const& kernel,
TLoopBody&& loopBody,
TIdx const numIterations,
TSchedule const& schedule)
{
ParallelForImpl<TKernel, TSchedule, omp::Schedule::NoSchedule>{}(
kernel,
std::forward<TLoopBody>(loopBody),
numIterations,
schedule);
}
};

template<typename TKernel>
struct ParallelFor<TKernel, omp::Schedule>
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const& kernel,
TLoopBody&& loopBody,
TIdx const numIterations,
omp::Schedule const& schedule)
{
switch(schedule.kind)
{
case omp::Schedule::NoSchedule:
ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::NoSchedule>{}(
kernel,
std::forward<TLoopBody>(loopBody),
numIterations,
schedule);
break;
case omp::Schedule::Static:
ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Static>{}(
kernel,
std::forward<TLoopBody>(loopBody),
numIterations,
schedule);
break;
case omp::Schedule::Dynamic:
ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Dynamic>{}(
kernel,
std::forward<TLoopBody>(loopBody),
numIterations,
schedule);
break;
case omp::Schedule::Guided:
ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Guided>{}(
kernel,
std::forward<TLoopBody>(loopBody),
numIterations,
schedule);
break;
#    if _OPENMP >= 200805
case omp::Schedule::Auto:
ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Auto>{}(
kernel,
std::forward<TLoopBody>(loopBody),
numIterations,
schedule);
break;
#    endif
case omp::Schedule::Runtime:
ParallelForImpl<TKernel, omp::Schedule, omp::Schedule::Runtime>{}(
kernel,
std::forward<TLoopBody>(loopBody),
numIterations,
schedule);
break;
}
}
};

template<typename TSchedule>
using IsOmpScheduleTraitSpecialized
= std::integral_constant<bool, std::is_same<TSchedule, omp::Schedule>::value>;

template<typename TKernel, typename TSchedule>
using UseScheduleKind
= std::enable_if_t<sizeof(TKernel::ompScheduleKind) && !IsOmpScheduleTraitSpecialized<TSchedule>::value>;

template<typename TKernel, typename TSchedule>
struct ParallelFor<TKernel, TSchedule, UseScheduleKind<TKernel, TSchedule>>
{
template<typename TLoopBody, typename TIdx>
ALPAKA_FN_HOST void operator()(
TKernel const& kernel,
TLoopBody&& loopBody,
TIdx const numIterations,
TSchedule const& schedule)
{
ParallelForImpl<TKernel, TSchedule, TKernel::ompScheduleKind>{}(
kernel,
std::forward<TLoopBody>(loopBody),
numIterations,
schedule);
}
};

template<typename TKernel, typename TLoopBody, typename TIdx, typename TSchedule>
ALPAKA_FN_HOST ALPAKA_FN_INLINE void parallelFor(
TKernel const& kernel,
TLoopBody&& loopBody,
TIdx const numIterations,
TSchedule const& schedule)
{
ParallelFor<TKernel, TSchedule>{}(kernel, std::forward<TLoopBody>(loopBody), numIterations, schedule);
}

} 

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
class TaskKernelCpuOmp2Blocks final : public WorkDivMembers<TDim, TIdx>
{
public:
template<typename TWorkDiv>
ALPAKA_FN_HOST TaskKernelCpuOmp2Blocks(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
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
return getBlockSharedMemDynSizeBytes<AccCpuOmp2Blocks<TDim, TIdx>>(
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

TIdx const numBlocksInGrid(gridBlockExtent.prod());
if(blockThreadExtent.prod() != static_cast<TIdx>(1u))
{
throw std::runtime_error("Only one thread per block allowed in the OpenMP 2.0 block accelerator!");
}

auto const schedule = std::apply(
[&](ALPAKA_DECAY_T(TArgs) const&... args) {
return getOmpSchedule<AccCpuOmp2Blocks<TDim, TIdx>>(
m_kernelFnObj,
blockThreadExtent,
threadElemExtent,
args...);
},
m_args);

if(::omp_in_parallel() != 0)
{
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
std::cout << __func__ << " already within a parallel region." << std::endl;
#    endif
parallelFn(blockSharedMemDynSizeBytes, numBlocksInGrid, gridBlockExtent, schedule);
}
else
{
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
std::cout << __func__ << " opening new parallel region." << std::endl;
#    endif
#    pragma omp parallel
parallelFn(blockSharedMemDynSizeBytes, numBlocksInGrid, gridBlockExtent, schedule);
}
}

private:
template<typename TSchedule>
ALPAKA_FN_HOST auto parallelFn(
std::size_t const& blockSharedMemDynSizeBytes,
TIdx const& numBlocksInGrid,
Vec<TDim, TIdx> const& gridBlockExtent,
TSchedule const& schedule) const -> void
{
#    pragma omp single nowait
{
if((numBlocksInGrid > 1) && (::omp_get_max_threads() > 1) && (::omp_in_parallel() == 0))
{
throw std::runtime_error("The OpenMP 2.0 runtime did not create a parallel region!");
}

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
std::cout << __func__ << " omp_get_num_threads: " << ::omp_get_num_threads() << std::endl;
#    endif
}

AccCpuOmp2Blocks<TDim, TIdx> acc(
*static_cast<WorkDivMembers<TDim, TIdx> const*>(this),
blockSharedMemDynSizeBytes);

auto loopBody = [&](auto currentIndex)
{
#    if _OPENMP < 200805
auto const i_tidx = static_cast<TIdx>(currentIndex); 
auto const index = Vec<DimInt<1u>, TIdx>(i_tidx); 
#    else
auto const index = Vec<DimInt<1u>, TIdx>(currentIndex); 
#    endif
acc.m_gridBlockIdx = mapIdx<TDim::value>(index, gridBlockExtent);

std::apply(m_kernelFnObj, std::tuple_cat(std::tie(acc), m_args));

freeSharedVars(acc);
};

detail::parallelFor(m_kernelFnObj, loopBody, numBlocksInGrid, schedule);
}

TKernelFnObj m_kernelFnObj;
std::tuple<std::decay_t<TArgs>...> m_args;
};

namespace trait
{
template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct AccType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = AccCpuOmp2Blocks<TDim, TIdx>;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct DevType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = DevCpu;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct DimType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = TDim;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct PltfType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = PltfCpu;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct IdxType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = TIdx;
};
} 
} 

#endif
