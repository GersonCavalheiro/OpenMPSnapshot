

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

#    include <alpaka/acc/AccCpuOmp2Threads.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/meta/NdLoop.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <omp.h>

#    include <functional>
#    include <stdexcept>
#    include <tuple>
#    include <type_traits>
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#        include <iostream>
#    endif

namespace alpaka
{
template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
class TaskKernelCpuOmp2Threads final : public WorkDivMembers<TDim, TIdx>
{
public:
template<typename TWorkDiv>
ALPAKA_FN_HOST TaskKernelCpuOmp2Threads(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
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
return getBlockSharedMemDynSizeBytes<AccCpuOmp2Threads<TDim, TIdx>>(
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

AccCpuOmp2Threads<TDim, TIdx> acc(
*static_cast<WorkDivMembers<TDim, TIdx> const*>(this),
blockSharedMemDynSizeBytes);

TIdx const blockThreadCount(blockThreadExtent.prod());
[[maybe_unused]] int const iBlockThreadCount(static_cast<int>(blockThreadCount));

if(::omp_in_parallel() != 0)
{
throw std::runtime_error(
"The OpenMP 2.0 thread backend can not be used within an existing parallel region!");
}

int const ompIsDynamic(::omp_get_dynamic());
::omp_set_dynamic(0);

meta::ndLoopIncIdx(
gridBlockExtent,
[&](Vec<TDim, TIdx> const& gridBlockIdx)
{
acc.m_gridBlockIdx = gridBlockIdx;


#    pragma omp parallel num_threads(iBlockThreadCount)
{
if constexpr((!BOOST_COMP_GNUC) || (BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(8, 1, 0)))
{
#    pragma omp single nowait
{
if((iBlockThreadCount > 1) && (::omp_in_parallel() == 0))
{
throw std::runtime_error(
"The OpenMP 2.0 runtime did not create a parallel region!");
}

int const numThreads = ::omp_get_num_threads();
if(numThreads != iBlockThreadCount)
{
throw std::runtime_error(
"The OpenMP 2.0 runtime did not use the number of threads "
"that had been required!");
}
}
}

std::apply(m_kernelFnObj, std::tuple_cat(std::tie(acc), m_args));

}

freeSharedVars(acc);
});

::omp_set_dynamic(ompIsDynamic);
}

private:
TKernelFnObj m_kernelFnObj;
std::tuple<std::decay_t<TArgs>...> m_args;
};

namespace trait
{
template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct AccType<TaskKernelCpuOmp2Threads<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = AccCpuOmp2Threads<TDim, TIdx>;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct DevType<TaskKernelCpuOmp2Threads<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = DevCpu;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct DimType<TaskKernelCpuOmp2Threads<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = TDim;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct PltfType<TaskKernelCpuOmp2Threads<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = PltfCpu;
};

template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
struct IdxType<TaskKernelCpuOmp2Threads<TDim, TIdx, TKernelFnObj, TArgs...>>
{
using type = TIdx;
};
} 
} 

#endif
