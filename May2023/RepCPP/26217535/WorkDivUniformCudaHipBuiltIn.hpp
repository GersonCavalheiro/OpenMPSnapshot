

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>
#    include <alpaka/workdiv/Traits.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

namespace alpaka
{
template<typename TDim, typename TIdx>
class WorkDivUniformCudaHipBuiltIn
: public concepts::Implements<ConceptWorkDiv, WorkDivUniformCudaHipBuiltIn<TDim, TIdx>>
{
public:
ALPAKA_FN_HOST_ACC WorkDivUniformCudaHipBuiltIn(Vec<TDim, TIdx> const& threadElemExtent)
: m_threadElemExtent(threadElemExtent)
{
}

Vec<TDim, TIdx> const& m_threadElemExtent;
};

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

namespace trait
{
template<typename TDim, typename TIdx>
struct DimType<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx>
struct IdxType<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>>
{
using type = TIdx;
};

template<typename TDim, typename TIdx>
struct GetWorkDiv<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>, origin::Grid, unit::Blocks>
{
__device__ static auto getWorkDiv(WorkDivUniformCudaHipBuiltIn<TDim, TIdx> const& )
-> Vec<TDim, TIdx>
{
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
return castVec<TIdx>(getExtentVecEnd<TDim>(gridDim));
#        else
return getExtentVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
static_cast<TIdx>(hipGridDim_z),
static_cast<TIdx>(hipGridDim_y),
static_cast<TIdx>(hipGridDim_x)));
#        endif
}
};

template<typename TDim, typename TIdx>
struct GetWorkDiv<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>, origin::Block, unit::Threads>
{
__device__ static auto getWorkDiv(WorkDivUniformCudaHipBuiltIn<TDim, TIdx> const& )
-> Vec<TDim, TIdx>
{
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
return castVec<TIdx>(getExtentVecEnd<TDim>(blockDim));
#        else
return getExtentVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
static_cast<TIdx>(hipBlockDim_z),
static_cast<TIdx>(hipBlockDim_y),
static_cast<TIdx>(hipBlockDim_x)));
#        endif
}
};

template<typename TDim, typename TIdx>
struct GetWorkDiv<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>, origin::Thread, unit::Elems>
{
__device__ static auto getWorkDiv(WorkDivUniformCudaHipBuiltIn<TDim, TIdx> const& workDiv)
-> Vec<TDim, TIdx>
{
return workDiv.m_threadElemExtent;
}
};
} 

#    endif

} 

#endif
