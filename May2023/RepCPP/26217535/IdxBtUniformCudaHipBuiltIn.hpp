

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/Positioning.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

namespace alpaka
{
namespace bt
{
template<typename TDim, typename TIdx>
class IdxBtUniformCudaHipBuiltIn
: public concepts::Implements<ConceptIdxBt, IdxBtUniformCudaHipBuiltIn<TDim, TIdx>>
{
};
} 

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
struct DimType<bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx>
struct GetIdx<bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx>, origin::Block, unit::Threads>
{
template<typename TWorkDiv>
__device__ static auto getIdx(bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx> const& , TWorkDiv const&)
-> Vec<TDim, TIdx>
{
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
return castVec<TIdx>(getOffsetVecEnd<TDim>(threadIdx));
#        else
return getOffsetVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
static_cast<TIdx>(hipThreadIdx_z),
static_cast<TIdx>(hipThreadIdx_y),
static_cast<TIdx>(hipThreadIdx_x)));
#        endif
}
};

template<typename TDim, typename TIdx>
struct IdxType<bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx>>
{
using type = TIdx;
};
} 

#    endif

} 

#endif
