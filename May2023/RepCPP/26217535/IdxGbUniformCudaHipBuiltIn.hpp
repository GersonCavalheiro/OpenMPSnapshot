

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
namespace gb
{
template<typename TDim, typename TIdx>
class IdxGbUniformCudaHipBuiltIn
: public concepts::Implements<ConceptIdxGb, IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
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
struct DimType<gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx>
struct GetIdx<gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>, origin::Grid, unit::Blocks>
{
template<typename TWorkDiv>
__device__ static auto getIdx(gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx> const& , TWorkDiv const&)
-> Vec<TDim, TIdx>
{
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
return castVec<TIdx>(getOffsetVecEnd<TDim>(blockIdx));
#        else
return getOffsetVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
static_cast<TIdx>(hipBlockIdx_z),
static_cast<TIdx>(hipBlockIdx_y),
static_cast<TIdx>(hipBlockIdx_x)));
#        endif
}
};

template<typename TDim, typename TIdx>
struct IdxType<gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
{
using type = TIdx;
};
} 

#    endif

} 

#endif
