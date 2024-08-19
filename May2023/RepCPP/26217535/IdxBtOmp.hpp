

#pragma once

#ifdef _OPENMP

#    include <alpaka/core/Assert.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/Positioning.hpp>
#    include <alpaka/idx/MapIdx.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>
#    include <alpaka/workdiv/Traits.hpp>

#    include <omp.h>

namespace alpaka
{
namespace bt
{
template<typename TDim, typename TIdx>
class IdxBtOmp : public concepts::Implements<ConceptIdxBt, IdxBtOmp<TDim, TIdx>>
{
};
} 

namespace trait
{
template<typename TDim, typename TIdx>
struct DimType<bt::IdxBtOmp<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx>
struct GetIdx<bt::IdxBtOmp<TDim, TIdx>, origin::Block, unit::Threads>
{
template<typename TWorkDiv>
static auto getIdx(bt::IdxBtOmp<TDim, TIdx> const& , TWorkDiv const& workDiv) -> Vec<TDim, TIdx>
{
ALPAKA_ASSERT_OFFLOAD(::omp_get_thread_num() >= 0);
return mapIdx<TDim::value>(
Vec<DimInt<1u>, TIdx>(static_cast<TIdx>(::omp_get_thread_num())),
getWorkDiv<Block, Threads>(workDiv));
}
};

template<typename TIdx>
struct GetIdx<bt::IdxBtOmp<DimInt<1u>, TIdx>, origin::Block, unit::Threads>
{
template<typename TWorkDiv>
static auto getIdx(bt::IdxBtOmp<DimInt<1u>, TIdx> const& , TWorkDiv const&)
-> Vec<DimInt<1u>, TIdx>
{
return Vec<DimInt<1u>, TIdx>(static_cast<TIdx>(::omp_get_thread_num()));
}
};

template<typename TDim, typename TIdx>
struct IdxType<bt::IdxBtOmp<TDim, TIdx>>
{
using type = TIdx;
};
} 
} 

#endif
