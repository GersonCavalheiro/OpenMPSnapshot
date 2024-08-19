

#pragma once

#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

namespace alpaka
{
namespace bt
{
template<typename TDim, typename TIdx>
class IdxBtZero : public concepts::Implements<ConceptIdxBt, IdxBtZero<TDim, TIdx>>
{
};
} 

namespace trait
{
template<typename TDim, typename TIdx>
struct DimType<bt::IdxBtZero<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx>
struct GetIdx<bt::IdxBtZero<TDim, TIdx>, origin::Block, unit::Threads>
{
template<typename TWorkDiv>
ALPAKA_FN_HOST static auto getIdx(
bt::IdxBtZero<TDim, TIdx> const& ,
TWorkDiv const& ) -> Vec<TDim, TIdx>
{
return Vec<TDim, TIdx>::zeros();
}
};

template<typename TDim, typename TIdx>
struct IdxType<bt::IdxBtZero<TDim, TIdx>>
{
using type = TIdx;
};
} 
} 
