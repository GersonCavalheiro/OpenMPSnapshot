

#pragma once

#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

namespace alpaka
{
namespace gb
{
template<typename TDim, typename TIdx>
class IdxGbRef : public concepts::Implements<ConceptIdxGb, IdxGbRef<TDim, TIdx>>
{
public:
IdxGbRef(Vec<TDim, TIdx> const& gridBlockIdx) : m_gridBlockIdx(gridBlockIdx)
{
}

Vec<TDim, TIdx> const& m_gridBlockIdx;
};
} 

namespace trait
{
template<typename TDim, typename TIdx>
struct DimType<gb::IdxGbRef<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx>
struct GetIdx<gb::IdxGbRef<TDim, TIdx>, origin::Grid, unit::Blocks>
{
template<typename TWorkDiv>
ALPAKA_FN_HOST static auto getIdx(gb::IdxGbRef<TDim, TIdx> const& idx, TWorkDiv const& )
-> Vec<TDim, TIdx>
{
return idx.m_gridBlockIdx;
}
};

template<typename TDim, typename TIdx>
struct IdxType<gb::IdxGbRef<TDim, TIdx>>
{
using type = TIdx;
};
} 
} 
