

#pragma once

#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/Traits.hpp>

namespace alpaka
{
namespace bt
{
template<typename TDim, typename TIdx>
class IdxBtLinear : public concepts::Implements<ConceptIdxBt, IdxBtLinear<TDim, TIdx>>
{
public:
IdxBtLinear(TIdx blockThreadIdx) : m_blockThreadIdx(blockThreadIdx)
{
}

const TIdx m_blockThreadIdx;
};
} 

namespace trait
{
template<typename TDim, typename TIdx>
struct DimType<bt::IdxBtLinear<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx>
struct GetIdx<bt::IdxBtLinear<TDim, TIdx>, origin::Block, unit::Threads>
{
template<typename TWorkDiv>
static auto getIdx(bt::IdxBtLinear<TDim, TIdx> const& idx, TWorkDiv const& workDiv) -> Vec<TDim, TIdx>
{
return mapIdx<TDim::value>(
Vec<DimInt<1u>, TIdx>(idx.m_blockThreadIdx),
getWorkDiv<Block, Threads>(workDiv));
}
};

template<typename TIdx>
struct GetIdx<bt::IdxBtLinear<DimInt<1u>, TIdx>, origin::Block, unit::Threads>
{
template<typename TWorkDiv>
static auto getIdx(bt::IdxBtLinear<DimInt<1u>, TIdx> const& idx, TWorkDiv const&) -> Vec<DimInt<1u>, TIdx>
{
return idx.m_blockThreadIdx;
}
};

template<typename TDim, typename TIdx>
struct IdxType<bt::IdxBtLinear<TDim, TIdx>>
{
using type = TIdx;
};
} 
} 
