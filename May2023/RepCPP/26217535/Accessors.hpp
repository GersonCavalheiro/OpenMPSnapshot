

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/Traits.hpp>

#include <utility>

namespace alpaka
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename TOrigin, typename TUnit, typename TIdx, typename TWorkDiv>
ALPAKA_FN_HOST_ACC auto getIdx(TIdx const& idx, TWorkDiv const& workDiv) -> Vec<Dim<TWorkDiv>, Idx<TIdx>>
{
return trait::GetIdx<TIdx, TOrigin, TUnit>::getIdx(idx, workDiv);
}
ALPAKA_NO_HOST_ACC_WARNING
template<typename TOrigin, typename TUnit, typename TIdxWorkDiv>
ALPAKA_FN_HOST_ACC auto getIdx(TIdxWorkDiv const& idxWorkDiv) -> Vec<Dim<TIdxWorkDiv>, Idx<TIdxWorkDiv>>
{
return trait::GetIdx<TIdxWorkDiv, TOrigin, TUnit>::getIdx(idxWorkDiv, idxWorkDiv);
}

namespace trait
{
template<typename TIdxGb>
struct GetIdx<TIdxGb, origin::Grid, unit::Blocks>
{
using ImplementationBase = concepts::ImplementationBase<ConceptIdxGb, TIdxGb>;
ALPAKA_NO_HOST_ACC_WARNING
template<typename TWorkDiv>
ALPAKA_FN_HOST_ACC static auto getIdx(TIdxGb const& idx, TWorkDiv const& workDiv)
-> Vec<Dim<ImplementationBase>, Idx<ImplementationBase>>
{
return trait::GetIdx<ImplementationBase, origin::Grid, unit::Blocks>::getIdx(idx, workDiv);
}
};

template<typename TIdxBt>
struct GetIdx<TIdxBt, origin::Block, unit::Threads>
{
using ImplementationBase = concepts::ImplementationBase<ConceptIdxBt, TIdxBt>;
ALPAKA_NO_HOST_ACC_WARNING
template<typename TWorkDiv>
ALPAKA_FN_HOST_ACC static auto getIdx(TIdxBt const& idx, TWorkDiv const& workDiv)
-> Vec<Dim<ImplementationBase>, Idx<ImplementationBase>>
{
return trait::GetIdx<ImplementationBase, origin::Block, unit::Threads>::getIdx(idx, workDiv);
}
};

template<typename TIdx>
struct GetIdx<TIdx, origin::Grid, unit::Threads>
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename TWorkDiv>
ALPAKA_FN_HOST_ACC static auto getIdx(TIdx const& idx, TWorkDiv const& workDiv)
{
return alpaka::getIdx<origin::Grid, unit::Blocks>(idx, workDiv)
* getWorkDiv<origin::Block, unit::Threads>(workDiv)
+ alpaka::getIdx<origin::Block, unit::Threads>(idx, workDiv);
}
};
} 
ALPAKA_NO_HOST_ACC_WARNING
template<typename TIdxWorkDiv, typename TGridThreadIdx, typename TThreadElemExtent>
ALPAKA_FN_HOST_ACC auto getIdxThreadFirstElem(
[[maybe_unused]] TIdxWorkDiv const& idxWorkDiv,
TGridThreadIdx const& gridThreadIdx,
TThreadElemExtent const& threadElemExtent) -> Vec<Dim<TIdxWorkDiv>, Idx<TIdxWorkDiv>>
{
return gridThreadIdx * threadElemExtent;
}
ALPAKA_NO_HOST_ACC_WARNING
template<typename TIdxWorkDiv, typename TGridThreadIdx>
ALPAKA_FN_HOST_ACC auto getIdxThreadFirstElem(TIdxWorkDiv const& idxWorkDiv, TGridThreadIdx const& gridThreadIdx)
-> Vec<Dim<TIdxWorkDiv>, Idx<TIdxWorkDiv>>
{
auto const threadElemExtent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(idxWorkDiv));
return getIdxThreadFirstElem(idxWorkDiv, gridThreadIdx, threadElemExtent);
}
ALPAKA_NO_HOST_ACC_WARNING
template<typename TIdxWorkDiv>
ALPAKA_FN_HOST_ACC auto getIdxThreadFirstElem(TIdxWorkDiv const& idxWorkDiv)
-> Vec<Dim<TIdxWorkDiv>, Idx<TIdxWorkDiv>>
{
auto const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(idxWorkDiv));
return getIdxThreadFirstElem(idxWorkDiv, gridThreadIdx);
}
} 
