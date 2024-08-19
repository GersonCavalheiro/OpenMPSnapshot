

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include <utility>

namespace alpaka::meta
{
namespace detail
{
template<typename TIndexSequence>
struct NdLoop;

template<>
struct NdLoop<std::index_sequence<>>
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename TIndex, typename TExtentVec, typename TFnObj>
ALPAKA_FN_HOST_ACC static auto ndLoop(TIndex const& idx, TExtentVec const& , TFnObj const& f)
-> void
{
f(idx);
}
};

template<std::size_t Tdim0, std::size_t... Tdims>
struct NdLoop<std::index_sequence<Tdim0, Tdims...>>
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename TIndex, typename TExtentVec, typename TFnObj>
ALPAKA_FN_HOST_ACC static auto ndLoop(TIndex& idx, TExtentVec const& extent, TFnObj const& f) -> void
{
static_assert(Dim<TIndex>::value > 0u, "The dimension given to ndLoop has to be larger than zero!");
static_assert(
Dim<TIndex>::value == Dim<TExtentVec>::value,
"The dimensions of the iteration vector and the extent vector have to be identical!");
static_assert(Dim<TIndex>::value > Tdim0, "The current dimension has to be in the range [0,dim-1]!");

for(idx[Tdim0] = 0u; idx[Tdim0] < extent[Tdim0]; ++idx[Tdim0])
{
NdLoop<std::index_sequence<Tdims...>>::template ndLoop(idx, extent, f);
}
}
};
} 

ALPAKA_NO_HOST_ACC_WARNING
template<typename TExtentVec, typename TFnObj, std::size_t... Tdims>
ALPAKA_FN_HOST_ACC auto ndLoop(
[[maybe_unused]] std::index_sequence<Tdims...> const& indexSequence,
TExtentVec const& extent,
TFnObj const& f) -> void
{
static_assert(
IntegerSequenceValuesInRange<std::index_sequence<Tdims...>, std::size_t, 0, Dim<TExtentVec>::value>::value,
"The values in the index_sequence have to be in the range [0,dim-1]!");
static_assert(
IntegerSequenceValuesUnique<std::index_sequence<Tdims...>>::value,
"The values in the index_sequence have to be unique!");

auto idx = Vec<Dim<TExtentVec>, Idx<TExtentVec>>::zeros();

detail::NdLoop<std::index_sequence<Tdims...>>::template ndLoop(idx, extent, f);
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TExtentVec, typename TFnObj>
ALPAKA_FN_HOST_ACC auto ndLoopIncIdx(TExtentVec const& extent, TFnObj const& f) -> void
{
ndLoop(std::make_index_sequence<Dim<TExtentVec>::value>(), extent, f);
}
} 
