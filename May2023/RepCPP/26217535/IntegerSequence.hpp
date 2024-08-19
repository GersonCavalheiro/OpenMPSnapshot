

#pragma once

#include <alpaka/meta/Set.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace alpaka::meta
{
namespace detail
{
template<typename TDstType, typename TIntegerSequence>
struct ConvertIntegerSequence;
template<typename TDstType, typename T, T... Tvals>
struct ConvertIntegerSequence<TDstType, std::integer_sequence<T, Tvals...>>
{
using type = std::integer_sequence<TDstType, static_cast<TDstType>(Tvals)...>;
};
} 
template<typename TDstType, typename TIntegerSequence>
using ConvertIntegerSequence = typename detail::ConvertIntegerSequence<TDstType, TIntegerSequence>::type;

namespace detail
{
template<bool TisSizeNegative, bool TbIsBegin, typename T, T Tbegin, typename TIntCon, typename TIntSeq>
struct MakeIntegerSequenceHelper
{
static_assert(!TisSizeNegative, "MakeIntegerSequence<T, N> requires N to be non-negative.");
};
template<typename T, T Tbegin, T... Tvals>
struct MakeIntegerSequenceHelper<
false,
true,
T,
Tbegin,
std::integral_constant<T, Tbegin>,
std::integer_sequence<T, Tvals...>>
{
using type = std::integer_sequence<T, Tvals...>;
};
template<typename T, T Tbegin, T TIdx, T... Tvals>
struct MakeIntegerSequenceHelper<
false,
false,
T,
Tbegin,
std::integral_constant<T, TIdx>,
std::integer_sequence<T, Tvals...>>
{
using type = typename MakeIntegerSequenceHelper<
false,
TIdx == (Tbegin + 1),
T,
Tbegin,
std::integral_constant<T, TIdx - 1>,
std::integer_sequence<T, TIdx - 1, Tvals...>>::type;
};
} 

template<typename T, T Tbegin, T Tsize>
using MakeIntegerSequenceOffset = typename detail::MakeIntegerSequenceHelper<
(Tsize < 0),
(Tsize == 0),
T,
Tbegin,
std::integral_constant<T, Tbegin + Tsize>,
std::integer_sequence<T>>::type;

template<typename T, T... Tvals>
struct IntegralValuesUnique
{
static constexpr bool value = meta::IsParameterPackSet<std::integral_constant<T, Tvals>...>::value;
};

template<typename TIntegerSequence>
struct IntegerSequenceValuesUnique;
template<typename T, T... Tvals>
struct IntegerSequenceValuesUnique<std::integer_sequence<T, Tvals...>>
{
static constexpr bool value = IntegralValuesUnique<T, Tvals...>::value;
};

template<typename T, T Tmin, T Tmax, T... Tvals>
struct IntegralValuesInRange;
template<typename T, T Tmin, T Tmax>
struct IntegralValuesInRange<T, Tmin, Tmax>
{
static constexpr bool value = true;
};
template<typename T, T Tmin, T Tmax, T I, T... Tvals>
struct IntegralValuesInRange<T, Tmin, Tmax, I, Tvals...>
{
static constexpr bool value
= (I >= Tmin) && (I <= Tmax) && IntegralValuesInRange<T, Tmin, Tmax, Tvals...>::value;
};

template<typename TIntegerSequence, typename T, T Tmin, T Tmax>
struct IntegerSequenceValuesInRange;
template<typename T, T... Tvals, T Tmin, T Tmax>
struct IntegerSequenceValuesInRange<std::integer_sequence<T, Tvals...>, T, Tmin, Tmax>
{
static constexpr bool value = IntegralValuesInRange<T, Tmin, Tmax, Tvals...>::value;
};
} 
