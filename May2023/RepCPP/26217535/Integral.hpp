

#pragma once

#include <type_traits>

namespace alpaka::meta
{
template<typename TSuperset, typename TSubset>
using IsIntegralSuperset = std::integral_constant<
bool,
std::is_integral_v<TSuperset> && std::is_integral_v<TSubset>
&& (
((std::is_unsigned_v<TSuperset> == std::is_unsigned_v<TSubset>)
&& (sizeof(TSuperset) >= sizeof(TSubset)))
|| ((std::is_unsigned_v<TSuperset> != std::is_unsigned_v<TSubset>)
&& (sizeof(TSuperset) > sizeof(TSubset))))>;

template<typename T0, typename T1>
using HigherMax = std::conditional_t<
(sizeof(T0) > sizeof(T1)),
T0,
std::conditional_t<((sizeof(T0) == sizeof(T1)) && std::is_unsigned_v<T0> && std::is_signed_v<T1>), T0, T1>>;

template<typename T0, typename T1>
using LowerMax = std::conditional_t<
(sizeof(T0) < sizeof(T1)),
T0,
std::conditional_t<((sizeof(T0) == sizeof(T1)) && std::is_signed_v<T0> && std::is_unsigned_v<T1>), T0, T1>>;

template<typename T0, typename T1>
using HigherMin = std::conditional_t<
(std::is_unsigned_v<T0> == std::is_unsigned_v<T1>),
std::conditional_t<
std::is_unsigned_v<T0>,
std::conditional_t<(sizeof(T0) < sizeof(T1)), T1, T0>,
std::conditional_t<(sizeof(T0) < sizeof(T1)), T0, T1>>,
std::conditional_t<std::is_unsigned_v<T0>, T0, T1>>;

template<typename T0, typename T1>
using LowerMin = std::conditional_t<
(std::is_unsigned_v<T0> == std::is_unsigned_v<T1>),
std::conditional_t<(sizeof(T0) > sizeof(T1)), T0, T1>,
std::conditional_t<std::is_signed_v<T0>, T0, T1>>;
} 
