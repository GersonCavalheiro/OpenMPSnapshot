

#pragma once

#include <alpaka/meta/Concatenate.hpp>

namespace alpaka::meta
{
namespace detail
{
template<typename... Ts>
struct CartesianProductImplHelper;
template<template<typename...> class TList, typename... Ts>
struct CartesianProductImplHelper<TList<Ts...>>
{
using type = TList<Ts...>;
};
template<template<typename...> class TList, typename... Ts>
struct CartesianProductImplHelper<TList<TList<>>, Ts...>
{
using type = TList<>;
};
template<template<typename...> class TList, typename... Ts, typename... Rests>
struct CartesianProductImplHelper<TList<Ts...>, TList<>, Rests...>
{
using type = TList<>;
};
template<template<typename...> class TList, typename... X, typename H, typename... Rests>
struct CartesianProductImplHelper<TList<X...>, TList<H>, Rests...>
{
using type1 = TList<Concatenate<X, TList<H>>...>;
using type = typename CartesianProductImplHelper<type1, Rests...>::type;
};
template<
template<typename...>
class TList,
typename... X,
template<typename...>
class Head,
typename T,
typename... Ts,
typename... Rests>
struct CartesianProductImplHelper<TList<X...>, Head<T, Ts...>, Rests...>
{
using type1 = TList<Concatenate<X, TList<T>>...>;
using type2 = typename CartesianProductImplHelper<TList<X...>, TList<Ts...>>::type;
using type3 = Concatenate<type1, type2>;
using type = typename CartesianProductImplHelper<type3, Rests...>::type;
};

template<template<typename...> class TList, typename... Ts>
struct CartesianProductImpl;
template<template<typename...> class TList>
struct CartesianProductImpl<TList>
{
using type = TList<>;
};
template<template<typename...> class TList, template<typename...> class Head, typename... Ts, typename... Tail>
struct CartesianProductImpl<TList, Head<Ts...>, Tail...>
{
using type = typename detail::CartesianProductImplHelper<TList<TList<Ts>...>, Tail...>::type;
};
} 

template<template<typename...> class TList, typename... Ts>
using CartesianProduct = typename detail::CartesianProductImpl<TList, Ts...>::type;
} 
