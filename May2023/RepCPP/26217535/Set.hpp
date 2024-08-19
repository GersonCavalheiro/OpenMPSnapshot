

#pragma once

#include <utility>

namespace alpaka::meta
{
namespace detail
{
template<typename T>
struct Empty
{
};

template<typename... Ts>
struct IsParameterPackSetImpl;
template<>
struct IsParameterPackSetImpl<>
{
static constexpr bool value = true;
};
template<typename T, typename... Ts>
struct IsParameterPackSetImpl<T, Ts...>
: public IsParameterPackSetImpl<Ts...>
, public virtual Empty<T>
{
using Base = IsParameterPackSetImpl<Ts...>;

static constexpr bool value = Base::value && !std::is_base_of_v<Empty<T>, Base>;
};
} 
template<typename... Ts>
using IsParameterPackSet = detail::IsParameterPackSetImpl<Ts...>;

namespace detail
{
template<typename TList>
struct IsSetImpl;
template<template<typename...> class TList, typename... Ts>
struct IsSetImpl<TList<Ts...>>
{
static constexpr bool value = IsParameterPackSet<Ts...>::value;
};
} 
template<typename TList>
using IsSet = detail::IsSetImpl<TList>;
} 
