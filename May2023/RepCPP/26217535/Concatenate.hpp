

#pragma once

namespace alpaka::meta
{
namespace detail
{
template<typename... T>
struct ConcatenateImpl;
template<typename T>
struct ConcatenateImpl<T>
{
using type = T;
};
template<template<typename...> class TList, typename... As, typename... Bs, typename... TRest>
struct ConcatenateImpl<TList<As...>, TList<Bs...>, TRest...>
{
using type = typename ConcatenateImpl<TList<As..., Bs...>, TRest...>::type;
};
} 
template<typename... T>
using Concatenate = typename detail::ConcatenateImpl<T...>::type;
} 
