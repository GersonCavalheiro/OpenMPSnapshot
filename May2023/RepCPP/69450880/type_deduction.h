
#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <hydra/detail/external/hydra_thrust/detail/preprocessor.h>

#include <utility>
#include <type_traits>


#define HYDRA_THRUST_FWD(x) ::std::forward<decltype(x)>(x)

#define HYDRA_THRUST_MVCAP(x) x = ::std::move(x)

#define HYDRA_THRUST_RETOF(...)   HYDRA_THRUST_PP_DISPATCH(HYDRA_THRUST_RETOF, __VA_ARGS__)
#define HYDRA_THRUST_RETOF1(C)    decltype(::std::declval<C>()())
#define HYDRA_THRUST_RETOF2(C, V) decltype(::std::declval<C>()(::std::declval<V>()))

#define HYDRA_THRUST_RETURNS(...)                                                   \
noexcept(noexcept(__VA_ARGS__))                                             \
{ return (__VA_ARGS__); }                                                   \


#define HYDRA_THRUST_DECLTYPE_RETURNS(...)                                          \
noexcept(noexcept(__VA_ARGS__))                                             \
-> decltype(__VA_ARGS__)                                                    \
{ return (__VA_ARGS__); }                                                   \


#define HYDRA_THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION(condition, ...)         \
noexcept(noexcept(__VA_ARGS__))                                             \
-> typename std::enable_if<condition, decltype(__VA_ARGS__)>::type          \
{ return (__VA_ARGS__); }                                                   \



#endif 

