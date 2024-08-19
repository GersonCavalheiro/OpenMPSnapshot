

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

#define __HYDRA_THRUST_DEFINE_HAS_MEMBER_FUNCTION(trait_name, member_function_name)                                \
template<typename T, typename Signature> class trait_name;                                                   \
\
template<typename T, typename Result>                                                                        \
class trait_name<T, Result(void)>                                                                            \
{                                                                                                            \
class yes { char m; };                                                                                    \
class no { yes m[2]; };                                                                                   \
struct base_mixin                                                                                         \
{                                                                                                         \
Result member_function_name();                                                                          \
};                                                                                                        \
struct base : public T, public base_mixin {};                                                             \
template <typename U, U t>  class helper{};                                                               \
template <typename U>                                                                                     \
static no deduce(U*, helper<Result (base_mixin::*)(), &U::member_function_name>* = 0);                    \
static yes deduce(...);                                                                                   \
public:                                                                                                      \
static const bool value = sizeof(yes) == sizeof(deduce(static_cast<base*>(0)));                           \
typedef hydra_thrust::detail::integral_constant<bool,value> type;                                               \
};                                                                                                           \
\
template<typename T, typename Result, typename Arg>                                                          \
class trait_name<T, Result(Arg)>                                                                             \
{                                                                                                            \
class yes { char m; };                                                                                    \
class no { yes m[2]; };                                                                                   \
struct base_mixin                                                                                         \
{                                                                                                         \
Result member_function_name(Arg);                                                                       \
};                                                                                                        \
struct base : public T, public base_mixin {};                                                             \
template <typename U, U t>  class helper{};                                                               \
template <typename U>                                                                                     \
static no deduce(U*, helper<Result (base_mixin::*)(Arg), &U::member_function_name>* = 0);                 \
static yes deduce(...);                                                                                   \
public:                                                                                                      \
static const bool value = sizeof(yes) == sizeof(deduce(static_cast<base*>(0)));                           \
typedef hydra_thrust::detail::integral_constant<bool,value> type;                                               \
};                                                                                                           \
\
template<typename T, typename Result, typename Arg1, typename Arg2>                                          \
class trait_name<T, Result(Arg1,Arg2)>                                                                       \
{                                                                                                            \
class yes { char m; };                                                                                    \
class no { yes m[2]; };                                                                                   \
struct base_mixin                                                                                         \
{                                                                                                         \
Result member_function_name(Arg1,Arg2);                                                                 \
};                                                                                                        \
struct base : public T, public base_mixin {};                                                             \
template <typename U, U t>  class helper{};                                                               \
template <typename U>                                                                                     \
static no deduce(U*, helper<Result (base_mixin::*)(Arg1,Arg2), &U::member_function_name>* = 0);           \
static yes deduce(...);                                                                                   \
public:                                                                                                      \
static const bool value = sizeof(yes) == sizeof(deduce(static_cast<base*>(0)));                           \
typedef hydra_thrust::detail::integral_constant<bool,value> type;                                               \
};                                                                                                           \
\
template<typename T, typename Result, typename Arg1, typename Arg2, typename Arg3>                           \
class trait_name<T, Result(Arg1,Arg2,Arg3)>                                                                  \
{                                                                                                            \
class yes { char m; };                                                                                    \
class no { yes m[2]; };                                                                                   \
struct base_mixin                                                                                         \
{                                                                                                         \
Result member_function_name(Arg1,Arg2,Arg3);                                                            \
};                                                                                                        \
struct base : public T, public base_mixin {};                                                             \
template <typename U, U t>  class helper{};                                                               \
template <typename U>                                                                                     \
static no deduce(U*, helper<Result (base_mixin::*)(Arg1,Arg2,Arg3), &U::member_function_name>* = 0);      \
static yes deduce(...);                                                                                   \
public:                                                                                                      \
static const bool value = sizeof(yes) == sizeof(deduce(static_cast<base*>(0)));                           \
typedef hydra_thrust::detail::integral_constant<bool,value> type;                                               \
};                                                                                                           \
\
template<typename T, typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4>            \
class trait_name<T, Result(Arg1,Arg2,Arg3,Arg4)>                                                             \
{                                                                                                            \
class yes { char m; };                                                                                    \
class no { yes m[2]; };                                                                                   \
struct base_mixin                                                                                         \
{                                                                                                         \
Result member_function_name(Arg1,Arg2,Arg3,Arg4);                                                       \
};                                                                                                        \
struct base : public T, public base_mixin {};                                                             \
template <typename U, U t>  class helper{};                                                               \
template <typename U>                                                                                     \
static no deduce(U*, helper<Result (base_mixin::*)(Arg1,Arg2,Arg3,Arg4), &U::member_function_name>* = 0); \
static yes deduce(...);                                                                                   \
public:                                                                                                      \
static const bool value = sizeof(yes) == sizeof(deduce(static_cast<base*>(0)));                           \
typedef hydra_thrust::detail::integral_constant<bool,value> type;                                               \
};                                                                                                           

