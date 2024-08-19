
#ifndef ASIO_EXECUTION_RECEIVER_HPP
#define ASIO_EXECUTION_RECEIVER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/variadic_templates.hpp"
#include "asio/execution/set_done.hpp"
#include "asio/execution/set_error.hpp"
#include "asio/execution/set_value.hpp"

#if defined(ASIO_HAS_STD_EXCEPTION_PTR)
# include <exception>
#else 
# include "asio/error_code.hpp"
#endif 

#if defined(ASIO_HAS_DEDUCED_SET_DONE_FREE_TRAIT) \
&& defined(ASIO_HAS_DEDUCED_SET_DONE_MEMBER_TRAIT) \
&& defined(ASIO_HAS_DEDUCED_SET_ERROR_FREE_TRAIT) \
&& defined(ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT) \
&& defined(ASIO_HAS_DEDUCED_SET_VALUE_FREE_TRAIT) \
&& defined(ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT) \
&& defined(ASIO_HAS_DEDUCED_RECEIVER_OF_FREE_TRAIT) \
&& defined(ASIO_HAS_DEDUCED_RECEIVER_OF_MEMBER_TRAIT)
# define ASIO_HAS_DEDUCED_EXECUTION_IS_RECEIVER_TRAIT 1
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename T, typename E>
struct is_receiver_base :
integral_constant<bool,
is_move_constructible<typename remove_cvref<T>::type>::value
&& is_constructible<typename remove_cvref<T>::type, T>::value
>
{
};

} 

#if defined(ASIO_HAS_STD_EXCEPTION_PTR)
# define ASIO_EXECUTION_RECEIVER_ERROR_DEFAULT = std::exception_ptr
#else 
# define ASIO_EXECUTION_RECEIVER_ERROR_DEFAULT \
= ::asio::error_code
#endif 


template <typename T, typename E ASIO_EXECUTION_RECEIVER_ERROR_DEFAULT>
struct is_receiver :
#if defined(GENERATING_DOCUMENTATION)
integral_constant<bool, automatically_determined>
#else 
conditional<
can_set_done<typename remove_cvref<T>::type>::value
&& is_nothrow_set_done<typename remove_cvref<T>::type>::value
&& can_set_error<typename remove_cvref<T>::type, E>::value
&& is_nothrow_set_error<typename remove_cvref<T>::type, E>::value,
detail::is_receiver_base<T, E>,
false_type
>::type
#endif 
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename E ASIO_EXECUTION_RECEIVER_ERROR_DEFAULT>
ASIO_CONSTEXPR const bool is_receiver_v = is_receiver<T, E>::value;

#endif 

#if defined(ASIO_HAS_CONCEPTS)

template <typename T, typename E ASIO_EXECUTION_RECEIVER_ERROR_DEFAULT>
ASIO_CONCEPT receiver = is_receiver<T, E>::value;

#define ASIO_EXECUTION_RECEIVER ::asio::execution::receiver

#else 

#define ASIO_EXECUTION_RECEIVER typename

#endif 

#if defined(ASIO_HAS_VARIADIC_TEMPLATES) \
|| defined(GENERATING_DOCUMENTATION)


template <typename T, typename... Vs>
struct is_receiver_of :
#if defined(GENERATING_DOCUMENTATION)
integral_constant<bool, automatically_determined>
#else 
conditional<
is_receiver<T>::value,
can_set_value<typename remove_cvref<T>::type, Vs...>,
false_type
>::type
#endif 
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename... Vs>
ASIO_CONSTEXPR const bool is_receiver_of_v =
is_receiver_of<T, Vs...>::value;

#endif 

#if defined(ASIO_HAS_CONCEPTS)

template <typename T, typename... Vs>
ASIO_CONCEPT receiver_of = is_receiver_of<T, Vs...>::value;

#define ASIO_EXECUTION_RECEIVER_OF_0 \
::asio::execution::receiver_of

#define ASIO_EXECUTION_RECEIVER_OF_1(v) \
::asio::execution::receiver_of<v>

#else 

#define ASIO_EXECUTION_RECEIVER_OF_0 typename
#define ASIO_EXECUTION_RECEIVER_OF_1(v) typename

#endif 

#else 

template <typename T, typename = void,
typename = void, typename = void, typename = void, typename = void,
typename = void, typename = void, typename = void, typename = void>
struct is_receiver_of;

template <typename T>
struct is_receiver_of<T> :
conditional<
is_receiver<T>::value,
can_set_value<typename remove_cvref<T>::type>,
false_type
>::type
{
};

#define ASIO_PRIVATE_RECEIVER_OF_TRAITS_DEF(n) \
template <typename T, ASIO_VARIADIC_TPARAMS(n)> \
struct is_receiver_of<T, ASIO_VARIADIC_TARGS(n)> : \
conditional< \
conditional<true, is_receiver<T>, void>::type::value, \
can_set_value< \
typename remove_cvref<T>::type, \
ASIO_VARIADIC_TARGS(n)>, \
false_type \
>::type \
{ \
}; \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_RECEIVER_OF_TRAITS_DEF)
#undef ASIO_PRIVATE_RECEIVER_OF_TRAITS_DEF

#define ASIO_EXECUTION_RECEIVER_OF_0 typename
#define ASIO_EXECUTION_RECEIVER_OF_1(v) typename

#endif 

#if defined(ASIO_HAS_VARIADIC_TEMPLATES) \
|| defined(GENERATING_DOCUMENTATION)


template <typename T, typename... Vs>
struct is_nothrow_receiver_of :
#if defined(GENERATING_DOCUMENTATION)
integral_constant<bool, automatically_determined>
#else 
integral_constant<bool,
is_receiver_of<T, Vs...>::value
&& is_nothrow_set_value<typename remove_cvref<T>::type, Vs...>::value
>
#endif 
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename... Vs>
ASIO_CONSTEXPR const bool is_nothrow_receiver_of_v =
is_nothrow_receiver_of<T, Vs...>::value;

#endif 

#else 

template <typename T, typename = void,
typename = void, typename = void, typename = void, typename = void,
typename = void, typename = void, typename = void, typename = void>
struct is_nothrow_receiver_of;

template <typename T>
struct is_nothrow_receiver_of<T> :
integral_constant<bool,
is_receiver_of<T>::value
&& is_nothrow_set_value<typename remove_cvref<T>::type>::value
>
{
};

#define ASIO_PRIVATE_NOTHROW_RECEIVER_OF_TRAITS_DEF(n) \
template <typename T, ASIO_VARIADIC_TPARAMS(n)> \
struct is_nothrow_receiver_of<T, ASIO_VARIADIC_TARGS(n)> : \
integral_constant<bool, \
is_receiver_of<T, ASIO_VARIADIC_TARGS(n)>::value \
&& is_nothrow_set_value<typename remove_cvref<T>::type, \
ASIO_VARIADIC_TARGS(n)>::value \
> \
{ \
}; \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_NOTHROW_RECEIVER_OF_TRAITS_DEF)
#undef ASIO_PRIVATE_NOTHROW_RECEIVER_OF_TRAITS_DEF

#endif 

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
