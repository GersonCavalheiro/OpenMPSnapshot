
#ifndef ASIO_TRAITS_SUBMIT_MEMBER_HPP
#define ASIO_TRAITS_SUBMIT_MEMBER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#if defined(ASIO_HAS_DECLTYPE) \
&& defined(ASIO_HAS_NOEXCEPT) \
&& defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define ASIO_HAS_DEDUCED_SUBMIT_MEMBER_TRAIT 1
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace traits {

template <typename S, typename R, typename = void>
struct submit_member_default;

template <typename S, typename R, typename = void>
struct submit_member;

} 
namespace detail {

struct no_submit_member
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(ASIO_HAS_DEDUCED_SUBMIT_MEMBER_TRAIT)

template <typename S, typename R, typename = void>
struct submit_member_trait : no_submit_member
{
};

template <typename S, typename R>
struct submit_member_trait<S, R,
typename void_type<
decltype(declval<S>().submit(declval<R>()))
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(
declval<S>().submit(declval<R>()));

ASIO_STATIC_CONSTEXPR(bool, is_noexcept = noexcept(
declval<S>().submit(declval<R>())));
};

#else 

template <typename S, typename R, typename = void>
struct submit_member_trait :
conditional<
is_same<S, typename remove_reference<S>::type>::value
&& is_same<R, typename decay<R>::type>::value,
typename conditional<
is_same<S, typename add_const<S>::type>::value,
no_submit_member,
traits::submit_member<typename add_const<S>::type, R>
>::type,
traits::submit_member<
typename remove_reference<S>::type,
typename decay<R>::type>
>::type
{
};

#endif 

} 
namespace traits {

template <typename S, typename R, typename>
struct submit_member_default :
detail::submit_member_trait<S, R>
{
};

template <typename S, typename R, typename>
struct submit_member :
submit_member_default<S, R>
{
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
