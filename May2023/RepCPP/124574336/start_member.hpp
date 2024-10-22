
#ifndef BOOST_ASIO_TRAITS_START_MEMBER_HPP
#define BOOST_ASIO_TRAITS_START_MEMBER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>

#if defined(BOOST_ASIO_HAS_DECLTYPE) \
&& defined(BOOST_ASIO_HAS_NOEXCEPT) \
&& defined(BOOST_ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define BOOST_ASIO_HAS_DEDUCED_START_MEMBER_TRAIT 1
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace traits {

template <typename T, typename = void>
struct start_member_default;

template <typename T, typename = void>
struct start_member;

} 
namespace detail {

struct no_start_member
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(BOOST_ASIO_HAS_DEDUCED_START_MEMBER_TRAIT)

template <typename T, typename = void>
struct start_member_trait : no_start_member
{
};

template <typename T>
struct start_member_trait<T,
typename void_type<
decltype(declval<T>().start())
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(declval<T>().start());

BOOST_ASIO_STATIC_CONSTEXPR(bool,
is_noexcept = noexcept(declval<T>().start()));
};

#else 

template <typename T, typename = void>
struct start_member_trait :
conditional<
is_same<T, typename remove_reference<T>::type>::value,
typename conditional<
is_same<T, typename add_const<T>::type>::value,
no_start_member,
traits::start_member<typename add_const<T>::type>
>::type,
traits::start_member<typename remove_reference<T>::type>
>::type
{
};

#endif 

} 
namespace traits {

template <typename T, typename>
struct start_member_default :
detail::start_member_trait<T>
{
};

template <typename T, typename>
struct start_member :
start_member_default<T>
{
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
