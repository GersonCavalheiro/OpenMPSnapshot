
#ifndef BOOST_ASIO_TRAITS_PREFER_MEMBER_HPP
#define BOOST_ASIO_TRAITS_PREFER_MEMBER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>

#if defined(BOOST_ASIO_HAS_DECLTYPE) \
&& defined(BOOST_ASIO_HAS_NOEXCEPT) \
&& defined(BOOST_ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define BOOST_ASIO_HAS_DEDUCED_PREFER_MEMBER_TRAIT 1
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace traits {

template <typename T, typename Property, typename = void>
struct prefer_member_default;

template <typename T, typename Property, typename = void>
struct prefer_member;

} 
namespace detail {

struct no_prefer_member
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(BOOST_ASIO_HAS_DEDUCED_PREFER_MEMBER_TRAIT)

template <typename T, typename Property, typename = void>
struct prefer_member_trait : no_prefer_member
{
};

template <typename T, typename Property>
struct prefer_member_trait<T, Property,
typename void_type<
decltype(declval<T>().prefer(declval<Property>()))
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(
declval<T>().prefer(declval<Property>()));

BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = noexcept(
declval<T>().prefer(declval<Property>())));
};

#else 

template <typename T, typename Property, typename = void>
struct prefer_member_trait :
conditional<
is_same<T, typename decay<T>::type>::value
&& is_same<Property, typename decay<Property>::type>::value,
no_prefer_member,
traits::prefer_member<
typename decay<T>::type,
typename decay<Property>::type>
>::type
{
};

#endif 

} 
namespace traits {

template <typename T, typename Property, typename>
struct prefer_member_default :
detail::prefer_member_trait<T, Property>
{
};

template <typename T, typename Property, typename>
struct prefer_member :
prefer_member_default<T, Property>
{
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
