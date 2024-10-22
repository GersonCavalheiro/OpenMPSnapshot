
#ifndef BOOST_ASIO_TRAITS_STATIC_REQUIRE_HPP
#define BOOST_ASIO_TRAITS_STATIC_REQUIRE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/traits/static_query.hpp>

#if defined(BOOST_ASIO_HAS_DECLTYPE) \
&& defined(BOOST_ASIO_HAS_NOEXCEPT)
# define BOOST_ASIO_HAS_DEDUCED_STATIC_REQUIRE_TRAIT 1
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace traits {

template <typename T, typename Property, typename = void>
struct static_require_default;

template <typename T, typename Property, typename = void>
struct static_require;

} 
namespace detail {

struct no_static_require
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
};

template <typename T, typename Property, typename = void>
struct static_require_trait :
conditional<
is_same<T, typename decay<T>::type>::value
&& is_same<Property, typename decay<Property>::type>::value,
no_static_require,
traits::static_require<
typename decay<T>::type,
typename decay<Property>::type>
>::type
{
};

#if defined(BOOST_ASIO_HAS_DEDUCED_STATIC_REQUIRE_TRAIT)

#if defined(BOOST_ASIO_HAS_WORKING_EXPRESSION_SFINAE)

template <typename T, typename Property>
struct static_require_trait<T, Property,
typename enable_if<
decay<Property>::type::value() == traits::static_query<T, Property>::value()
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
};

#else 

false_type static_require_test(...);

template <typename T, typename Property>
true_type static_require_test(T*, Property*,
typename enable_if<
Property::value() == traits::static_query<T, Property>::value()
>::type* = 0);

template <typename T, typename Property>
struct has_static_require
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, value =
decltype((static_require_test)(
static_cast<T*>(0), static_cast<Property*>(0)))::value);
};

template <typename T, typename Property>
struct static_require_trait<T, Property,
typename enable_if<
has_static_require<typename decay<T>::type,
typename decay<Property>::type>::value
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
};

#endif 

#endif 

} 
namespace traits {

template <typename T, typename Property, typename>
struct static_require_default : detail::static_require_trait<T, Property>
{
};

template <typename T, typename Property, typename>
struct static_require : static_require_default<T, Property>
{
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
