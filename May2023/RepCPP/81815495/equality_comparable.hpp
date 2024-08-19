
#ifndef ASIO_TRAITS_EQUALITY_COMPARABLE_HPP
#define ASIO_TRAITS_EQUALITY_COMPARABLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#if defined(ASIO_HAS_DECLTYPE) \
&& defined(ASIO_HAS_NOEXCEPT) \
&& defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT 1
#endif 

namespace asio {
namespace traits {

template <typename T, typename = void>
struct equality_comparable_default;

template <typename T, typename = void>
struct equality_comparable;

} 
namespace detail {

struct no_equality_comparable
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

template <typename T, typename = void>
struct equality_comparable_trait : no_equality_comparable
{
};

template <typename T>
struct equality_comparable_trait<T,
typename void_type<
decltype(
static_cast<void>(
static_cast<bool>(declval<const T>() == declval<const T>())
),
static_cast<void>(
static_cast<bool>(declval<const T>() != declval<const T>())
)
)
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
noexcept(declval<const T>() == declval<const T>())
&& noexcept(declval<const T>() != declval<const T>()));
};

#else 

template <typename T, typename = void>
struct equality_comparable_trait :
conditional<
is_same<T, typename decay<T>::type>::value,
no_equality_comparable,
traits::equality_comparable<typename decay<T>::type>
>::type
{
};

#endif 

} 
namespace traits {

template <typename T, typename>
struct equality_comparable_default : detail::equality_comparable_trait<T>
{
};

template <typename T, typename>
struct equality_comparable : equality_comparable_default<T>
{
};

} 
} 

#endif 
