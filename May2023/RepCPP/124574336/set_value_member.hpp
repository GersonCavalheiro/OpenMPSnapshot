
#ifndef BOOST_ASIO_TRAITS_SET_VALUE_MEMBER_HPP
#define BOOST_ASIO_TRAITS_SET_VALUE_MEMBER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/detail/variadic_templates.hpp>

#if defined(BOOST_ASIO_HAS_DECLTYPE) \
&& defined(BOOST_ASIO_HAS_NOEXCEPT) \
&& defined(BOOST_ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define BOOST_ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT 1
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace traits {

template <typename T, typename Vs, typename = void>
struct set_value_member_default;

template <typename T, typename Vs, typename = void>
struct set_value_member;

} 
namespace detail {

struct no_set_value_member
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(BOOST_ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT)

template <typename T, typename Vs, typename = void>
struct set_value_member_trait : no_set_value_member
{
};

#if defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES)

template <typename T, typename... Vs>
struct set_value_member_trait<T, void(Vs...),
typename void_type<
decltype(declval<T>().set_value(declval<Vs>()...))
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(
declval<T>().set_value(declval<Vs>()...));

BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = noexcept(
declval<T>().set_value(declval<Vs>()...)));
};

#else 

template <typename T>
struct set_value_member_trait<T, void(),
typename void_type<
decltype(declval<T>().set_value())
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(declval<T>().set_value());

BOOST_ASIO_STATIC_CONSTEXPR(bool,
is_noexcept = noexcept(declval<T>().set_value()));
};

#define BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_TRAIT_DEF(n) \
template <typename T, BOOST_ASIO_VARIADIC_TPARAMS(n)> \
struct set_value_member_trait<T, void(BOOST_ASIO_VARIADIC_TARGS(n)), \
typename void_type< \
decltype(declval<T>().set_value(BOOST_ASIO_VARIADIC_DECLVAL(n))) \
>::type> \
{ \
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true); \
\
using result_type = decltype( \
declval<T>().set_value(BOOST_ASIO_VARIADIC_DECLVAL(n))); \
\
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = noexcept( \
declval<T>().set_value(BOOST_ASIO_VARIADIC_DECLVAL(n)))); \
}; \

BOOST_ASIO_VARIADIC_GENERATE(BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_TRAIT_DEF)
#undef BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_TRAIT_DEF

#endif 

#else 

template <typename T, typename Vs, typename = void>
struct set_value_member_trait;

#if defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES)

template <typename T, typename... Vs>
struct set_value_member_trait<T, void(Vs...)> :
conditional<
is_same<T, typename remove_reference<T>::type>::value
&& conjunction<is_same<Vs, typename decay<Vs>::type>...>::value,
typename conditional<
is_same<T, typename add_const<T>::type>::value,
no_set_value_member,
traits::set_value_member<typename add_const<T>::type, void(Vs...)>
>::type,
traits::set_value_member<
typename remove_reference<T>::type,
void(typename decay<Vs>::type...)>
>::type
{
};

#else 

template <typename T>
struct set_value_member_trait<T, void()> :
conditional<
is_same<T, typename remove_reference<T>::type>::value,
typename conditional<
is_same<T, typename add_const<T>::type>::value,
no_set_value_member,
traits::set_value_member<typename add_const<T>::type, void()>
>::type,
traits::set_value_member<typename remove_reference<T>::type, void()>
>::type
{
};

#define BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME(n) \
BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_##n

#define BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_1 \
&& is_same<T1, typename decay<T1>::type>::value
#define BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_2 \
BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_1 \
&& is_same<T2, typename decay<T2>::type>::value
#define BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_3 \
BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_2 \
&& is_same<T3, typename decay<T3>::type>::value
#define BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_4 \
BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_3 \
&& is_same<T4, typename decay<T4>::type>::value
#define BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_5 \
BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_4 \
&& is_same<T5, typename decay<T5>::type>::value
#define BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_6 \
BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_5 \
&& is_same<T6, typename decay<T6>::type>::value
#define BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_7 \
BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_6 \
&& is_same<T7, typename decay<T7>::type>::value
#define BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_8 \
BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_7 \
&& is_same<T8, typename decay<T8>::type>::value

#define BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_TRAIT_DEF(n) \
template <typename T, BOOST_ASIO_VARIADIC_TPARAMS(n)> \
struct set_value_member_trait<T, void(BOOST_ASIO_VARIADIC_TARGS(n))> : \
conditional< \
is_same<T, typename remove_reference<T>::type>::value \
BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME(n), \
typename conditional< \
is_same<T, typename add_const<T>::type>::value, \
no_set_value_member, \
traits::set_value_member< \
typename add_const<T>::type, \
void(BOOST_ASIO_VARIADIC_TARGS(n))> \
>::type, \
traits::set_value_member< \
typename remove_reference<T>::type, \
void(BOOST_ASIO_VARIADIC_DECAY(n))> \
>::type \
{ \
}; \

BOOST_ASIO_VARIADIC_GENERATE(BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_TRAIT_DEF)
#undef BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_TRAIT_DEF
#undef BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME
#undef BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_1
#undef BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_2
#undef BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_3
#undef BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_4
#undef BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_5
#undef BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_6
#undef BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_7
#undef BOOST_ASIO_PRIVATE_SET_VALUE_MEMBER_IS_SAME_8

#endif 

#endif 

} 
namespace traits {

template <typename T, typename Vs, typename>
struct set_value_member_default :
detail::set_value_member_trait<T, Vs>
{
};

template <typename T, typename Vs, typename>
struct set_value_member :
set_value_member_default<T, Vs>
{
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
