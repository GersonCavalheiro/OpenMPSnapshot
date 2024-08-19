
#ifndef ASIO_QUERY_HPP
#define ASIO_QUERY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/is_applicable_property.hpp"
#include "asio/traits/query_member.hpp"
#include "asio/traits/query_free.hpp"
#include "asio/traits/static_query.hpp"

#include "asio/detail/push_options.hpp"

#if defined(GENERATING_DOCUMENTATION)

namespace asio {


inline constexpr unspecified query = unspecified;


template <typename T, typename Property>
struct can_query :
integral_constant<bool, automatically_determined>
{
};


template <typename T, typename Property>
struct is_nothrow_query :
integral_constant<bool, automatically_determined>
{
};


template <typename T, typename Property>
struct query_result
{
typedef automatically_determined type;
};

} 

#else 

namespace asio_query_fn {

using asio::conditional;
using asio::decay;
using asio::declval;
using asio::enable_if;
using asio::is_applicable_property;
using asio::traits::query_free;
using asio::traits::query_member;
using asio::traits::static_query;

void query();

enum overload_type
{
static_value,
call_member,
call_free,
ill_formed
};

template <typename Impl, typename T, typename Properties,
typename = void, typename = void, typename = void, typename = void>
struct call_traits
{
ASIO_STATIC_CONSTEXPR(overload_type, overload = ill_formed);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
typedef void result_type;
};

template <typename Impl, typename T, typename Property>
struct call_traits<Impl, T, void(Property),
typename enable_if<
is_applicable_property<
typename decay<T>::type,
typename decay<Property>::type
>::value
>::type,
typename enable_if<
static_query<T, Property>::is_valid
>::type> :
static_query<T, Property>
{
ASIO_STATIC_CONSTEXPR(overload_type, overload = static_value);
};

template <typename Impl, typename T, typename Property>
struct call_traits<Impl, T, void(Property),
typename enable_if<
is_applicable_property<
typename decay<T>::type,
typename decay<Property>::type
>::value
>::type,
typename enable_if<
!static_query<T, Property>::is_valid
>::type,
typename enable_if<
query_member<typename Impl::template proxy<T>::type, Property>::is_valid
>::type> :
query_member<typename Impl::template proxy<T>::type, Property>
{
ASIO_STATIC_CONSTEXPR(overload_type, overload = call_member);
};

template <typename Impl, typename T, typename Property>
struct call_traits<Impl, T, void(Property),
typename enable_if<
is_applicable_property<
typename decay<T>::type,
typename decay<Property>::type
>::value
>::type,
typename enable_if<
!static_query<T, Property>::is_valid
>::type,
typename enable_if<
!query_member<typename Impl::template proxy<T>::type, Property>::is_valid
>::type,
typename enable_if<
query_free<T, Property>::is_valid
>::type> :
query_free<T, Property>
{
ASIO_STATIC_CONSTEXPR(overload_type, overload = call_free);
};

struct impl
{
template <typename T>
struct proxy
{
#if defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)
struct type
{
template <typename P>
auto query(ASIO_MOVE_ARG(P) p)
noexcept(
noexcept(
declval<typename conditional<true, T, P>::type>().query(
ASIO_MOVE_CAST(P)(p))
)
)
-> decltype(
declval<typename conditional<true, T, P>::type>().query(
ASIO_MOVE_CAST(P)(p))
);
};
#else 
typedef T type;
#endif 
};

template <typename T, typename Property>
ASIO_NODISCARD ASIO_CONSTEXPR typename enable_if<
call_traits<impl, T, void(Property)>::overload == static_value,
typename call_traits<impl, T, void(Property)>::result_type
>::type
operator()(
ASIO_MOVE_ARG(T),
ASIO_MOVE_ARG(Property)) const
ASIO_NOEXCEPT_IF((
call_traits<impl, T, void(Property)>::is_noexcept))
{
return static_query<
typename decay<T>::type,
typename decay<Property>::type
>::value();
}

template <typename T, typename Property>
ASIO_NODISCARD ASIO_CONSTEXPR typename enable_if<
call_traits<impl, T, void(Property)>::overload == call_member,
typename call_traits<impl, T, void(Property)>::result_type
>::type
operator()(
ASIO_MOVE_ARG(T) t,
ASIO_MOVE_ARG(Property) p) const
ASIO_NOEXCEPT_IF((
call_traits<impl, T, void(Property)>::is_noexcept))
{
return ASIO_MOVE_CAST(T)(t).query(ASIO_MOVE_CAST(Property)(p));
}

template <typename T, typename Property>
ASIO_NODISCARD ASIO_CONSTEXPR typename enable_if<
call_traits<impl, T, void(Property)>::overload == call_free,
typename call_traits<impl, T, void(Property)>::result_type
>::type
operator()(
ASIO_MOVE_ARG(T) t,
ASIO_MOVE_ARG(Property) p) const
ASIO_NOEXCEPT_IF((
call_traits<impl, T, void(Property)>::is_noexcept))
{
return query(ASIO_MOVE_CAST(T)(t), ASIO_MOVE_CAST(Property)(p));
}
};

template <typename T = impl>
struct static_instance
{
static const T instance;
};

template <typename T>
const T static_instance<T>::instance = {};

} 
namespace asio {
namespace {

static ASIO_CONSTEXPR const asio_query_fn::impl&
query = asio_query_fn::static_instance<>::instance;

} 

typedef asio_query_fn::impl query_t;

template <typename T, typename Property>
struct can_query :
integral_constant<bool,
asio_query_fn::call_traits<query_t, T, void(Property)>::overload !=
asio_query_fn::ill_formed>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename Property>
constexpr bool can_query_v
= can_query<T, Property>::value;

#endif 

template <typename T, typename Property>
struct is_nothrow_query :
integral_constant<bool,
asio_query_fn::call_traits<query_t, T, void(Property)>::is_noexcept>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename Property>
constexpr bool is_nothrow_query_v
= is_nothrow_query<T, Property>::value;

#endif 

template <typename T, typename Property>
struct query_result
{
typedef typename asio_query_fn::call_traits<
query_t, T, void(Property)>::result_type type;
};

} 

#endif 

#include "asio/detail/pop_options.hpp"

#endif 
