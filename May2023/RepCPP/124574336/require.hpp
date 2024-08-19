
#ifndef BOOST_ASIO_REQUIRE_HPP
#define BOOST_ASIO_REQUIRE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/is_applicable_property.hpp>
#include <boost/asio/traits/require_member.hpp>
#include <boost/asio/traits/require_free.hpp>
#include <boost/asio/traits/static_require.hpp>

#include <boost/asio/detail/push_options.hpp>

#if defined(GENERATING_DOCUMENTATION)

namespace boost {
namespace asio {


inline constexpr unspecified require = unspecified;


template <typename T, typename... Properties>
struct can_require :
integral_constant<bool, automatically_determined>
{
};


template <typename T, typename... Properties>
struct is_nothrow_require :
integral_constant<bool, automatically_determined>
{
};


template <typename T, typename... Properties>
struct require_result
{
typedef automatically_determined type;
};

} 
} 

#else 

namespace asio_require_fn {

using boost::asio::decay;
using boost::asio::declval;
using boost::asio::enable_if;
using boost::asio::is_applicable_property;
using boost::asio::traits::require_free;
using boost::asio::traits::require_member;
using boost::asio::traits::static_require;

void require();

enum overload_type
{
identity,
call_member,
call_free,
two_props,
n_props,
ill_formed
};

template <typename T, typename Properties, typename = void>
struct call_traits
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = ill_formed);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
typedef void result_type;
};

template <typename T, typename Property>
struct call_traits<T, void(Property),
typename enable_if<
(
is_applicable_property<
typename decay<T>::type,
typename decay<Property>::type
>::value
&&
decay<Property>::type::is_requirable
&&
static_require<T, Property>::is_valid
)
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = identity);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

#if defined(BOOST_ASIO_HAS_MOVE)
typedef BOOST_ASIO_MOVE_ARG(T) result_type;
#else 
typedef BOOST_ASIO_MOVE_ARG(typename decay<T>::type) result_type;
#endif 
};

template <typename T, typename Property>
struct call_traits<T, void(Property),
typename enable_if<
(
is_applicable_property<
typename decay<T>::type,
typename decay<Property>::type
>::value
&&
decay<Property>::type::is_requirable
&&
!static_require<T, Property>::is_valid
&&
require_member<T, Property>::is_valid
)
>::type> :
require_member<T, Property>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = call_member);
};

template <typename T, typename Property>
struct call_traits<T, void(Property),
typename enable_if<
(
is_applicable_property<
typename decay<T>::type,
typename decay<Property>::type
>::value
&&
decay<Property>::type::is_requirable
&&
!static_require<T, Property>::is_valid
&&
!require_member<T, Property>::is_valid
&&
require_free<T, Property>::is_valid
)
>::type> :
require_free<T, Property>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = call_free);
};

template <typename T, typename P0, typename P1>
struct call_traits<T, void(P0, P1),
typename enable_if<
call_traits<T, void(P0)>::overload != ill_formed
&&
call_traits<
typename call_traits<T, void(P0)>::result_type,
void(P1)
>::overload != ill_formed
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = two_props);

BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(
call_traits<T, void(P0)>::is_noexcept
&&
call_traits<
typename call_traits<T, void(P0)>::result_type,
void(P1)
>::is_noexcept
));

typedef typename decay<
typename call_traits<
typename call_traits<T, void(P0)>::result_type,
void(P1)
>::result_type
>::type result_type;
};

template <typename T, typename P0, typename P1, typename BOOST_ASIO_ELLIPSIS PN>
struct call_traits<T, void(P0, P1, PN BOOST_ASIO_ELLIPSIS),
typename enable_if<
call_traits<T, void(P0)>::overload != ill_formed
&&
call_traits<
typename call_traits<T, void(P0)>::result_type,
void(P1, PN BOOST_ASIO_ELLIPSIS)
>::overload != ill_formed
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = n_props);

BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(
call_traits<T, void(P0)>::is_noexcept
&&
call_traits<
typename call_traits<T, void(P0)>::result_type,
void(P1, PN BOOST_ASIO_ELLIPSIS)
>::is_noexcept
));

typedef typename decay<
typename call_traits<
typename call_traits<T, void(P0)>::result_type,
void(P1, PN BOOST_ASIO_ELLIPSIS)
>::result_type
>::type result_type;
};

struct impl
{
template <typename T, typename Property>
BOOST_ASIO_NODISCARD BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<T, void(Property)>::overload == identity,
typename call_traits<T, void(Property)>::result_type
>::type
operator()(
BOOST_ASIO_MOVE_ARG(T) t,
BOOST_ASIO_MOVE_ARG(Property)) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<T, void(Property)>::is_noexcept))
{
return BOOST_ASIO_MOVE_CAST(T)(t);
}

template <typename T, typename Property>
BOOST_ASIO_NODISCARD BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<T, void(Property)>::overload == call_member,
typename call_traits<T, void(Property)>::result_type
>::type
operator()(
BOOST_ASIO_MOVE_ARG(T) t,
BOOST_ASIO_MOVE_ARG(Property) p) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<T, void(Property)>::is_noexcept))
{
return BOOST_ASIO_MOVE_CAST(T)(t).require(
BOOST_ASIO_MOVE_CAST(Property)(p));
}

template <typename T, typename Property>
BOOST_ASIO_NODISCARD BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<T, void(Property)>::overload == call_free,
typename call_traits<T, void(Property)>::result_type
>::type
operator()(
BOOST_ASIO_MOVE_ARG(T) t,
BOOST_ASIO_MOVE_ARG(Property) p) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<T, void(Property)>::is_noexcept))
{
return require(
BOOST_ASIO_MOVE_CAST(T)(t),
BOOST_ASIO_MOVE_CAST(Property)(p));
}

template <typename T, typename P0, typename P1>
BOOST_ASIO_NODISCARD BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<T, void(P0, P1)>::overload == two_props,
typename call_traits<T, void(P0, P1)>::result_type
>::type
operator()(
BOOST_ASIO_MOVE_ARG(T) t,
BOOST_ASIO_MOVE_ARG(P0) p0,
BOOST_ASIO_MOVE_ARG(P1) p1) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<T, void(P0, P1)>::is_noexcept))
{
return (*this)(
(*this)(
BOOST_ASIO_MOVE_CAST(T)(t),
BOOST_ASIO_MOVE_CAST(P0)(p0)),
BOOST_ASIO_MOVE_CAST(P1)(p1));
}

template <typename T, typename P0, typename P1,
typename BOOST_ASIO_ELLIPSIS PN>
BOOST_ASIO_NODISCARD BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<T, void(P0, P1, PN BOOST_ASIO_ELLIPSIS)>::overload == n_props,
typename call_traits<T, void(P0, P1, PN BOOST_ASIO_ELLIPSIS)>::result_type
>::type
operator()(
BOOST_ASIO_MOVE_ARG(T) t,
BOOST_ASIO_MOVE_ARG(P0) p0,
BOOST_ASIO_MOVE_ARG(P1) p1,
BOOST_ASIO_MOVE_ARG(PN) BOOST_ASIO_ELLIPSIS pn) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<T, void(P0, P1, PN BOOST_ASIO_ELLIPSIS)>::is_noexcept))
{
return (*this)(
(*this)(
BOOST_ASIO_MOVE_CAST(T)(t),
BOOST_ASIO_MOVE_CAST(P0)(p0)),
BOOST_ASIO_MOVE_CAST(P1)(p1),
BOOST_ASIO_MOVE_CAST(PN)(pn) BOOST_ASIO_ELLIPSIS);
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
namespace boost {
namespace asio {
namespace {

static BOOST_ASIO_CONSTEXPR const asio_require_fn::impl&
require = asio_require_fn::static_instance<>::instance;

} 

#if defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES)

template <typename T, typename... Properties>
struct can_require :
integral_constant<bool,
asio_require_fn::call_traits<T, void(Properties...)>::overload
!= asio_require_fn::ill_formed>
{
};

#else 

template <typename T, typename P0 = void,
typename P1 = void, typename P2 = void>
struct can_require :
integral_constant<bool,
asio_require_fn::call_traits<T, void(P0, P1, P2)>::overload
!= asio_require_fn::ill_formed>
{
};

template <typename T, typename P0, typename P1>
struct can_require<T, P0, P1> :
integral_constant<bool,
asio_require_fn::call_traits<T, void(P0, P1)>::overload
!= asio_require_fn::ill_formed>
{
};

template <typename T, typename P0>
struct can_require<T, P0> :
integral_constant<bool,
asio_require_fn::call_traits<T, void(P0)>::overload
!= asio_require_fn::ill_formed>
{
};

template <typename T>
struct can_require<T> :
false_type
{
};

#endif 

#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename BOOST_ASIO_ELLIPSIS Properties>
constexpr bool can_require_v
= can_require<T, Properties BOOST_ASIO_ELLIPSIS>::value;

#endif 

#if defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES)

template <typename T, typename... Properties>
struct is_nothrow_require :
integral_constant<bool,
asio_require_fn::call_traits<T, void(Properties...)>::is_noexcept>
{
};

#else 

template <typename T, typename P0 = void,
typename P1 = void, typename P2 = void>
struct is_nothrow_require :
integral_constant<bool,
asio_require_fn::call_traits<T, void(P0, P1, P2)>::is_noexcept>
{
};

template <typename T, typename P0, typename P1>
struct is_nothrow_require<T, P0, P1> :
integral_constant<bool,
asio_require_fn::call_traits<T, void(P0, P1)>::is_noexcept>
{
};

template <typename T, typename P0>
struct is_nothrow_require<T, P0> :
integral_constant<bool,
asio_require_fn::call_traits<T, void(P0)>::is_noexcept>
{
};

template <typename T>
struct is_nothrow_require<T> :
false_type
{
};

#endif 

#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename BOOST_ASIO_ELLIPSIS Properties>
constexpr bool is_nothrow_require_v
= is_nothrow_require<T, Properties BOOST_ASIO_ELLIPSIS>::value;

#endif 

#if defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES)

template <typename T, typename... Properties>
struct require_result
{
typedef typename asio_require_fn::call_traits<
T, void(Properties...)>::result_type type;
};

#else 

template <typename T, typename P0 = void,
typename P1 = void, typename P2 = void>
struct require_result
{
typedef typename asio_require_fn::call_traits<
T, void(P0, P1, P2)>::result_type type;
};

template <typename T, typename P0, typename P1>
struct require_result<T, P0, P1>
{
typedef typename asio_require_fn::call_traits<
T, void(P0, P1)>::result_type type;
};

template <typename T, typename P0>
struct require_result<T, P0>
{
typedef typename asio_require_fn::call_traits<
T, void(P0)>::result_type type;
};

template <typename T>
struct require_result<T>
{
};

#endif 

} 
} 

#endif 

#include <boost/asio/detail/pop_options.hpp>

#endif 
