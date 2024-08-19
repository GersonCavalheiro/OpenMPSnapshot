
#ifndef BOOST_ASIO_QUERY_HPP
#define BOOST_ASIO_QUERY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/is_applicable_property.hpp>
#include <boost/asio/traits/query_member.hpp>
#include <boost/asio/traits/query_free.hpp>
#include <boost/asio/traits/static_query.hpp>

#include <boost/asio/detail/push_options.hpp>

#if defined(GENERATING_DOCUMENTATION)

namespace boost {
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
} 

#else 

namespace asio_query_fn {

using boost::asio::decay;
using boost::asio::declval;
using boost::asio::enable_if;
using boost::asio::is_applicable_property;
using boost::asio::traits::query_free;
using boost::asio::traits::query_member;
using boost::asio::traits::static_query;

void query();

enum overload_type
{
static_value,
call_member,
call_free,
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
static_query<T, Property>::is_valid
)
>::type> :
static_query<T, Property>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = static_value);
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
!static_query<T, Property>::is_valid
&&
query_member<T, Property>::is_valid
)
>::type> :
query_member<T, Property>
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
!static_query<T, Property>::is_valid
&&
!query_member<T, Property>::is_valid
&&
query_free<T, Property>::is_valid
)
>::type> :
query_free<T, Property>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = call_free);
};

struct impl
{
template <typename T, typename Property>
BOOST_ASIO_NODISCARD BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<T, void(Property)>::overload == static_value,
typename call_traits<T, void(Property)>::result_type
>::type
operator()(
BOOST_ASIO_MOVE_ARG(T),
BOOST_ASIO_MOVE_ARG(Property)) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<T, void(Property)>::is_noexcept))
{
return static_query<
typename decay<T>::type,
typename decay<Property>::type
>::value();
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
return BOOST_ASIO_MOVE_CAST(T)(t).query(BOOST_ASIO_MOVE_CAST(Property)(p));
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
return query(BOOST_ASIO_MOVE_CAST(T)(t), BOOST_ASIO_MOVE_CAST(Property)(p));
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

static BOOST_ASIO_CONSTEXPR const asio_query_fn::impl&
query = asio_query_fn::static_instance<>::instance;

} 

template <typename T, typename Property>
struct can_query :
integral_constant<bool,
asio_query_fn::call_traits<T, void(Property)>::overload !=
asio_query_fn::ill_formed>
{
};

#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename Property>
constexpr bool can_query_v
= can_query<T, Property>::value;

#endif 

template <typename T, typename Property>
struct is_nothrow_query :
integral_constant<bool,
asio_query_fn::call_traits<T, void(Property)>::is_noexcept>
{
};

#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename Property>
constexpr bool is_nothrow_query_v
= is_nothrow_query<T, Property>::value;

#endif 

template <typename T, typename Property>
struct query_result
{
typedef typename asio_query_fn::call_traits<
T, void(Property)>::result_type type;
};

} 
} 

#endif 

#include <boost/asio/detail/pop_options.hpp>

#endif 
