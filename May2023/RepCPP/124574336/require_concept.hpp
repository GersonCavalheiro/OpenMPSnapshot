
#ifndef BOOST_ASIO_REQUIRE_CONCEPT_HPP
#define BOOST_ASIO_REQUIRE_CONCEPT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/is_applicable_property.hpp>
#include <boost/asio/traits/require_concept_member.hpp>
#include <boost/asio/traits/require_concept_free.hpp>
#include <boost/asio/traits/static_require_concept.hpp>

#include <boost/asio/detail/push_options.hpp>

#if defined(GENERATING_DOCUMENTATION)

namespace boost {
namespace asio {


inline constexpr unspecified require_concept = unspecified;


template <typename T, typename Property>
struct can_require_concept :
integral_constant<bool, automatically_determined>
{
};


template <typename T, typename Property>
struct is_nothrow_require_concept :
integral_constant<bool, automatically_determined>
{
};


template <typename T, typename Property>
struct require_concept_result
{
typedef automatically_determined type;
};

} 
} 

#else 

namespace asio_require_concept_fn {

using boost::asio::decay;
using boost::asio::declval;
using boost::asio::enable_if;
using boost::asio::is_applicable_property;
using boost::asio::traits::require_concept_free;
using boost::asio::traits::require_concept_member;
using boost::asio::traits::static_require_concept;

void require_concept();

enum overload_type
{
identity,
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
decay<Property>::type::is_requirable_concept
&&
static_require_concept<T, Property>::is_valid
)
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = identity);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);
typedef BOOST_ASIO_MOVE_ARG(T) result_type;
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
decay<Property>::type::is_requirable_concept
&&
!static_require_concept<T, Property>::is_valid
&&
require_concept_member<T, Property>::is_valid
)
>::type> :
require_concept_member<T, Property>
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
decay<Property>::type::is_requirable_concept
&&
!static_require_concept<T, Property>::is_valid
&&
!require_concept_member<T, Property>::is_valid
&&
require_concept_free<T, Property>::is_valid
)
>::type> :
require_concept_free<T, Property>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = call_free);
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
return BOOST_ASIO_MOVE_CAST(T)(t).require_concept(
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
return require_concept(
BOOST_ASIO_MOVE_CAST(T)(t),
BOOST_ASIO_MOVE_CAST(Property)(p));
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

static BOOST_ASIO_CONSTEXPR const asio_require_concept_fn::impl&
require_concept = asio_require_concept_fn::static_instance<>::instance;

} 

template <typename T, typename Property>
struct can_require_concept :
integral_constant<bool,
asio_require_concept_fn::call_traits<T, void(Property)>::overload !=
asio_require_concept_fn::ill_formed>
{
};

#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename Property>
constexpr bool can_require_concept_v
= can_require_concept<T, Property>::value;

#endif 

template <typename T, typename Property>
struct is_nothrow_require_concept :
integral_constant<bool,
asio_require_concept_fn::call_traits<T, void(Property)>::is_noexcept>
{
};

#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename Property>
constexpr bool is_nothrow_require_concept_v
= is_nothrow_require_concept<T, Property>::value;

#endif 

template <typename T, typename Property>
struct require_concept_result
{
typedef typename asio_require_concept_fn::call_traits<
T, void(Property)>::result_type type;
};

} 
} 

#endif 

#include <boost/asio/detail/pop_options.hpp>

#endif 
