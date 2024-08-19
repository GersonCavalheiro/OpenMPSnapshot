
#ifndef ASIO_REQUIRE_CONCEPT_HPP
#define ASIO_REQUIRE_CONCEPT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/is_applicable_property.hpp"
#include "asio/traits/require_concept_member.hpp"
#include "asio/traits/require_concept_free.hpp"
#include "asio/traits/static_require_concept.hpp"

#include "asio/detail/push_options.hpp"

#if defined(GENERATING_DOCUMENTATION)

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

#else 

namespace asio_require_concept_fn {

using asio::conditional;
using asio::decay;
using asio::declval;
using asio::enable_if;
using asio::is_applicable_property;
using asio::traits::require_concept_free;
using asio::traits::require_concept_member;
using asio::traits::static_require_concept;

void require_concept();

enum overload_type
{
identity,
call_member,
call_free,
ill_formed
};

template <typename Impl, typename T, typename Properties, typename = void,
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
decay<Property>::type::is_requirable_concept
>::type,
typename enable_if<
static_require_concept<T, Property>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(overload_type, overload = identity);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);
typedef ASIO_MOVE_ARG(T) result_type;
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
decay<Property>::type::is_requirable_concept
>::type,
typename enable_if<
!static_require_concept<T, Property>::is_valid
>::type,
typename enable_if<
require_concept_member<
typename Impl::template proxy<T>::type,
Property
>::is_valid
>::type> :
require_concept_member<
typename Impl::template proxy<T>::type,
Property
>
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
decay<Property>::type::is_requirable_concept
>::type,
typename enable_if<
!static_require_concept<T, Property>::is_valid
>::type,
typename enable_if<
!require_concept_member<
typename Impl::template proxy<T>::type,
Property
>::is_valid
>::type,
typename enable_if<
require_concept_free<T, Property>::is_valid
>::type> :
require_concept_free<T, Property>
{
ASIO_STATIC_CONSTEXPR(overload_type, overload = call_free);
};

struct impl
{
template <typename T>
struct proxy
{
#if defined(ASIO_HAS_DEDUCED_REQUIRE_CONCEPT_MEMBER_TRAIT)
struct type
{
template <typename P>
auto require_concept(ASIO_MOVE_ARG(P) p)
noexcept(
noexcept(
declval<typename conditional<true, T, P>::type>().require_concept(
ASIO_MOVE_CAST(P)(p))
)
)
-> decltype(
declval<typename conditional<true, T, P>::type>().require_concept(
ASIO_MOVE_CAST(P)(p))
);
};
#else 
typedef T type;
#endif 
};

template <typename T, typename Property>
ASIO_NODISCARD ASIO_CONSTEXPR typename enable_if<
call_traits<impl, T, void(Property)>::overload == identity,
typename call_traits<impl, T, void(Property)>::result_type
>::type
operator()(
ASIO_MOVE_ARG(T) t,
ASIO_MOVE_ARG(Property)) const
ASIO_NOEXCEPT_IF((
call_traits<impl, T, void(Property)>::is_noexcept))
{
return ASIO_MOVE_CAST(T)(t);
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
return ASIO_MOVE_CAST(T)(t).require_concept(
ASIO_MOVE_CAST(Property)(p));
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
return require_concept(
ASIO_MOVE_CAST(T)(t),
ASIO_MOVE_CAST(Property)(p));
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

static ASIO_CONSTEXPR const asio_require_concept_fn::impl&
require_concept = asio_require_concept_fn::static_instance<>::instance;

} 

typedef asio_require_concept_fn::impl require_concept_t;

template <typename T, typename Property>
struct can_require_concept :
integral_constant<bool,
asio_require_concept_fn::call_traits<
require_concept_t, T, void(Property)>::overload !=
asio_require_concept_fn::ill_formed>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename Property>
constexpr bool can_require_concept_v
= can_require_concept<T, Property>::value;

#endif 

template <typename T, typename Property>
struct is_nothrow_require_concept :
integral_constant<bool,
asio_require_concept_fn::call_traits<
require_concept_t, T, void(Property)>::is_noexcept>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename Property>
constexpr bool is_nothrow_require_concept_v
= is_nothrow_require_concept<T, Property>::value;

#endif 

template <typename T, typename Property>
struct require_concept_result
{
typedef typename asio_require_concept_fn::call_traits<
require_concept_t, T, void(Property)>::result_type type;
};

} 

#endif 

#include "asio/detail/pop_options.hpp"

#endif 
