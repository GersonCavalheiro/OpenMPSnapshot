
#ifndef ASIO_EXECUTION_CONTEXT_AS_HPP
#define ASIO_EXECUTION_CONTEXT_AS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/context.hpp"
#include "asio/execution/executor.hpp"
#include "asio/execution/scheduler.hpp"
#include "asio/execution/sender.hpp"
#include "asio/is_applicable_property.hpp"
#include "asio/query.hpp"
#include "asio/traits/query_static_constexpr_member.hpp"
#include "asio/traits/static_query.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(GENERATING_DOCUMENTATION)

namespace execution {

template <typename U>
struct context_as_t
{
template <typename T>
static constexpr bool is_applicable_property_v =
is_executor_v<T> || is_sender_v<T> || is_scheduler_v<T>;

static constexpr bool is_requirable = false;

static constexpr bool is_preferable = false;

typedef T polymorphic_query_result_type;
};

template <typename U>
constexpr context_as_t context_as;

} 

#else 

namespace execution {

template <typename T>
struct context_as_t
{
#if defined(ASIO_HAS_VARIABLE_TEMPLATES)
template <typename U>
ASIO_STATIC_CONSTEXPR(bool,
is_applicable_property_v = (
is_executor<U>::value
|| conditional<
is_executor<U>::value,
false_type,
is_sender<U>
>::type::value
|| conditional<
is_executor<U>::value,
false_type,
is_scheduler<U>
>::type::value));
#endif 

ASIO_STATIC_CONSTEXPR(bool, is_requirable = false);
ASIO_STATIC_CONSTEXPR(bool, is_preferable = false);

typedef T polymorphic_query_result_type;

ASIO_CONSTEXPR context_as_t()
{
}

ASIO_CONSTEXPR context_as_t(context_t)
{
}

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <typename E>
static ASIO_CONSTEXPR
typename context_t::query_static_constexpr_member<E>::result_type
static_query()
ASIO_NOEXCEPT_IF((
context_t::query_static_constexpr_member<E>::is_noexcept))
{
return context_t::query_static_constexpr_member<E>::value();
}

template <typename E, typename U = decltype(context_as_t::static_query<E>())>
static ASIO_CONSTEXPR const U static_query_v
= context_as_t::static_query<E>();
#endif 

template <typename Executor, typename U>
friend ASIO_CONSTEXPR U query(
const Executor& ex, const context_as_t<U>&,
typename enable_if<
is_same<T, U>::value
>::type* = 0,
typename enable_if<
can_query<const Executor&, const context_t&>::value
>::type* = 0)
#if !defined(__clang__) 
#if defined(ASIO_MSVC) 
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, const context_t&>::value))
#else 
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, const context_t&>::value))
#endif 
#endif 
{
return asio::query(ex, context);
}
};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <typename T> template <typename E, typename U>
const U context_as_t<T>::static_query_v;
#endif 

#if (defined(ASIO_HAS_VARIABLE_TEMPLATES) \
&& defined(ASIO_HAS_CONSTEXPR)) \
|| defined(GENERATING_DOCUMENTATION)
template <typename T>
constexpr context_as_t<T> context_as{};
#endif 

} 

#if !defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename U>
struct is_applicable_property<T, execution::context_as_t<U> >
: integral_constant<bool,
execution::is_executor<T>::value
|| conditional<
execution::is_executor<T>::value,
false_type,
execution::is_sender<T>
>::type::value
|| conditional<
execution::is_executor<T>::value,
false_type,
execution::is_scheduler<T>
>::type::value>
{
};

#endif 

namespace traits {

#if !defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
|| !defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)

template <typename T, typename U>
struct static_query<T, execution::context_as_t<U>,
typename enable_if<
static_query<T, execution::context_t>::is_valid
>::type> : static_query<T, execution::context_t>
{
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_QUERY_FREE_TRAIT)

template <typename T, typename U>
struct query_free<T, execution::context_as_t<U>,
typename enable_if<
can_query<const T&, const execution::context_t&>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_query<const T&, const execution::context_t&>::value));

typedef U result_type;
};

#endif 

} 

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
