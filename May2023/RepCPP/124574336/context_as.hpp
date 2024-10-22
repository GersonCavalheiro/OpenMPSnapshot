
#ifndef BOOST_ASIO_EXECUTION_CONTEXT_AS_HPP
#define BOOST_ASIO_EXECUTION_CONTEXT_AS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/execution/context.hpp>
#include <boost/asio/execution/executor.hpp>
#include <boost/asio/execution/scheduler.hpp>
#include <boost/asio/execution/sender.hpp>
#include <boost/asio/is_applicable_property.hpp>
#include <boost/asio/query.hpp>
#include <boost/asio/traits/query_static_constexpr_member.hpp>
#include <boost/asio/traits/static_query.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
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
#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)
template <typename U>
BOOST_ASIO_STATIC_CONSTEXPR(bool,
is_applicable_property_v = is_executor<U>::value
|| is_sender<U>::value || is_scheduler<U>::value);
#endif 

BOOST_ASIO_STATIC_CONSTEXPR(bool, is_requirable = false);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_preferable = false);

typedef T polymorphic_query_result_type;

BOOST_ASIO_CONSTEXPR context_as_t()
{
}

BOOST_ASIO_CONSTEXPR context_as_t(context_t)
{
}

#if defined(BOOST_ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(BOOST_ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <typename E>
static BOOST_ASIO_CONSTEXPR
typename traits::query_static_constexpr_member<E, context_t>::result_type
static_query()
BOOST_ASIO_NOEXCEPT_IF((
traits::query_static_constexpr_member<E, context_t>::is_noexcept))
{
return traits::query_static_constexpr_member<E, context_t>::value();
}

template <typename E, typename U = decltype(context_as_t::static_query<E>())>
static BOOST_ASIO_CONSTEXPR const U static_query_v
= context_as_t::static_query<E>();
#endif 

template <typename Executor, typename U>
friend BOOST_ASIO_CONSTEXPR U query(
const Executor& ex, const context_as_t<U>&,
typename enable_if<
is_same<T, U>::value
&& can_query<const Executor&, const context_t&>::value
>::type* = 0)
#if !defined(__clang__) 
#if defined(BOOST_ASIO_MSVC) 
BOOST_ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, const context_t&>::value))
#else 
BOOST_ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, const context_t&>::value))
#endif 
#endif 
{
return boost::asio::query(ex, context);
}
};

#if defined(BOOST_ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(BOOST_ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <typename T> template <typename E, typename U>
const U context_as_t<T>::static_query_v;
#endif 

#if (defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES) \
&& defined(BOOST_ASIO_HAS_CONSTEXPR)) \
|| defined(GENERATING_DOCUMENTATION)
template <typename T>
constexpr context_as_t<T> context_as{};
#endif 

} 

#if !defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename U>
struct is_applicable_property<T, execution::context_as_t<U> >
: integral_constant<bool,
execution::is_executor<T>::value
|| execution::is_sender<T>::value
|| execution::is_scheduler<T>::value>
{
};

#endif 

namespace traits {

#if !defined(BOOST_ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
|| !defined(BOOST_ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)

template <typename T, typename U>
struct static_query<T, execution::context_as_t<U>,
typename enable_if<
static_query<T, execution::context_t>::is_valid
>::type> : static_query<T, execution::context_t>
{
};

#endif 

#if !defined(BOOST_ASIO_HAS_DEDUCED_QUERY_FREE_TRAIT)

template <typename T, typename U>
struct query_free<T, execution::context_as_t<U>,
typename enable_if<
can_query<const T&, const execution::context_t&>::value
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_query<const T&, const execution::context_t&>::value));

typedef U result_type;
};

#endif 

} 

#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
