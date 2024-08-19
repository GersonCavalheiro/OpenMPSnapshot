
#ifndef BOOST_ASIO_ASSOCIATED_EXECUTOR_HPP
#define BOOST_ASIO_ASSOCIATED_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/execution/executor.hpp>
#include <boost/asio/is_executor.hpp>
#include <boost/asio/system_executor.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename T, typename E, typename = void>
struct associated_executor_impl
{
typedef void asio_associated_executor_is_unspecialised;

typedef E type;

static type get(const T&, const E& e = E()) BOOST_ASIO_NOEXCEPT
{
return e;
}
};

template <typename T, typename E>
struct associated_executor_impl<T, E,
typename void_type<typename T::executor_type>::type>
{
typedef typename T::executor_type type;

static type get(const T& t, const E& = E()) BOOST_ASIO_NOEXCEPT
{
return t.get_executor();
}
};

} 


template <typename T, typename Executor = system_executor>
struct associated_executor
#if !defined(GENERATING_DOCUMENTATION)
: detail::associated_executor_impl<T, Executor>
#endif 
{
#if defined(GENERATING_DOCUMENTATION)
typedef see_below type;

static type get(const T& t,
const Executor& ex = Executor()) BOOST_ASIO_NOEXCEPT;
#endif 
};


template <typename T>
inline typename associated_executor<T>::type
get_associated_executor(const T& t) BOOST_ASIO_NOEXCEPT
{
return associated_executor<T>::get(t);
}


template <typename T, typename Executor>
inline typename associated_executor<T, Executor>::type
get_associated_executor(const T& t, const Executor& ex,
typename enable_if<
is_executor<Executor>::value || execution::is_executor<Executor>::value
>::type* = 0) BOOST_ASIO_NOEXCEPT
{
return associated_executor<T, Executor>::get(t, ex);
}


template <typename T, typename ExecutionContext>
inline typename associated_executor<T,
typename ExecutionContext::executor_type>::type
get_associated_executor(const T& t, ExecutionContext& ctx,
typename enable_if<is_convertible<ExecutionContext&,
execution_context&>::value>::type* = 0) BOOST_ASIO_NOEXCEPT
{
return associated_executor<T,
typename ExecutionContext::executor_type>::get(t, ctx.get_executor());
}

#if defined(BOOST_ASIO_HAS_ALIAS_TEMPLATES)

template <typename T, typename Executor = system_executor>
using associated_executor_t = typename associated_executor<T, Executor>::type;

#endif 

namespace detail {

template <typename T, typename E, typename = void>
struct associated_executor_forwarding_base
{
};

template <typename T, typename E>
struct associated_executor_forwarding_base<T, E,
typename enable_if<
is_same<
typename associated_executor<T,
E>::asio_associated_executor_is_unspecialised,
void
>::value
>::type>
{
typedef void asio_associated_executor_is_unspecialised;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
