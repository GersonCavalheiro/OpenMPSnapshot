
#ifndef ASIO_ASSOCIATED_EXECUTOR_HPP
#define ASIO_ASSOCIATED_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/associator.hpp"
#include "asio/detail/functional.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/executor.hpp"
#include "asio/is_executor.hpp"
#include "asio/system_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <typename T, typename Executor>
struct associated_executor;

namespace detail {

template <typename T, typename = void>
struct has_executor_type : false_type
{
};

template <typename T>
struct has_executor_type<T,
typename void_type<typename T::executor_type>::type>
: true_type
{
};

template <typename T, typename E, typename = void, typename = void>
struct associated_executor_impl
{
typedef void asio_associated_executor_is_unspecialised;

typedef E type;

static type get(const T&, const E& e = E()) ASIO_NOEXCEPT
{
return e;
}
};

template <typename T, typename E>
struct associated_executor_impl<T, E,
typename void_type<typename T::executor_type>::type>
{
typedef typename T::executor_type type;

static type get(const T& t, const E& = E()) ASIO_NOEXCEPT
{
return t.get_executor();
}
};

template <typename T, typename E>
struct associated_executor_impl<T, E,
typename enable_if<
!has_executor_type<T>::value
>::type,
typename void_type<
typename associator<associated_executor, T, E>::type
>::type> : associator<associated_executor, T, E>
{
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
const Executor& ex = Executor()) ASIO_NOEXCEPT;
#endif 
};


template <typename T>
inline typename associated_executor<T>::type
get_associated_executor(const T& t) ASIO_NOEXCEPT
{
return associated_executor<T>::get(t);
}


template <typename T, typename Executor>
inline typename associated_executor<T, Executor>::type
get_associated_executor(const T& t, const Executor& ex,
typename constraint<
is_executor<Executor>::value || execution::is_executor<Executor>::value
>::type = 0) ASIO_NOEXCEPT
{
return associated_executor<T, Executor>::get(t, ex);
}


template <typename T, typename ExecutionContext>
inline typename associated_executor<T,
typename ExecutionContext::executor_type>::type
get_associated_executor(const T& t, ExecutionContext& ctx,
typename constraint<is_convertible<ExecutionContext&,
execution_context&>::value>::type = 0) ASIO_NOEXCEPT
{
return associated_executor<T,
typename ExecutionContext::executor_type>::get(t, ctx.get_executor());
}

#if defined(ASIO_HAS_ALIAS_TEMPLATES)

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

#if defined(ASIO_HAS_STD_REFERENCE_WRAPPER) \
|| defined(GENERATING_DOCUMENTATION)

template <typename T, typename Executor>
struct associated_executor<reference_wrapper<T>, Executor>
#if !defined(GENERATING_DOCUMENTATION)
: detail::associated_executor_forwarding_base<T, Executor>
#endif 
{
typedef typename associated_executor<T, Executor>::type type;

static type get(reference_wrapper<T> t,
const Executor& ex = Executor()) ASIO_NOEXCEPT
{
return associated_executor<T, Executor>::get(t.get(), ex);
}
};

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
