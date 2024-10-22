
#ifndef ASIO_IMPL_DETACHED_HPP
#define ASIO_IMPL_DETACHED_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/variadic_templates.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class detached_handler
{
public:
typedef void result_type;

detached_handler(detached_t)
{
}

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename... Args>
void operator()(Args...)
{
}

#else 

void operator()()
{
}

#define ASIO_PRIVATE_DETACHED_DEF(n) \
template <ASIO_VARIADIC_TPARAMS(n)> \
void operator()(ASIO_VARIADIC_TARGS(n)) \
{ \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_DETACHED_DEF)
#undef ASIO_PRIVATE_DETACHED_DEF

#endif 
};

} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename Signature>
struct async_result<detached_t, Signature>
{
typedef asio::detail::detached_handler completion_handler_type;

typedef void return_type;

explicit async_result(completion_handler_type&)
{
}

void get()
{
}

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Initiation, typename RawCompletionToken, typename... Args>
static return_type initiate(
ASIO_MOVE_ARG(Initiation) initiation,
ASIO_MOVE_ARG(RawCompletionToken),
ASIO_MOVE_ARG(Args)... args)
{
ASIO_MOVE_CAST(Initiation)(initiation)(
detail::detached_handler(detached_t()),
ASIO_MOVE_CAST(Args)(args)...);
}

#else 

template <typename Initiation, typename RawCompletionToken>
static return_type initiate(
ASIO_MOVE_ARG(Initiation) initiation,
ASIO_MOVE_ARG(RawCompletionToken))
{
ASIO_MOVE_CAST(Initiation)(initiation)(
detail::detached_handler(detached_t()));
}

#define ASIO_PRIVATE_INITIATE_DEF(n) \
template <typename Initiation, typename RawCompletionToken, \
ASIO_VARIADIC_TPARAMS(n)> \
static return_type initiate( \
ASIO_MOVE_ARG(Initiation) initiation, \
ASIO_MOVE_ARG(RawCompletionToken), \
ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
ASIO_MOVE_CAST(Initiation)(initiation)( \
detail::detached_handler(detached_t()), \
ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_INITIATE_DEF)
#undef ASIO_PRIVATE_INITIATE_DEF

#endif 
};

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
