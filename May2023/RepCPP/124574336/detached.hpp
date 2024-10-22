
#ifndef BOOST_ASIO_IMPL_DETACHED_HPP
#define BOOST_ASIO_IMPL_DETACHED_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/async_result.hpp>
#include <boost/asio/detail/variadic_templates.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class detached_handler
{
public:
typedef void result_type;

detached_handler(detached_t)
{
}

#if defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES)

template <typename... Args>
void operator()(Args...)
{
}

#else 

void operator()()
{
}

#define BOOST_ASIO_PRIVATE_DETACHED_DEF(n) \
template <BOOST_ASIO_VARIADIC_TPARAMS(n)> \
void operator()(BOOST_ASIO_VARIADIC_TARGS(n)) \
{ \
} \

BOOST_ASIO_VARIADIC_GENERATE(BOOST_ASIO_PRIVATE_DETACHED_DEF)
#undef BOOST_ASIO_PRIVATE_DETACHED_DEF

#endif 
};

} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename Signature>
struct async_result<detached_t, Signature>
{
typedef boost::asio::detail::detached_handler completion_handler_type;

typedef void return_type;

explicit async_result(completion_handler_type&)
{
}

void get()
{
}

#if defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Initiation, typename RawCompletionToken, typename... Args>
static return_type initiate(
BOOST_ASIO_MOVE_ARG(Initiation) initiation,
BOOST_ASIO_MOVE_ARG(RawCompletionToken),
BOOST_ASIO_MOVE_ARG(Args)... args)
{
BOOST_ASIO_MOVE_CAST(Initiation)(initiation)(
detail::detached_handler(detached_t()),
BOOST_ASIO_MOVE_CAST(Args)(args)...);
}

#else 

template <typename Initiation, typename RawCompletionToken>
static return_type initiate(
BOOST_ASIO_MOVE_ARG(Initiation) initiation,
BOOST_ASIO_MOVE_ARG(RawCompletionToken))
{
BOOST_ASIO_MOVE_CAST(Initiation)(initiation)(
detail::detached_handler(detached_t()));
}

#define BOOST_ASIO_PRIVATE_INITIATE_DEF(n) \
template <typename Initiation, typename RawCompletionToken, \
BOOST_ASIO_VARIADIC_TPARAMS(n)> \
static return_type initiate( \
BOOST_ASIO_MOVE_ARG(Initiation) initiation, \
BOOST_ASIO_MOVE_ARG(RawCompletionToken), \
BOOST_ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
BOOST_ASIO_MOVE_CAST(Initiation)(initiation)( \
detail::detached_handler(detached_t()), \
BOOST_ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

BOOST_ASIO_VARIADIC_GENERATE(BOOST_ASIO_PRIVATE_INITIATE_DEF)
#undef BOOST_ASIO_PRIVATE_INITIATE_DEF

#endif 
};

#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
