
#ifndef BOOST_ASIO_EXECUTION_INVOCABLE_ARCHETYPE_HPP
#define BOOST_ASIO_EXECUTION_INVOCABLE_ARCHETYPE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/detail/variadic_templates.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace execution {

struct invocable_archetype
{
#if !defined(GENERATING_DOCUMENTATION)
typedef void result_type;
#endif 

#if defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES) \
|| defined(GENERATING_DOCUMENTATION)

template <typename... Args>
void operator()(BOOST_ASIO_MOVE_ARG(Args)...)
{
}

#else 

void operator()()
{
}

#define BOOST_ASIO_PRIVATE_INVOCABLE_ARCHETYPE_CALL_DEF(n) \
template <BOOST_ASIO_VARIADIC_TPARAMS(n)> \
void operator()(BOOST_ASIO_VARIADIC_UNNAMED_MOVE_PARAMS(n)) \
{ \
} \

BOOST_ASIO_VARIADIC_GENERATE(BOOST_ASIO_PRIVATE_INVOCABLE_ARCHETYPE_CALL_DEF)
#undef BOOST_ASIO_PRIVATE_INVOCABLE_ARCHETYPE_CALL_DEF

#endif 
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

