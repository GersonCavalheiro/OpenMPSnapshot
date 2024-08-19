
#ifndef ASIO_EXECUTION_INVOCABLE_ARCHETYPE_HPP
#define ASIO_EXECUTION_INVOCABLE_ARCHETYPE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/variadic_templates.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

struct invocable_archetype
{
#if !defined(GENERATING_DOCUMENTATION)
typedef void result_type;
#endif 

#if defined(ASIO_HAS_VARIADIC_TEMPLATES) \
|| defined(GENERATING_DOCUMENTATION)

template <typename... Args>
void operator()(ASIO_MOVE_ARG(Args)...)
{
}

#else 

void operator()()
{
}

#define ASIO_PRIVATE_INVOCABLE_ARCHETYPE_CALL_DEF(n) \
template <ASIO_VARIADIC_TPARAMS(n)> \
void operator()(ASIO_VARIADIC_UNNAMED_MOVE_PARAMS(n)) \
{ \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_INVOCABLE_ARCHETYPE_CALL_DEF)
#undef ASIO_PRIVATE_INVOCABLE_ARCHETYPE_CALL_DEF

#endif 
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

