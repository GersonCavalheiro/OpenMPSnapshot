
#ifndef ASIO_DETAIL_NULL_REACTOR_HPP
#define ASIO_DETAIL_NULL_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP) || defined(ASIO_WINDOWS_RUNTIME)

#include "asio/detail/scheduler_operation.hpp"
#include "asio/execution_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class null_reactor
: public execution_context_service_base<null_reactor>
{
public:
null_reactor(asio::execution_context& ctx)
: execution_context_service_base<null_reactor>(ctx)
{
}

~null_reactor()
{
}

void shutdown()
{
}

void run(long , op_queue<scheduler_operation>& )
{
}

void interrupt()
{
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
