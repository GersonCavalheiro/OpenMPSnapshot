
#ifndef BOOST_ASIO_DETAIL_NULL_REACTOR_HPP
#define BOOST_ASIO_DETAIL_NULL_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_IOCP) || defined(BOOST_ASIO_WINDOWS_RUNTIME)

#include <boost/asio/detail/scheduler_operation.hpp>
#include <boost/asio/execution_context.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class null_reactor
: public execution_context_service_base<null_reactor>
{
public:
null_reactor(boost::asio::execution_context& ctx)
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
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
