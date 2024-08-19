
#ifndef ASIO_SSL_DETAIL_SHUTDOWN_OP_HPP
#define ASIO_SSL_DETAIL_SHUTDOWN_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/ssl/detail/engine.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

class shutdown_op
{
public:
static ASIO_CONSTEXPR const char* tracking_name()
{
return "ssl::stream<>::async_shutdown";
}

engine::want operator()(engine& eng,
asio::error_code& ec,
std::size_t& bytes_transferred) const
{
bytes_transferred = 0;
return eng.shutdown(ec);
}

template <typename Handler>
void call_handler(Handler& handler,
const asio::error_code& ec,
const std::size_t&) const
{
if (ec == asio::error::eof)
{
ASIO_MOVE_OR_LVALUE(Handler)(handler)(asio::error_code());
}
else
{
ASIO_MOVE_OR_LVALUE(Handler)(handler)(ec);
}
}
};

} 
} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
