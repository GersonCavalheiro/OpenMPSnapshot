
#ifndef BOOST_ASIO_SSL_DETAIL_SHUTDOWN_OP_HPP
#define BOOST_ASIO_SSL_DETAIL_SHUTDOWN_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#include <boost/asio/ssl/detail/engine.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ssl {
namespace detail {

class shutdown_op
{
public:
static BOOST_ASIO_CONSTEXPR const char* tracking_name()
{
return "ssl::stream<>::async_shutdown";
}

engine::want operator()(engine& eng,
boost::system::error_code& ec,
std::size_t& bytes_transferred) const
{
bytes_transferred = 0;
return eng.shutdown(ec);
}

template <typename Handler>
void call_handler(Handler& handler,
const boost::system::error_code& ec,
const std::size_t&) const
{
if (ec == boost::asio::error::eof)
{
handler(boost::system::error_code());
}
else
{
handler(ec);
}
}
};

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
