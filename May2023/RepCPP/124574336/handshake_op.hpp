
#ifndef BOOST_ASIO_SSL_DETAIL_HANDSHAKE_OP_HPP
#define BOOST_ASIO_SSL_DETAIL_HANDSHAKE_OP_HPP

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

class handshake_op
{
public:
static BOOST_ASIO_CONSTEXPR const char* tracking_name()
{
return "ssl::stream<>::async_handshake";
}

handshake_op(stream_base::handshake_type type)
: type_(type)
{
}

engine::want operator()(engine& eng,
boost::system::error_code& ec,
std::size_t& bytes_transferred) const
{
bytes_transferred = 0;
return eng.handshake(type_, ec);
}

template <typename Handler>
void call_handler(Handler& handler,
const boost::system::error_code& ec,
const std::size_t&) const
{
handler(ec);
}

private:
stream_base::handshake_type type_;
};

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
