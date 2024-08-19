
#ifndef BOOST_ASIO_SSL_DETAIL_READ_OP_HPP
#define BOOST_ASIO_SSL_DETAIL_READ_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#include <boost/asio/detail/buffer_sequence_adapter.hpp>
#include <boost/asio/ssl/detail/engine.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ssl {
namespace detail {

template <typename MutableBufferSequence>
class read_op
{
public:
static BOOST_ASIO_CONSTEXPR const char* tracking_name()
{
return "ssl::stream<>::async_read_some";
}

read_op(const MutableBufferSequence& buffers)
: buffers_(buffers)
{
}

engine::want operator()(engine& eng,
boost::system::error_code& ec,
std::size_t& bytes_transferred) const
{
boost::asio::mutable_buffer buffer =
boost::asio::detail::buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence>::first(buffers_);

return eng.read(buffer, ec, bytes_transferred);
}

template <typename Handler>
void call_handler(Handler& handler,
const boost::system::error_code& ec,
const std::size_t& bytes_transferred) const
{
handler(ec, bytes_transferred);
}

private:
MutableBufferSequence buffers_;
};

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
