
#ifndef BOOST_ASIO_SSL_DETAIL_WRITE_OP_HPP
#define BOOST_ASIO_SSL_DETAIL_WRITE_OP_HPP

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

template <typename ConstBufferSequence>
class write_op
{
public:
static BOOST_ASIO_CONSTEXPR const char* tracking_name()
{
return "ssl::stream<>::async_write_some";
}

write_op(const ConstBufferSequence& buffers)
: buffers_(buffers)
{
}

engine::want operator()(engine& eng,
boost::system::error_code& ec,
std::size_t& bytes_transferred) const
{
unsigned char storage[
boost::asio::detail::buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence>::linearisation_storage_size];

boost::asio::const_buffer buffer =
boost::asio::detail::buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence>::linearise(buffers_, boost::asio::buffer(storage));

return eng.write(buffer, ec, bytes_transferred);
}

template <typename Handler>
void call_handler(Handler& handler,
const boost::system::error_code& ec,
const std::size_t& bytes_transferred) const
{
handler(ec, bytes_transferred);
}

private:
ConstBufferSequence buffers_;
};

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
