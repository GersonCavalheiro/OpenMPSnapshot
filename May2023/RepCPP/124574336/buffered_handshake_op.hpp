
#ifndef BOOST_ASIO_SSL_DETAIL_BUFFERED_HANDSHAKE_OP_HPP
#define BOOST_ASIO_SSL_DETAIL_BUFFERED_HANDSHAKE_OP_HPP

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

template <typename ConstBufferSequence>
class buffered_handshake_op
{
public:
static BOOST_ASIO_CONSTEXPR const char* tracking_name()
{
return "ssl::stream<>::async_buffered_handshake";
}

buffered_handshake_op(stream_base::handshake_type type,
const ConstBufferSequence& buffers)
: type_(type),
buffers_(buffers),
total_buffer_size_(boost::asio::buffer_size(buffers_))
{
}

engine::want operator()(engine& eng,
boost::system::error_code& ec,
std::size_t& bytes_transferred) const
{
return this->process(eng, ec, bytes_transferred,
boost::asio::buffer_sequence_begin(buffers_),
boost::asio::buffer_sequence_end(buffers_));
}

template <typename Handler>
void call_handler(Handler& handler,
const boost::system::error_code& ec,
const std::size_t& bytes_transferred) const
{
handler(ec, bytes_transferred);
}

private:
template <typename Iterator>
engine::want process(engine& eng,
boost::system::error_code& ec,
std::size_t& bytes_transferred,
Iterator begin, Iterator end) const
{
Iterator iter = begin;
std::size_t accumulated_size = 0;

for (;;)
{
engine::want want = eng.handshake(type_, ec);
if (want != engine::want_input_and_retry
|| bytes_transferred == total_buffer_size_)
return want;

while (iter != end)
{
const_buffer buffer(*iter);

if (bytes_transferred >= accumulated_size + buffer.size())
{
accumulated_size += buffer.size();
++iter;
continue;
}

if (bytes_transferred > accumulated_size)
buffer = buffer + (bytes_transferred - accumulated_size);

bytes_transferred += buffer.size();
buffer = eng.put_input(buffer);
bytes_transferred -= buffer.size();
break;
}
}
}

stream_base::handshake_type type_;
ConstBufferSequence buffers_;
std::size_t total_buffer_size_;
};

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
