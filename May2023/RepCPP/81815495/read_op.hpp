
#ifndef ASIO_SSL_DETAIL_READ_OP_HPP
#define ASIO_SSL_DETAIL_READ_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/ssl/detail/engine.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

template <typename MutableBufferSequence>
class read_op
{
public:
static ASIO_CONSTEXPR const char* tracking_name()
{
return "ssl::stream<>::async_read_some";
}

read_op(const MutableBufferSequence& buffers)
: buffers_(buffers)
{
}

engine::want operator()(engine& eng,
asio::error_code& ec,
std::size_t& bytes_transferred) const
{
asio::mutable_buffer buffer =
asio::detail::buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence>::first(buffers_);

return eng.read(buffer, ec, bytes_transferred);
}

template <typename Handler>
void call_handler(Handler& handler,
const asio::error_code& ec,
const std::size_t& bytes_transferred) const
{
ASIO_MOVE_OR_LVALUE(Handler)(handler)(ec, bytes_transferred);
}

private:
MutableBufferSequence buffers_;
};

} 
} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
