
#ifndef ASIO_SSL_DETAIL_WRITE_OP_HPP
#define ASIO_SSL_DETAIL_WRITE_OP_HPP

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

template <typename ConstBufferSequence>
class write_op
{
public:
static ASIO_CONSTEXPR const char* tracking_name()
{
return "ssl::stream<>::async_write_some";
}

write_op(const ConstBufferSequence& buffers)
: buffers_(buffers)
{
}

engine::want operator()(engine& eng,
asio::error_code& ec,
std::size_t& bytes_transferred) const
{
unsigned char storage[
asio::detail::buffer_sequence_adapter<asio::const_buffer,
ConstBufferSequence>::linearisation_storage_size];

asio::const_buffer buffer =
asio::detail::buffer_sequence_adapter<asio::const_buffer,
ConstBufferSequence>::linearise(buffers_, asio::buffer(storage));

return eng.write(buffer, ec, bytes_transferred);
}

template <typename Handler>
void call_handler(Handler& handler,
const asio::error_code& ec,
const std::size_t& bytes_transferred) const
{
ASIO_MOVE_OR_LVALUE(Handler)(handler)(ec, bytes_transferred);
}

private:
ConstBufferSequence buffers_;
};

} 
} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
