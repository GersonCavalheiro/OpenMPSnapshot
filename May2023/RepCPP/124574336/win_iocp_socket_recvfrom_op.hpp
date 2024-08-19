
#ifndef BOOST_ASIO_DETAIL_WIN_IOCP_SOCKET_RECVFROM_OP_HPP
#define BOOST_ASIO_DETAIL_WIN_IOCP_SOCKET_RECVFROM_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_IOCP)

#include <boost/asio/detail/bind_handler.hpp>
#include <boost/asio/detail/buffer_sequence_adapter.hpp>
#include <boost/asio/detail/fenced_block.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/handler_invoke_helpers.hpp>
#include <boost/asio/detail/handler_work.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/operation.hpp>
#include <boost/asio/detail/socket_ops.hpp>
#include <boost/asio/error.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename MutableBufferSequence, typename Endpoint,
typename Handler, typename IoExecutor>
class win_iocp_socket_recvfrom_op : public operation
{
public:
BOOST_ASIO_DEFINE_HANDLER_PTR(win_iocp_socket_recvfrom_op);

win_iocp_socket_recvfrom_op(Endpoint& endpoint,
socket_ops::weak_cancel_token_type cancel_token,
const MutableBufferSequence& buffers, Handler& handler,
const IoExecutor& io_ex)
: operation(&win_iocp_socket_recvfrom_op::do_complete),
endpoint_(endpoint),
endpoint_size_(static_cast<int>(endpoint.capacity())),
cancel_token_(cancel_token),
buffers_(buffers),
handler_(BOOST_ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

int& endpoint_size()
{
return endpoint_size_;
}

static void do_complete(void* owner, operation* base,
const boost::system::error_code& result_ec,
std::size_t bytes_transferred)
{
boost::system::error_code ec(result_ec);

win_iocp_socket_recvfrom_op* o(
static_cast<win_iocp_socket_recvfrom_op*>(base));
ptr p = { boost::asio::detail::addressof(o->handler_), o, o };

BOOST_ASIO_HANDLER_COMPLETION((*o));

handler_work<Handler, IoExecutor> w(
BOOST_ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
o->work_));

#if defined(BOOST_ASIO_ENABLE_BUFFER_DEBUGGING)
if (owner)
{
buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence>::validate(o->buffers_);
}
#endif 

socket_ops::complete_iocp_recvfrom(o->cancel_token_, ec);

o->endpoint_.resize(o->endpoint_size_);

detail::binder2<Handler, boost::system::error_code, std::size_t>
handler(o->handler_, ec, bytes_transferred);
p.h = boost::asio::detail::addressof(handler.handler_);
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
BOOST_ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_, handler.arg2_));
w.complete(handler, handler.handler_);
BOOST_ASIO_HANDLER_INVOCATION_END;
}
}

private:
Endpoint& endpoint_;
int endpoint_size_;
socket_ops::weak_cancel_token_type cancel_token_;
MutableBufferSequence buffers_;
Handler handler_;
handler_work<Handler, IoExecutor> work_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
