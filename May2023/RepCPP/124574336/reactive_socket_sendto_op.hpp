
#ifndef BOOST_ASIO_DETAIL_REACTIVE_SOCKET_SENDTO_OP_HPP
#define BOOST_ASIO_DETAIL_REACTIVE_SOCKET_SENDTO_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/bind_handler.hpp>
#include <boost/asio/detail/buffer_sequence_adapter.hpp>
#include <boost/asio/detail/fenced_block.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/handler_invoke_helpers.hpp>
#include <boost/asio/detail/handler_work.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/reactor_op.hpp>
#include <boost/asio/detail/socket_ops.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename ConstBufferSequence, typename Endpoint>
class reactive_socket_sendto_op_base : public reactor_op
{
public:
reactive_socket_sendto_op_base(const boost::system::error_code& success_ec,
socket_type socket, const ConstBufferSequence& buffers,
const Endpoint& endpoint, socket_base::message_flags flags,
func_type complete_func)
: reactor_op(success_ec,
&reactive_socket_sendto_op_base::do_perform, complete_func),
socket_(socket),
buffers_(buffers),
destination_(endpoint),
flags_(flags)
{
}

static status do_perform(reactor_op* base)
{
reactive_socket_sendto_op_base* o(
static_cast<reactive_socket_sendto_op_base*>(base));

typedef buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence> bufs_type;

status result;
if (bufs_type::is_single_buffer)
{
result = socket_ops::non_blocking_sendto1(o->socket_,
bufs_type::first(o->buffers_).data(),
bufs_type::first(o->buffers_).size(), o->flags_,
o->destination_.data(), o->destination_.size(),
o->ec_, o->bytes_transferred_) ? done : not_done;
}
else
{
bufs_type bufs(o->buffers_);
result = socket_ops::non_blocking_sendto(o->socket_,
bufs.buffers(), bufs.count(), o->flags_,
o->destination_.data(), o->destination_.size(),
o->ec_, o->bytes_transferred_) ? done : not_done;
}

BOOST_ASIO_HANDLER_REACTOR_OPERATION((*o, "non_blocking_sendto",
o->ec_, o->bytes_transferred_));

return result;
}

private:
socket_type socket_;
ConstBufferSequence buffers_;
Endpoint destination_;
socket_base::message_flags flags_;
};

template <typename ConstBufferSequence, typename Endpoint,
typename Handler, typename IoExecutor>
class reactive_socket_sendto_op :
public reactive_socket_sendto_op_base<ConstBufferSequence, Endpoint>
{
public:
BOOST_ASIO_DEFINE_HANDLER_PTR(reactive_socket_sendto_op);

reactive_socket_sendto_op(const boost::system::error_code& success_ec,
socket_type socket, const ConstBufferSequence& buffers,
const Endpoint& endpoint, socket_base::message_flags flags,
Handler& handler, const IoExecutor& io_ex)
: reactive_socket_sendto_op_base<ConstBufferSequence, Endpoint>(
success_ec, socket, buffers, endpoint, flags,
&reactive_socket_sendto_op::do_complete),
handler_(BOOST_ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

static void do_complete(void* owner, operation* base,
const boost::system::error_code& ,
std::size_t )
{
reactive_socket_sendto_op* o(static_cast<reactive_socket_sendto_op*>(base));
ptr p = { boost::asio::detail::addressof(o->handler_), o, o };

BOOST_ASIO_HANDLER_COMPLETION((*o));

handler_work<Handler, IoExecutor> w(
BOOST_ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
o->work_));

detail::binder2<Handler, boost::system::error_code, std::size_t>
handler(o->handler_, o->ec_, o->bytes_transferred_);
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
Handler handler_;
handler_work<Handler, IoExecutor> work_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
