
#ifndef ASIO_DETAIL_REACTIVE_SOCKET_CONNECT_OP_HPP
#define ASIO_DETAIL_REACTIVE_SOCKET_CONNECT_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/handler_work.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/socket_ops.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class reactive_socket_connect_op_base : public reactor_op
{
public:
reactive_socket_connect_op_base(const asio::error_code& success_ec,
socket_type socket, func_type complete_func)
: reactor_op(success_ec,
&reactive_socket_connect_op_base::do_perform, complete_func),
socket_(socket)
{
}

static status do_perform(reactor_op* base)
{
reactive_socket_connect_op_base* o(
static_cast<reactive_socket_connect_op_base*>(base));

status result = socket_ops::non_blocking_connect(
o->socket_, o->ec_) ? done : not_done;

ASIO_HANDLER_REACTOR_OPERATION((*o, "non_blocking_connect", o->ec_));

return result;
}

private:
socket_type socket_;
};

template <typename Handler, typename IoExecutor>
class reactive_socket_connect_op : public reactive_socket_connect_op_base
{
public:
ASIO_DEFINE_HANDLER_PTR(reactive_socket_connect_op);

reactive_socket_connect_op(const asio::error_code& success_ec,
socket_type socket, Handler& handler, const IoExecutor& io_ex)
: reactive_socket_connect_op_base(success_ec, socket,
&reactive_socket_connect_op::do_complete),
handler_(ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

static void do_complete(void* owner, operation* base,
const asio::error_code& ,
std::size_t )
{
reactive_socket_connect_op* o
(static_cast<reactive_socket_connect_op*>(base));
ptr p = { asio::detail::addressof(o->handler_), o, o };

ASIO_HANDLER_COMPLETION((*o));

handler_work<Handler, IoExecutor> w(
ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
o->work_));

detail::binder1<Handler, asio::error_code>
handler(o->handler_, o->ec_);
p.h = asio::detail::addressof(handler.handler_);
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_));
w.complete(handler, handler.handler_);
ASIO_HANDLER_INVOCATION_END;
}
}

private:
Handler handler_;
handler_work<Handler, IoExecutor> work_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
