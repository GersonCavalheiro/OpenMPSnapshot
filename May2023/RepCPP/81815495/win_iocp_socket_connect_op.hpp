
#ifndef ASIO_DETAIL_WIN_IOCP_SOCKET_CONNECT_OP_HPP
#define ASIO_DETAIL_WIN_IOCP_SOCKET_CONNECT_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/detail/bind_handler.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/handler_work.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class win_iocp_socket_connect_op_base : public reactor_op
{
public:
win_iocp_socket_connect_op_base(socket_type socket, func_type complete_func)
: reactor_op(asio::error_code(),
&win_iocp_socket_connect_op_base::do_perform, complete_func),
socket_(socket),
connect_ex_(false)
{
}

static status do_perform(reactor_op* base)
{
win_iocp_socket_connect_op_base* o(
static_cast<win_iocp_socket_connect_op_base*>(base));

return socket_ops::non_blocking_connect(
o->socket_, o->ec_) ? done : not_done;
}

socket_type socket_;
bool connect_ex_;
};

template <typename Handler, typename IoExecutor>
class win_iocp_socket_connect_op : public win_iocp_socket_connect_op_base
{
public:
ASIO_DEFINE_HANDLER_PTR(win_iocp_socket_connect_op);

win_iocp_socket_connect_op(socket_type socket,
Handler& handler, const IoExecutor& io_ex)
: win_iocp_socket_connect_op_base(socket,
&win_iocp_socket_connect_op::do_complete),
handler_(ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

static void do_complete(void* owner, operation* base,
const asio::error_code& result_ec,
std::size_t )
{
asio::error_code ec(result_ec);

win_iocp_socket_connect_op* o(
static_cast<win_iocp_socket_connect_op*>(base));
ptr p = { asio::detail::addressof(o->handler_), o, o };

if (owner)
{
if (o->connect_ex_)
socket_ops::complete_iocp_connect(o->socket_, ec);
else
ec = o->ec_;
}

ASIO_HANDLER_COMPLETION((*o));

handler_work<Handler, IoExecutor> w(
ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
o->work_));

detail::binder1<Handler, asio::error_code>
handler(o->handler_, ec);
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

#endif 
