
#ifndef ASIO_DETAIL_WIN_IOCP_NULL_BUFFERS_OP_HPP
#define ASIO_DETAIL_WIN_IOCP_NULL_BUFFERS_OP_HPP

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

template <typename Handler, typename IoExecutor>
class win_iocp_null_buffers_op : public reactor_op
{
public:
ASIO_DEFINE_HANDLER_PTR(win_iocp_null_buffers_op);

win_iocp_null_buffers_op(socket_ops::weak_cancel_token_type cancel_token,
Handler& handler, const IoExecutor& io_ex)
: reactor_op(asio::error_code(),
&win_iocp_null_buffers_op::do_perform,
&win_iocp_null_buffers_op::do_complete),
cancel_token_(cancel_token),
handler_(ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

static status do_perform(reactor_op*)
{
return done;
}

static void do_complete(void* owner, operation* base,
const asio::error_code& result_ec,
std::size_t bytes_transferred)
{
asio::error_code ec(result_ec);

win_iocp_null_buffers_op* o(static_cast<win_iocp_null_buffers_op*>(base));
ptr p = { asio::detail::addressof(o->handler_), o, o };

ASIO_HANDLER_COMPLETION((*o));

handler_work<Handler, IoExecutor> w(
ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
o->work_));

if (o->ec_)
ec = o->ec_;

if (ec.value() == ERROR_NETNAME_DELETED)
{
if (o->cancel_token_.expired())
ec = asio::error::operation_aborted;
else
ec = asio::error::connection_reset;
}
else if (ec.value() == ERROR_PORT_UNREACHABLE)
{
ec = asio::error::connection_refused;
}

detail::binder2<Handler, asio::error_code, std::size_t>
handler(o->handler_, ec, bytes_transferred);
p.h = asio::detail::addressof(handler.handler_);
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_, handler.arg2_));
w.complete(handler, handler.handler_);
ASIO_HANDLER_INVOCATION_END;
}
}

private:
socket_ops::weak_cancel_token_type cancel_token_;
Handler handler_;
handler_work<Handler, IoExecutor> work_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
