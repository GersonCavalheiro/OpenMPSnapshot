
#ifndef ASIO_DETAIL_REACTIVE_SOCKET_ACCEPT_OP_HPP
#define ASIO_DETAIL_REACTIVE_SOCKET_ACCEPT_OP_HPP

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
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Socket, typename Protocol>
class reactive_socket_accept_op_base : public reactor_op
{
public:
reactive_socket_accept_op_base(const asio::error_code& success_ec,
socket_type socket, socket_ops::state_type state, Socket& peer,
const Protocol& protocol, typename Protocol::endpoint* peer_endpoint,
func_type complete_func)
: reactor_op(success_ec,
&reactive_socket_accept_op_base::do_perform, complete_func),
socket_(socket),
state_(state),
peer_(peer),
protocol_(protocol),
peer_endpoint_(peer_endpoint),
addrlen_(peer_endpoint ? peer_endpoint->capacity() : 0)
{
}

static status do_perform(reactor_op* base)
{
reactive_socket_accept_op_base* o(
static_cast<reactive_socket_accept_op_base*>(base));

socket_type new_socket = invalid_socket;
status result = socket_ops::non_blocking_accept(o->socket_,
o->state_, o->peer_endpoint_ ? o->peer_endpoint_->data() : 0,
o->peer_endpoint_ ? &o->addrlen_ : 0, o->ec_, new_socket)
? done : not_done;
o->new_socket_.reset(new_socket);

ASIO_HANDLER_REACTOR_OPERATION((*o, "non_blocking_accept", o->ec_));

return result;
}

void do_assign()
{
if (new_socket_.get() != invalid_socket)
{
if (peer_endpoint_)
peer_endpoint_->resize(addrlen_);
peer_.assign(protocol_, new_socket_.get(), ec_);
if (!ec_)
new_socket_.release();
}
}

private:
socket_type socket_;
socket_ops::state_type state_;
socket_holder new_socket_;
Socket& peer_;
Protocol protocol_;
typename Protocol::endpoint* peer_endpoint_;
std::size_t addrlen_;
};

template <typename Socket, typename Protocol,
typename Handler, typename IoExecutor>
class reactive_socket_accept_op :
public reactive_socket_accept_op_base<Socket, Protocol>
{
public:
ASIO_DEFINE_HANDLER_PTR(reactive_socket_accept_op);

reactive_socket_accept_op(const asio::error_code& success_ec,
socket_type socket, socket_ops::state_type state, Socket& peer,
const Protocol& protocol, typename Protocol::endpoint* peer_endpoint,
Handler& handler, const IoExecutor& io_ex)
: reactive_socket_accept_op_base<Socket, Protocol>(
success_ec, socket, state, peer, protocol, peer_endpoint,
&reactive_socket_accept_op::do_complete),
handler_(ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

static void do_complete(void* owner, operation* base,
const asio::error_code& ,
std::size_t )
{
reactive_socket_accept_op* o(static_cast<reactive_socket_accept_op*>(base));
ptr p = { asio::detail::addressof(o->handler_), o, o };

if (owner)
o->do_assign();

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

#if defined(ASIO_HAS_MOVE)

template <typename Protocol, typename PeerIoExecutor,
typename Handler, typename IoExecutor>
class reactive_socket_move_accept_op :
private Protocol::socket::template rebind_executor<PeerIoExecutor>::other,
public reactive_socket_accept_op_base<
typename Protocol::socket::template rebind_executor<PeerIoExecutor>::other,
Protocol>
{
public:
ASIO_DEFINE_HANDLER_PTR(reactive_socket_move_accept_op);

reactive_socket_move_accept_op(const asio::error_code& success_ec,
const PeerIoExecutor& peer_io_ex, socket_type socket,
socket_ops::state_type state, const Protocol& protocol,
typename Protocol::endpoint* peer_endpoint, Handler& handler,
const IoExecutor& io_ex)
: peer_socket_type(peer_io_ex),
reactive_socket_accept_op_base<peer_socket_type, Protocol>(
success_ec, socket, state, *this, protocol, peer_endpoint,
&reactive_socket_move_accept_op::do_complete),
handler_(ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

static void do_complete(void* owner, operation* base,
const asio::error_code& ,
std::size_t )
{
reactive_socket_move_accept_op* o(
static_cast<reactive_socket_move_accept_op*>(base));
ptr p = { asio::detail::addressof(o->handler_), o, o };

if (owner)
o->do_assign();

ASIO_HANDLER_COMPLETION((*o));

handler_work<Handler, IoExecutor> w(
ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
o->work_));

detail::move_binder2<Handler,
asio::error_code, peer_socket_type>
handler(0, ASIO_MOVE_CAST(Handler)(o->handler_), o->ec_,
ASIO_MOVE_CAST(peer_socket_type)(*o));
p.h = asio::detail::addressof(handler.handler_);
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_, "..."));
w.complete(handler, handler.handler_);
ASIO_HANDLER_INVOCATION_END;
}
}

private:
typedef typename Protocol::socket::template
rebind_executor<PeerIoExecutor>::other peer_socket_type;

Handler handler_;
handler_work<Handler, IoExecutor> work_;
};

#endif 

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
