
#ifndef BOOST_ASIO_DETAIL_WIN_IOCP_SOCKET_CONNECT_OP_HPP
#define BOOST_ASIO_DETAIL_WIN_IOCP_SOCKET_CONNECT_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_IOCP)

#include <boost/asio/detail/bind_handler.hpp>
#include <boost/asio/detail/fenced_block.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/handler_invoke_helpers.hpp>
#include <boost/asio/detail/handler_work.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/reactor_op.hpp>
#include <boost/asio/detail/socket_ops.hpp>
#include <boost/asio/error.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class win_iocp_socket_connect_op_base : public reactor_op
{
public:
win_iocp_socket_connect_op_base(socket_type socket, func_type complete_func)
: reactor_op(boost::system::error_code(),
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
BOOST_ASIO_DEFINE_HANDLER_PTR(win_iocp_socket_connect_op);

win_iocp_socket_connect_op(socket_type socket,
Handler& handler, const IoExecutor& io_ex)
: win_iocp_socket_connect_op_base(socket,
&win_iocp_socket_connect_op::do_complete),
handler_(BOOST_ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

static void do_complete(void* owner, operation* base,
const boost::system::error_code& result_ec,
std::size_t )
{
boost::system::error_code ec(result_ec);

win_iocp_socket_connect_op* o(
static_cast<win_iocp_socket_connect_op*>(base));
ptr p = { boost::asio::detail::addressof(o->handler_), o, o };

if (owner)
{
if (o->connect_ex_)
socket_ops::complete_iocp_connect(o->socket_, ec);
else
ec = o->ec_;
}

BOOST_ASIO_HANDLER_COMPLETION((*o));

handler_work<Handler, IoExecutor> w(
BOOST_ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
o->work_));

detail::binder1<Handler, boost::system::error_code>
handler(o->handler_, ec);
p.h = boost::asio::detail::addressof(handler.handler_);
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
BOOST_ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_));
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

#endif 
