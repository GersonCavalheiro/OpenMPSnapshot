
#ifndef BOOST_ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_BASE_HPP
#define BOOST_ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_HAS_IOCP) \
&& !defined(BOOST_ASIO_WINDOWS_RUNTIME)

#include <boost/asio/buffer.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/socket_base.hpp>
#include <boost/asio/detail/buffer_sequence_adapter.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/reactive_null_buffers_op.hpp>
#include <boost/asio/detail/reactive_socket_recv_op.hpp>
#include <boost/asio/detail/reactive_socket_recvmsg_op.hpp>
#include <boost/asio/detail/reactive_socket_send_op.hpp>
#include <boost/asio/detail/reactive_wait_op.hpp>
#include <boost/asio/detail/reactor.hpp>
#include <boost/asio/detail/reactor_op.hpp>
#include <boost/asio/detail/socket_holder.hpp>
#include <boost/asio/detail/socket_ops.hpp>
#include <boost/asio/detail/socket_types.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class reactive_socket_service_base
{
public:
typedef socket_type native_handle_type;

struct base_implementation_type
{
socket_type socket_;

socket_ops::state_type state_;

reactor::per_descriptor_data reactor_data_;
};

BOOST_ASIO_DECL reactive_socket_service_base(execution_context& context);

BOOST_ASIO_DECL void base_shutdown();

BOOST_ASIO_DECL void construct(base_implementation_type& impl);

BOOST_ASIO_DECL void base_move_construct(base_implementation_type& impl,
base_implementation_type& other_impl) BOOST_ASIO_NOEXCEPT;

BOOST_ASIO_DECL void base_move_assign(base_implementation_type& impl,
reactive_socket_service_base& other_service,
base_implementation_type& other_impl);

BOOST_ASIO_DECL void destroy(base_implementation_type& impl);

bool is_open(const base_implementation_type& impl) const
{
return impl.socket_ != invalid_socket;
}

BOOST_ASIO_DECL boost::system::error_code close(
base_implementation_type& impl, boost::system::error_code& ec);

BOOST_ASIO_DECL socket_type release(
base_implementation_type& impl, boost::system::error_code& ec);

native_handle_type native_handle(base_implementation_type& impl)
{
return impl.socket_;
}

BOOST_ASIO_DECL boost::system::error_code cancel(
base_implementation_type& impl, boost::system::error_code& ec);

bool at_mark(const base_implementation_type& impl,
boost::system::error_code& ec) const
{
return socket_ops::sockatmark(impl.socket_, ec);
}

std::size_t available(const base_implementation_type& impl,
boost::system::error_code& ec) const
{
return socket_ops::available(impl.socket_, ec);
}

boost::system::error_code listen(base_implementation_type& impl,
int backlog, boost::system::error_code& ec)
{
socket_ops::listen(impl.socket_, backlog, ec);
return ec;
}

template <typename IO_Control_Command>
boost::system::error_code io_control(base_implementation_type& impl,
IO_Control_Command& command, boost::system::error_code& ec)
{
socket_ops::ioctl(impl.socket_, impl.state_, command.name(),
static_cast<ioctl_arg_type*>(command.data()), ec);
return ec;
}

bool non_blocking(const base_implementation_type& impl) const
{
return (impl.state_ & socket_ops::user_set_non_blocking) != 0;
}

boost::system::error_code non_blocking(base_implementation_type& impl,
bool mode, boost::system::error_code& ec)
{
socket_ops::set_user_non_blocking(impl.socket_, impl.state_, mode, ec);
return ec;
}

bool native_non_blocking(const base_implementation_type& impl) const
{
return (impl.state_ & socket_ops::internal_non_blocking) != 0;
}

boost::system::error_code native_non_blocking(base_implementation_type& impl,
bool mode, boost::system::error_code& ec)
{
socket_ops::set_internal_non_blocking(impl.socket_, impl.state_, mode, ec);
return ec;
}

boost::system::error_code wait(base_implementation_type& impl,
socket_base::wait_type w, boost::system::error_code& ec)
{
switch (w)
{
case socket_base::wait_read:
socket_ops::poll_read(impl.socket_, impl.state_, -1, ec);
break;
case socket_base::wait_write:
socket_ops::poll_write(impl.socket_, impl.state_, -1, ec);
break;
case socket_base::wait_error:
socket_ops::poll_error(impl.socket_, impl.state_, -1, ec);
break;
default:
ec = boost::asio::error::invalid_argument;
break;
}

return ec;
}

template <typename Handler, typename IoExecutor>
void async_wait(base_implementation_type& impl,
socket_base::wait_type w, Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_wait_op<Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_wait"));

int op_type;
switch (w)
{
case socket_base::wait_read:
op_type = reactor::read_op;
break;
case socket_base::wait_write:
op_type = reactor::write_op;
break;
case socket_base::wait_error:
op_type = reactor::except_op;
break;
default:
p.p->ec_ = boost::asio::error::invalid_argument;
reactor_.post_immediate_completion(p.p, is_continuation);
p.v = p.p = 0;
return;
}

start_op(impl, op_type, p.p, is_continuation, false, false);
p.v = p.p = 0;
}

template <typename ConstBufferSequence>
size_t send(base_implementation_type& impl,
const ConstBufferSequence& buffers,
socket_base::message_flags flags, boost::system::error_code& ec)
{
typedef buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence> bufs_type;

if (bufs_type::is_single_buffer)
{
return socket_ops::sync_send1(impl.socket_,
impl.state_, bufs_type::first(buffers).data(),
bufs_type::first(buffers).size(), flags, ec);
}
else
{
bufs_type bufs(buffers);
return socket_ops::sync_send(impl.socket_, impl.state_,
bufs.buffers(), bufs.count(), flags, bufs.all_empty(), ec);
}
}

size_t send(base_implementation_type& impl, const null_buffers&,
socket_base::message_flags, boost::system::error_code& ec)
{
socket_ops::poll_write(impl.socket_, impl.state_, -1, ec);

return 0;
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_send(base_implementation_type& impl,
const ConstBufferSequence& buffers, socket_base::message_flags flags,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_socket_send_op<
ConstBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, impl.socket_,
impl.state_, buffers, flags, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_send"));

start_op(impl, reactor::write_op, p.p, is_continuation, true,
((impl.state_ & socket_ops::stream_oriented)
&& buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence>::all_empty(buffers)));
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_send(base_implementation_type& impl, const null_buffers&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_send(null_buffers)"));

start_op(impl, reactor::write_op, p.p, is_continuation, false, false);
p.v = p.p = 0;
}

template <typename MutableBufferSequence>
size_t receive(base_implementation_type& impl,
const MutableBufferSequence& buffers,
socket_base::message_flags flags, boost::system::error_code& ec)
{
typedef buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence> bufs_type;

if (bufs_type::is_single_buffer)
{
return socket_ops::sync_recv1(impl.socket_,
impl.state_, bufs_type::first(buffers).data(),
bufs_type::first(buffers).size(), flags, ec);
}
else
{
bufs_type bufs(buffers);
return socket_ops::sync_recv(impl.socket_, impl.state_,
bufs.buffers(), bufs.count(), flags, bufs.all_empty(), ec);
}
}

size_t receive(base_implementation_type& impl, const null_buffers&,
socket_base::message_flags, boost::system::error_code& ec)
{
socket_ops::poll_read(impl.socket_, impl.state_, -1, ec);

return 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_receive(base_implementation_type& impl,
const MutableBufferSequence& buffers, socket_base::message_flags flags,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_socket_recv_op<
MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, impl.socket_,
impl.state_, buffers, flags, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_receive"));

start_op(impl,
(flags & socket_base::message_out_of_band)
? reactor::except_op : reactor::read_op,
p.p, is_continuation,
(flags & socket_base::message_out_of_band) == 0,
((impl.state_ & socket_ops::stream_oriented)
&& buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence>::all_empty(buffers)));
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_receive(base_implementation_type& impl,
const null_buffers&, socket_base::message_flags flags,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_receive(null_buffers)"));

start_op(impl,
(flags & socket_base::message_out_of_band)
? reactor::except_op : reactor::read_op,
p.p, is_continuation, false, false);
p.v = p.p = 0;
}

template <typename MutableBufferSequence>
size_t receive_with_flags(base_implementation_type& impl,
const MutableBufferSequence& buffers,
socket_base::message_flags in_flags,
socket_base::message_flags& out_flags, boost::system::error_code& ec)
{
buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence> bufs(buffers);

return socket_ops::sync_recvmsg(impl.socket_, impl.state_,
bufs.buffers(), bufs.count(), in_flags, out_flags, ec);
}

size_t receive_with_flags(base_implementation_type& impl,
const null_buffers&, socket_base::message_flags,
socket_base::message_flags& out_flags, boost::system::error_code& ec)
{
socket_ops::poll_read(impl.socket_, impl.state_, -1, ec);

out_flags = 0;

return 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_receive_with_flags(base_implementation_type& impl,
const MutableBufferSequence& buffers, socket_base::message_flags in_flags,
socket_base::message_flags& out_flags, Handler& handler,
const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_socket_recvmsg_op<
MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, impl.socket_,
buffers, in_flags, out_flags, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_receive_with_flags"));

start_op(impl,
(in_flags & socket_base::message_out_of_band)
? reactor::except_op : reactor::read_op,
p.p, is_continuation,
(in_flags & socket_base::message_out_of_band) == 0, false);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_receive_with_flags(base_implementation_type& impl,
const null_buffers&, socket_base::message_flags in_flags,
socket_base::message_flags& out_flags, Handler& handler,
const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_receive_with_flags(null_buffers)"));

out_flags = 0;

start_op(impl,
(in_flags & socket_base::message_out_of_band)
? reactor::except_op : reactor::read_op,
p.p, is_continuation, false, false);
p.v = p.p = 0;
}

protected:
BOOST_ASIO_DECL boost::system::error_code do_open(
base_implementation_type& impl, int af,
int type, int protocol, boost::system::error_code& ec);

BOOST_ASIO_DECL boost::system::error_code do_assign(
base_implementation_type& impl, int type,
const native_handle_type& native_socket, boost::system::error_code& ec);

BOOST_ASIO_DECL void start_op(base_implementation_type& impl, int op_type,
reactor_op* op, bool is_continuation, bool is_non_blocking, bool noop);

BOOST_ASIO_DECL void start_accept_op(base_implementation_type& impl,
reactor_op* op, bool is_continuation, bool peer_is_open);

BOOST_ASIO_DECL void start_connect_op(base_implementation_type& impl,
reactor_op* op, bool is_continuation,
const socket_addr_type* addr, size_t addrlen);

reactor& reactor_;

const boost::system::error_code success_ec_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/reactive_socket_service_base.ipp>
#endif 

#endif 

#endif 
