
#ifndef BOOST_ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_BASE_HPP
#define BOOST_ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_IOCP)

#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/socket_base.hpp>
#include <boost/asio/detail/bind_handler.hpp>
#include <boost/asio/detail/buffer_sequence_adapter.hpp>
#include <boost/asio/detail/fenced_block.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/handler_invoke_helpers.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/mutex.hpp>
#include <boost/asio/detail/operation.hpp>
#include <boost/asio/detail/reactor_op.hpp>
#include <boost/asio/detail/select_reactor.hpp>
#include <boost/asio/detail/socket_holder.hpp>
#include <boost/asio/detail/socket_ops.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/detail/win_iocp_io_context.hpp>
#include <boost/asio/detail/win_iocp_null_buffers_op.hpp>
#include <boost/asio/detail/win_iocp_socket_connect_op.hpp>
#include <boost/asio/detail/win_iocp_socket_send_op.hpp>
#include <boost/asio/detail/win_iocp_socket_recv_op.hpp>
#include <boost/asio/detail/win_iocp_socket_recvmsg_op.hpp>
#include <boost/asio/detail/win_iocp_wait_op.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class win_iocp_socket_service_base
{
public:
struct base_implementation_type
{
socket_type socket_;

socket_ops::state_type state_;

socket_ops::shared_cancel_token_type cancel_token_;

select_reactor::per_descriptor_data reactor_data_;

#if defined(BOOST_ASIO_ENABLE_CANCELIO)
DWORD safe_cancellation_thread_id_;
#endif 

base_implementation_type* next_;
base_implementation_type* prev_;
};

BOOST_ASIO_DECL win_iocp_socket_service_base(execution_context& context);

BOOST_ASIO_DECL void base_shutdown();

BOOST_ASIO_DECL void construct(base_implementation_type& impl);

BOOST_ASIO_DECL void base_move_construct(base_implementation_type& impl,
base_implementation_type& other_impl) BOOST_ASIO_NOEXCEPT;

BOOST_ASIO_DECL void base_move_assign(base_implementation_type& impl,
win_iocp_socket_service_base& other_service,
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

typedef win_iocp_wait_op<Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl.cancel_token_, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_wait"));

switch (w)
{
case socket_base::wait_read:
start_null_buffers_receive_op(impl, 0, p.p);
break;
case socket_base::wait_write:
start_reactor_op(impl, select_reactor::write_op, p.p);
break;
case socket_base::wait_error:
start_reactor_op(impl, select_reactor::except_op, p.p);
break;
default:
p.p->ec_ = boost::asio::error::invalid_argument;
iocp_service_.post_immediate_completion(p.p, is_continuation);
break;
}

p.v = p.p = 0;
}

template <typename ConstBufferSequence>
size_t send(base_implementation_type& impl,
const ConstBufferSequence& buffers,
socket_base::message_flags flags, boost::system::error_code& ec)
{
buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence> bufs(buffers);

return socket_ops::sync_send(impl.socket_, impl.state_,
bufs.buffers(), bufs.count(), flags, bufs.all_empty(), ec);
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
typedef win_iocp_socket_send_op<
ConstBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl.cancel_token_, buffers, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_send"));

buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence> bufs(buffers);

start_send_op(impl, bufs.buffers(), bufs.count(), flags,
(impl.state_ & socket_ops::stream_oriented) != 0 && bufs.all_empty(),
p.p);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_send(base_implementation_type& impl, const null_buffers&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
typedef win_iocp_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl.cancel_token_, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_send(null_buffers)"));

start_reactor_op(impl, select_reactor::write_op, p.p);
p.v = p.p = 0;
}

template <typename MutableBufferSequence>
size_t receive(base_implementation_type& impl,
const MutableBufferSequence& buffers,
socket_base::message_flags flags, boost::system::error_code& ec)
{
buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence> bufs(buffers);

return socket_ops::sync_recv(impl.socket_, impl.state_,
bufs.buffers(), bufs.count(), flags, bufs.all_empty(), ec);
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
typedef win_iocp_socket_recv_op<
MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl.state_, impl.cancel_token_,
buffers, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_receive"));

buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence> bufs(buffers);

start_receive_op(impl, bufs.buffers(), bufs.count(), flags,
(impl.state_ & socket_ops::stream_oriented) != 0 && bufs.all_empty(),
p.p);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_receive(base_implementation_type& impl,
const null_buffers&, socket_base::message_flags flags,
Handler& handler, const IoExecutor& io_ex)
{
typedef win_iocp_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl.cancel_token_, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_receive(null_buffers)"));

start_null_buffers_receive_op(impl, flags, p.p);
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
typedef win_iocp_socket_recvmsg_op<
MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl.cancel_token_,
buffers, out_flags, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_receive_with_flags"));

buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence> bufs(buffers);

start_receive_op(impl, bufs.buffers(), bufs.count(), in_flags, false, p.p);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_receive_with_flags(base_implementation_type& impl,
const null_buffers&, socket_base::message_flags in_flags,
socket_base::message_flags& out_flags, Handler& handler,
const IoExecutor& io_ex)
{
typedef win_iocp_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl.cancel_token_, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_receive_with_flags(null_buffers)"));

out_flags = 0;

start_null_buffers_receive_op(impl, in_flags, p.p);
p.v = p.p = 0;
}

BOOST_ASIO_DECL void restart_accept_op(socket_type s,
socket_holder& new_socket, int family, int type, int protocol,
void* output_buffer, DWORD address_length, operation* op);

protected:
BOOST_ASIO_DECL boost::system::error_code do_open(
base_implementation_type& impl, int family, int type,
int protocol, boost::system::error_code& ec);

BOOST_ASIO_DECL boost::system::error_code do_assign(
base_implementation_type& impl, int type,
socket_type native_socket, boost::system::error_code& ec);

BOOST_ASIO_DECL void start_send_op(base_implementation_type& impl,
WSABUF* buffers, std::size_t buffer_count,
socket_base::message_flags flags, bool noop, operation* op);

BOOST_ASIO_DECL void start_send_to_op(base_implementation_type& impl,
WSABUF* buffers, std::size_t buffer_count,
const socket_addr_type* addr, int addrlen,
socket_base::message_flags flags, operation* op);

BOOST_ASIO_DECL void start_receive_op(base_implementation_type& impl,
WSABUF* buffers, std::size_t buffer_count,
socket_base::message_flags flags, bool noop, operation* op);

BOOST_ASIO_DECL void start_null_buffers_receive_op(
base_implementation_type& impl,
socket_base::message_flags flags, reactor_op* op);

BOOST_ASIO_DECL void start_receive_from_op(base_implementation_type& impl,
WSABUF* buffers, std::size_t buffer_count, socket_addr_type* addr,
socket_base::message_flags flags, int* addrlen, operation* op);

BOOST_ASIO_DECL void start_accept_op(base_implementation_type& impl,
bool peer_is_open, socket_holder& new_socket, int family, int type,
int protocol, void* output_buffer, DWORD address_length, operation* op);

BOOST_ASIO_DECL void start_reactor_op(base_implementation_type& impl,
int op_type, reactor_op* op);

BOOST_ASIO_DECL void start_connect_op(base_implementation_type& impl,
int family, int type, const socket_addr_type* remote_addr,
std::size_t remote_addrlen, win_iocp_socket_connect_op_base* op);

BOOST_ASIO_DECL void close_for_destruction(base_implementation_type& impl);

BOOST_ASIO_DECL void update_cancellation_thread_id(
base_implementation_type& impl);

BOOST_ASIO_DECL select_reactor& get_reactor();

typedef BOOL (PASCAL *connect_ex_fn)(SOCKET,
const socket_addr_type*, int, void*, DWORD, DWORD*, OVERLAPPED*);

BOOST_ASIO_DECL connect_ex_fn get_connect_ex(
base_implementation_type& impl, int type);

typedef LONG (NTAPI *nt_set_info_fn)(HANDLE, ULONG_PTR*, void*, ULONG, ULONG);

BOOST_ASIO_DECL nt_set_info_fn get_nt_set_info();

BOOST_ASIO_DECL void* interlocked_compare_exchange_pointer(
void** dest, void* exch, void* cmp);

BOOST_ASIO_DECL void* interlocked_exchange_pointer(void** dest, void* val);

execution_context& context_;

win_iocp_io_context& iocp_service_;

select_reactor* reactor_;

void* connect_ex_;

void* nt_set_info_;

boost::asio::detail::mutex mutex_;

base_implementation_type* impl_list_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/win_iocp_socket_service_base.ipp>
#endif 

#endif 

#endif 
