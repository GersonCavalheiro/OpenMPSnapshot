
#ifndef ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_BASE_HPP
#define ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/associated_cancellation_slot.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/operation.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/win_iocp_io_context.hpp"
#include "asio/detail/win_iocp_null_buffers_op.hpp"
#include "asio/detail/win_iocp_socket_connect_op.hpp"
#include "asio/detail/win_iocp_socket_send_op.hpp"
#include "asio/detail/win_iocp_socket_recv_op.hpp"
#include "asio/detail/win_iocp_socket_recvmsg_op.hpp"
#include "asio/detail/win_iocp_wait_op.hpp"

#include "asio/detail/push_options.hpp"

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

#if defined(ASIO_ENABLE_CANCELIO)
DWORD safe_cancellation_thread_id_;
#endif 

base_implementation_type* next_;
base_implementation_type* prev_;
};

ASIO_DECL win_iocp_socket_service_base(execution_context& context);

ASIO_DECL void base_shutdown();

ASIO_DECL void construct(base_implementation_type& impl);

ASIO_DECL void base_move_construct(base_implementation_type& impl,
base_implementation_type& other_impl) ASIO_NOEXCEPT;

ASIO_DECL void base_move_assign(base_implementation_type& impl,
win_iocp_socket_service_base& other_service,
base_implementation_type& other_impl);

ASIO_DECL void destroy(base_implementation_type& impl);

bool is_open(const base_implementation_type& impl) const
{
return impl.socket_ != invalid_socket;
}

ASIO_DECL asio::error_code close(
base_implementation_type& impl, asio::error_code& ec);

ASIO_DECL socket_type release(
base_implementation_type& impl, asio::error_code& ec);

ASIO_DECL asio::error_code cancel(
base_implementation_type& impl, asio::error_code& ec);

bool at_mark(const base_implementation_type& impl,
asio::error_code& ec) const
{
return socket_ops::sockatmark(impl.socket_, ec);
}

std::size_t available(const base_implementation_type& impl,
asio::error_code& ec) const
{
return socket_ops::available(impl.socket_, ec);
}

asio::error_code listen(base_implementation_type& impl,
int backlog, asio::error_code& ec)
{
socket_ops::listen(impl.socket_, backlog, ec);
return ec;
}

template <typename IO_Control_Command>
asio::error_code io_control(base_implementation_type& impl,
IO_Control_Command& command, asio::error_code& ec)
{
socket_ops::ioctl(impl.socket_, impl.state_, command.name(),
static_cast<ioctl_arg_type*>(command.data()), ec);
return ec;
}

bool non_blocking(const base_implementation_type& impl) const
{
return (impl.state_ & socket_ops::user_set_non_blocking) != 0;
}

asio::error_code non_blocking(base_implementation_type& impl,
bool mode, asio::error_code& ec)
{
socket_ops::set_user_non_blocking(impl.socket_, impl.state_, mode, ec);
return ec;
}

bool native_non_blocking(const base_implementation_type& impl) const
{
return (impl.state_ & socket_ops::internal_non_blocking) != 0;
}

asio::error_code native_non_blocking(base_implementation_type& impl,
bool mode, asio::error_code& ec)
{
socket_ops::set_internal_non_blocking(impl.socket_, impl.state_, mode, ec);
return ec;
}

asio::error_code wait(base_implementation_type& impl,
socket_base::wait_type w, asio::error_code& ec)
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
ec = asio::error::invalid_argument;
break;
}

return ec;
}

template <typename Handler, typename IoExecutor>
void async_wait(base_implementation_type& impl,
socket_base::wait_type w, Handler& handler, const IoExecutor& io_ex)
{
typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

bool is_continuation =
asio_handler_cont_helpers::is_continuation(handler);

typedef win_iocp_wait_op<Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl.cancel_token_, handler, io_ex);

ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_wait"));

operation* iocp_op = p.p;
if (slot.is_connected())
{
p.p->cancellation_key_ = iocp_op =
&slot.template emplace<reactor_op_cancellation>(
impl.socket_, iocp_op);
}

int op_type = -1;
switch (w)
{
case socket_base::wait_read:
op_type = start_null_buffers_receive_op(impl, 0, p.p, iocp_op);
break;
case socket_base::wait_write:
op_type = select_reactor::write_op;
start_reactor_op(impl, select_reactor::write_op, p.p);
break;
case socket_base::wait_error:
op_type = select_reactor::read_op;
start_reactor_op(impl, select_reactor::except_op, p.p);
break;
default:
p.p->ec_ = asio::error::invalid_argument;
iocp_service_.post_immediate_completion(p.p, is_continuation);
break;
}

p.v = p.p = 0;

if (slot.is_connected() && op_type != -1)
{
static_cast<reactor_op_cancellation*>(iocp_op)->use_reactor(
&get_reactor(), &impl.reactor_data_, op_type);
}
}

template <typename ConstBufferSequence>
size_t send(base_implementation_type& impl,
const ConstBufferSequence& buffers,
socket_base::message_flags flags, asio::error_code& ec)
{
buffer_sequence_adapter<asio::const_buffer,
ConstBufferSequence> bufs(buffers);

return socket_ops::sync_send(impl.socket_, impl.state_,
bufs.buffers(), bufs.count(), flags, bufs.all_empty(), ec);
}

size_t send(base_implementation_type& impl, const null_buffers&,
socket_base::message_flags, asio::error_code& ec)
{
socket_ops::poll_write(impl.socket_, impl.state_, -1, ec);

return 0;
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_send(base_implementation_type& impl,
const ConstBufferSequence& buffers, socket_base::message_flags flags,
Handler& handler, const IoExecutor& io_ex)
{
typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef win_iocp_socket_send_op<
ConstBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
operation* o = p.p = new (p.v) op(
impl.cancel_token_, buffers, handler, io_ex);

ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_send"));

buffer_sequence_adapter<asio::const_buffer,
ConstBufferSequence> bufs(buffers);

if (slot.is_connected())
o = &slot.template emplace<iocp_op_cancellation>(impl.socket_, o);

start_send_op(impl, bufs.buffers(), bufs.count(), flags,
(impl.state_ & socket_ops::stream_oriented) != 0 && bufs.all_empty(),
o);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_send(base_implementation_type& impl, const null_buffers&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
typedef win_iocp_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl.cancel_token_, handler, io_ex);

ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_send(null_buffers)"));

start_reactor_op(impl, select_reactor::write_op, p.p);
p.v = p.p = 0;
}

template <typename MutableBufferSequence>
size_t receive(base_implementation_type& impl,
const MutableBufferSequence& buffers,
socket_base::message_flags flags, asio::error_code& ec)
{
buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence> bufs(buffers);

return socket_ops::sync_recv(impl.socket_, impl.state_,
bufs.buffers(), bufs.count(), flags, bufs.all_empty(), ec);
}

size_t receive(base_implementation_type& impl, const null_buffers&,
socket_base::message_flags, asio::error_code& ec)
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
typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef win_iocp_socket_recv_op<
MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
operation* o = p.p = new (p.v) op(impl.state_,
impl.cancel_token_, buffers, handler, io_ex);

ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_receive"));

buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence> bufs(buffers);

if (slot.is_connected())
o = &slot.template emplace<iocp_op_cancellation>(impl.socket_, o);

start_receive_op(impl, bufs.buffers(), bufs.count(), flags,
(impl.state_ & socket_ops::stream_oriented) != 0 && bufs.all_empty(),
o);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_receive(base_implementation_type& impl,
const null_buffers&, socket_base::message_flags flags,
Handler& handler, const IoExecutor& io_ex)
{
typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef win_iocp_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl.cancel_token_, handler, io_ex);

ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_receive(null_buffers)"));

operation* iocp_op = p.p;
if (slot.is_connected())
{
p.p->cancellation_key_ = iocp_op =
&slot.template emplace<reactor_op_cancellation>(
impl.socket_, iocp_op);
}

int op_type = start_null_buffers_receive_op(impl, flags, p.p, iocp_op);
p.v = p.p = 0;

if (slot.is_connected() && op_type != -1)
{
static_cast<reactor_op_cancellation*>(iocp_op)->use_reactor(
&get_reactor(), &impl.reactor_data_, op_type);
}
}

template <typename MutableBufferSequence>
size_t receive_with_flags(base_implementation_type& impl,
const MutableBufferSequence& buffers,
socket_base::message_flags in_flags,
socket_base::message_flags& out_flags, asio::error_code& ec)
{
buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence> bufs(buffers);

return socket_ops::sync_recvmsg(impl.socket_, impl.state_,
bufs.buffers(), bufs.count(), in_flags, out_flags, ec);
}

size_t receive_with_flags(base_implementation_type& impl,
const null_buffers&, socket_base::message_flags,
socket_base::message_flags& out_flags, asio::error_code& ec)
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
typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef win_iocp_socket_recvmsg_op<
MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
operation* o = p.p = new (p.v) op(impl.cancel_token_,
buffers, out_flags, handler, io_ex);

ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_receive_with_flags"));

buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence> bufs(buffers);

if (slot.is_connected())
o = &slot.template emplace<iocp_op_cancellation>(impl.socket_, o);

start_receive_op(impl, bufs.buffers(), bufs.count(), in_flags, false, o);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_receive_with_flags(base_implementation_type& impl,
const null_buffers&, socket_base::message_flags in_flags,
socket_base::message_flags& out_flags, Handler& handler,
const IoExecutor& io_ex)
{
typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef win_iocp_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl.cancel_token_, handler, io_ex);

ASIO_HANDLER_CREATION((context_, *p.p, "socket",
&impl, impl.socket_, "async_receive_with_flags(null_buffers)"));

out_flags = 0;

operation* iocp_op = p.p;
if (slot.is_connected())
{
p.p->cancellation_key_ = iocp_op =
&slot.template emplace<reactor_op_cancellation>(
impl.socket_, iocp_op);
}

int op_type = start_null_buffers_receive_op(impl, in_flags, p.p, iocp_op);
p.v = p.p = 0;

if (slot.is_connected() && op_type != -1)
{
static_cast<reactor_op_cancellation*>(iocp_op)->use_reactor(
&get_reactor(), &impl.reactor_data_, op_type);
}
}

ASIO_DECL void restart_accept_op(socket_type s,
socket_holder& new_socket, int family, int type,
int protocol, void* output_buffer, DWORD address_length,
long* cancel_requested, operation* op);

protected:
ASIO_DECL asio::error_code do_open(
base_implementation_type& impl, int family, int type,
int protocol, asio::error_code& ec);

ASIO_DECL asio::error_code do_assign(
base_implementation_type& impl, int type,
socket_type native_socket, asio::error_code& ec);

ASIO_DECL void start_send_op(base_implementation_type& impl,
WSABUF* buffers, std::size_t buffer_count,
socket_base::message_flags flags, bool noop, operation* op);

ASIO_DECL void start_send_to_op(base_implementation_type& impl,
WSABUF* buffers, std::size_t buffer_count,
const socket_addr_type* addr, int addrlen,
socket_base::message_flags flags, operation* op);

ASIO_DECL void start_receive_op(base_implementation_type& impl,
WSABUF* buffers, std::size_t buffer_count,
socket_base::message_flags flags, bool noop, operation* op);

ASIO_DECL int start_null_buffers_receive_op(
base_implementation_type& impl, socket_base::message_flags flags,
reactor_op* op, operation* iocp_op);

ASIO_DECL void start_receive_from_op(base_implementation_type& impl,
WSABUF* buffers, std::size_t buffer_count, socket_addr_type* addr,
socket_base::message_flags flags, int* addrlen, operation* op);

ASIO_DECL void start_accept_op(base_implementation_type& impl,
bool peer_is_open, socket_holder& new_socket, int family, int type,
int protocol, void* output_buffer, DWORD address_length, operation* op);

ASIO_DECL void start_reactor_op(base_implementation_type& impl,
int op_type, reactor_op* op);

ASIO_DECL int start_connect_op(base_implementation_type& impl,
int family, int type, const socket_addr_type* remote_addr,
std::size_t remote_addrlen, win_iocp_socket_connect_op_base* op,
operation* iocp_op);

ASIO_DECL void close_for_destruction(base_implementation_type& impl);

ASIO_DECL void update_cancellation_thread_id(
base_implementation_type& impl);

ASIO_DECL select_reactor& get_reactor();

typedef BOOL (PASCAL *connect_ex_fn)(SOCKET,
const socket_addr_type*, int, void*, DWORD, DWORD*, OVERLAPPED*);

ASIO_DECL connect_ex_fn get_connect_ex(
base_implementation_type& impl, int type);

typedef LONG (NTAPI *nt_set_info_fn)(HANDLE, ULONG_PTR*, void*, ULONG, ULONG);

ASIO_DECL nt_set_info_fn get_nt_set_info();

ASIO_DECL void* interlocked_compare_exchange_pointer(
void** dest, void* exch, void* cmp);

ASIO_DECL void* interlocked_exchange_pointer(void** dest, void* val);

class iocp_op_cancellation : public operation
{
public:
iocp_op_cancellation(SOCKET s, operation* target)
: operation(&iocp_op_cancellation::do_complete),
socket_(s),
target_(target)
{
}

static void do_complete(void* owner, operation* base,
const asio::error_code& result_ec,
std::size_t bytes_transferred)
{
iocp_op_cancellation* o = static_cast<iocp_op_cancellation*>(base);
o->target_->complete(owner, result_ec, bytes_transferred);
}

void operator()(cancellation_type_t type)
{
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0600)
if (!!(type &
(cancellation_type::terminal
| cancellation_type::partial
| cancellation_type::total)))
{
HANDLE sock_as_handle = reinterpret_cast<HANDLE>(socket_);
::CancelIoEx(sock_as_handle, this);
}
#else 
(void)type;
#endif 
}

private:
SOCKET socket_;
operation* target_;
};

class accept_op_cancellation : public operation
{
public:
accept_op_cancellation(SOCKET s, operation* target)
: operation(&iocp_op_cancellation::do_complete),
socket_(s),
target_(target),
cancel_requested_(0)
{
}

static void do_complete(void* owner, operation* base,
const asio::error_code& result_ec,
std::size_t bytes_transferred)
{
accept_op_cancellation* o = static_cast<accept_op_cancellation*>(base);
o->target_->complete(owner, result_ec, bytes_transferred);
}

long* get_cancel_requested()
{
return &cancel_requested_;
}

void operator()(cancellation_type_t type)
{
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0600)
if (!!(type &
(cancellation_type::terminal
| cancellation_type::partial
| cancellation_type::total)))
{
HANDLE sock_as_handle = reinterpret_cast<HANDLE>(socket_);
::CancelIoEx(sock_as_handle, this);
}
#else 
(void)type;
#endif 
}

private:
SOCKET socket_;
operation* target_;
long cancel_requested_;
};

class reactor_op_cancellation : public operation
{
public:
reactor_op_cancellation(SOCKET s, operation* base)
: operation(&reactor_op_cancellation::do_complete),
socket_(s),
target_(base),
reactor_(0),
reactor_data_(0),
op_type_(-1)
{
}

void use_reactor(select_reactor* r,
select_reactor::per_descriptor_data* p, int o)
{
reactor_ = r;
reactor_data_ = p;
op_type_ = o;
}

static void do_complete(void* owner, operation* base,
const asio::error_code& result_ec,
std::size_t bytes_transferred)
{
reactor_op_cancellation* o = static_cast<reactor_op_cancellation*>(base);
o->target_->complete(owner, result_ec, bytes_transferred);
}

void operator()(cancellation_type_t type)
{
if (!!(type &
(cancellation_type::terminal
| cancellation_type::partial
| cancellation_type::total)))
{
if (reactor_)
{
reactor_->cancel_ops_by_key(socket_,
*reactor_data_, op_type_, this);
}
else
{
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0600)
HANDLE sock_as_handle = reinterpret_cast<HANDLE>(socket_);
::CancelIoEx(sock_as_handle, this);
#endif 
}
}
}

private:
SOCKET socket_;
operation* target_;
select_reactor* reactor_;
select_reactor::per_descriptor_data* reactor_data_;
int op_type_;
};

execution_context& context_;

win_iocp_io_context& iocp_service_;

select_reactor* reactor_;

void* connect_ex_;

void* nt_set_info_;

asio::detail::mutex mutex_;

base_implementation_type* impl_list_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/win_iocp_socket_service_base.ipp"
#endif 

#endif 

#endif 
