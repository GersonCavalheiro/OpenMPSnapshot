
#ifndef ASIO_DETAIL_WINRT_SSOCKET_SERVICE_BASE_HPP
#define ASIO_DETAIL_WINRT_SSOCKET_SERVICE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)

#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/winrt_async_manager.hpp"
#include "asio/detail/winrt_socket_recv_op.hpp"
#include "asio/detail/winrt_socket_send_op.hpp"

#if defined(ASIO_HAS_IOCP)
# include "asio/detail/win_iocp_io_context.hpp"
#else 
# include "asio/detail/scheduler.hpp"
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class winrt_ssocket_service_base
{
public:
typedef Windows::Networking::Sockets::StreamSocket^ native_handle_type;

struct base_implementation_type
{
base_implementation_type()
: socket_(nullptr),
next_(0),
prev_(0)
{
}

native_handle_type socket_;

base_implementation_type* next_;
base_implementation_type* prev_;
};

ASIO_DECL winrt_ssocket_service_base(execution_context& context);

ASIO_DECL void base_shutdown();

ASIO_DECL void construct(base_implementation_type&);

ASIO_DECL void base_move_construct(base_implementation_type& impl,
base_implementation_type& other_impl) ASIO_NOEXCEPT;

ASIO_DECL void base_move_assign(base_implementation_type& impl,
winrt_ssocket_service_base& other_service,
base_implementation_type& other_impl);

ASIO_DECL void destroy(base_implementation_type& impl);

bool is_open(const base_implementation_type& impl) const
{
return impl.socket_ != nullptr;
}

ASIO_DECL asio::error_code close(
base_implementation_type& impl, asio::error_code& ec);

ASIO_DECL native_handle_type release(
base_implementation_type& impl, asio::error_code& ec);

native_handle_type native_handle(base_implementation_type& impl)
{
return impl.socket_;
}

asio::error_code cancel(base_implementation_type&,
asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

bool at_mark(const base_implementation_type&,
asio::error_code& ec) const
{
ec = asio::error::operation_not_supported;
return false;
}

std::size_t available(const base_implementation_type&,
asio::error_code& ec) const
{
ec = asio::error::operation_not_supported;
return 0;
}

template <typename IO_Control_Command>
asio::error_code io_control(base_implementation_type&,
IO_Control_Command&, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

bool non_blocking(const base_implementation_type&) const
{
return false;
}

asio::error_code non_blocking(base_implementation_type&,
bool, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

bool native_non_blocking(const base_implementation_type&) const
{
return false;
}

asio::error_code native_non_blocking(base_implementation_type&,
bool, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

template <typename ConstBufferSequence>
std::size_t send(base_implementation_type& impl,
const ConstBufferSequence& buffers,
socket_base::message_flags flags, asio::error_code& ec)
{
return do_send(impl,
buffer_sequence_adapter<asio::const_buffer,
ConstBufferSequence>::first(buffers), flags, ec);
}

std::size_t send(base_implementation_type&, const null_buffers&,
socket_base::message_flags, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_send(base_implementation_type& impl,
const ConstBufferSequence& buffers, socket_base::message_flags flags,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
asio_handler_cont_helpers::is_continuation(handler);

typedef winrt_socket_send_op<ConstBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(buffers, handler, io_ex);

ASIO_HANDLER_CREATION((scheduler_.context(),
*p.p, "socket", &impl, 0, "async_send"));

start_send_op(impl,
buffer_sequence_adapter<asio::const_buffer,
ConstBufferSequence>::first(buffers),
flags, p.p, is_continuation);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_send(base_implementation_type&, const null_buffers&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
asio::post(io_ex,
detail::bind_handler(handler, ec, bytes_transferred));
}

template <typename MutableBufferSequence>
std::size_t receive(base_implementation_type& impl,
const MutableBufferSequence& buffers,
socket_base::message_flags flags, asio::error_code& ec)
{
return do_receive(impl,
buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence>::first(buffers), flags, ec);
}

std::size_t receive(base_implementation_type&, const null_buffers&,
socket_base::message_flags, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_receive(base_implementation_type& impl,
const MutableBufferSequence& buffers, socket_base::message_flags flags,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
asio_handler_cont_helpers::is_continuation(handler);

typedef winrt_socket_recv_op<MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(buffers, handler, io_ex);

ASIO_HANDLER_CREATION((scheduler_.context(),
*p.p, "socket", &impl, 0, "async_receive"));

start_receive_op(impl,
buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence>::first(buffers),
flags, p.p, is_continuation);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_receive(base_implementation_type&, const null_buffers&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
asio::post(io_ex,
detail::bind_handler(handler, ec, bytes_transferred));
}

protected:
ASIO_DECL std::size_t do_get_endpoint(
const base_implementation_type& impl, bool local,
void* addr, std::size_t addr_len, asio::error_code& ec) const;

ASIO_DECL asio::error_code do_set_option(
base_implementation_type& impl,
int level, int optname, const void* optval,
std::size_t optlen, asio::error_code& ec);

ASIO_DECL void do_get_option(
const base_implementation_type& impl,
int level, int optname, void* optval,
std::size_t* optlen, asio::error_code& ec) const;

ASIO_DECL asio::error_code do_connect(
base_implementation_type& impl,
const void* addr, asio::error_code& ec);

ASIO_DECL void start_connect_op(
base_implementation_type& impl, const void* addr,
winrt_async_op<void>* op, bool is_continuation);

ASIO_DECL std::size_t do_send(
base_implementation_type& impl, const asio::const_buffer& data,
socket_base::message_flags flags, asio::error_code& ec);

ASIO_DECL void start_send_op(base_implementation_type& impl,
const asio::const_buffer& data, socket_base::message_flags flags,
winrt_async_op<unsigned int>* op, bool is_continuation);

ASIO_DECL std::size_t do_receive(
base_implementation_type& impl, const asio::mutable_buffer& data,
socket_base::message_flags flags, asio::error_code& ec);

ASIO_DECL void start_receive_op(base_implementation_type& impl,
const asio::mutable_buffer& data, socket_base::message_flags flags,
winrt_async_op<Windows::Storage::Streams::IBuffer^>* op,
bool is_continuation);

#if defined(ASIO_HAS_IOCP)
typedef class win_iocp_io_context scheduler_impl;
#else
typedef class scheduler scheduler_impl;
#endif
scheduler_impl& scheduler_;

winrt_async_manager& async_manager_;

asio::detail::mutex mutex_;

base_implementation_type* impl_list_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/winrt_ssocket_service_base.ipp"
#endif 

#endif 

#endif 
