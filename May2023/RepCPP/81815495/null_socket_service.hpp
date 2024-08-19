
#ifndef ASIO_DETAIL_NULL_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_NULL_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)

#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/post.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/bind_handler.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Protocol>
class null_socket_service :
public execution_context_service_base<null_socket_service<Protocol> >
{
public:
typedef Protocol protocol_type;

typedef typename Protocol::endpoint endpoint_type;

typedef int native_handle_type;

struct implementation_type
{
};

null_socket_service(execution_context& context)
: execution_context_service_base<null_socket_service<Protocol> >(context)
{
}

void shutdown()
{
}

void construct(implementation_type&)
{
}

void move_construct(implementation_type&, implementation_type&)
{
}

void move_assign(implementation_type&,
null_socket_service&, implementation_type&)
{
}

template <typename Protocol1>
void converting_move_construct(implementation_type&,
null_socket_service<Protocol1>&,
typename null_socket_service<Protocol1>::implementation_type&)
{
}

void destroy(implementation_type&)
{
}

asio::error_code open(implementation_type&,
const protocol_type&, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

asio::error_code assign(implementation_type&, const protocol_type&,
const native_handle_type&, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

bool is_open(const implementation_type&) const
{
return false;
}

asio::error_code close(implementation_type&,
asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

native_handle_type release(implementation_type&,
asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

native_handle_type native_handle(implementation_type&)
{
return 0;
}

asio::error_code cancel(implementation_type&,
asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

bool at_mark(const implementation_type&,
asio::error_code& ec) const
{
ec = asio::error::operation_not_supported;
return false;
}

std::size_t available(const implementation_type&,
asio::error_code& ec) const
{
ec = asio::error::operation_not_supported;
return 0;
}

asio::error_code listen(implementation_type&,
int, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

template <typename IO_Control_Command>
asio::error_code io_control(implementation_type&,
IO_Control_Command&, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

bool non_blocking(const implementation_type&) const
{
return false;
}

asio::error_code non_blocking(implementation_type&,
bool, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

bool native_non_blocking(const implementation_type&) const
{
return false;
}

asio::error_code native_non_blocking(implementation_type&,
bool, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

asio::error_code shutdown(implementation_type&,
socket_base::shutdown_type, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

asio::error_code bind(implementation_type&,
const endpoint_type&, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

template <typename Option>
asio::error_code set_option(implementation_type&,
const Option&, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

template <typename Option>
asio::error_code get_option(const implementation_type&,
Option&, asio::error_code& ec) const
{
ec = asio::error::operation_not_supported;
return ec;
}

endpoint_type local_endpoint(const implementation_type&,
asio::error_code& ec) const
{
ec = asio::error::operation_not_supported;
return endpoint_type();
}

endpoint_type remote_endpoint(const implementation_type&,
asio::error_code& ec) const
{
ec = asio::error::operation_not_supported;
return endpoint_type();
}

template <typename ConstBufferSequence>
std::size_t send(implementation_type&, const ConstBufferSequence&,
socket_base::message_flags, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

std::size_t send(implementation_type&, const null_buffers&,
socket_base::message_flags, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_send(implementation_type&, const ConstBufferSequence&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename Handler, typename IoExecutor>
void async_send(implementation_type&, const null_buffers&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename MutableBufferSequence>
std::size_t receive(implementation_type&, const MutableBufferSequence&,
socket_base::message_flags, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

std::size_t receive(implementation_type&, const null_buffers&,
socket_base::message_flags, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_receive(implementation_type&, const MutableBufferSequence&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename Handler, typename IoExecutor>
void async_receive(implementation_type&, const null_buffers&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename MutableBufferSequence>
std::size_t receive_with_flags(implementation_type&,
const MutableBufferSequence&, socket_base::message_flags,
socket_base::message_flags&, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

std::size_t receive_with_flags(implementation_type&,
const null_buffers&, socket_base::message_flags,
socket_base::message_flags&, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_receive_with_flags(implementation_type&,
const MutableBufferSequence&, socket_base::message_flags,
socket_base::message_flags&, Handler& handler, const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename Handler, typename IoExecutor>
void async_receive_with_flags(implementation_type&, const null_buffers&,
socket_base::message_flags, socket_base::message_flags&,
Handler& handler, const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename ConstBufferSequence>
std::size_t send_to(implementation_type&, const ConstBufferSequence&,
const endpoint_type&, socket_base::message_flags,
asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

std::size_t send_to(implementation_type&, const null_buffers&,
const endpoint_type&, socket_base::message_flags,
asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_send_to(implementation_type&, const ConstBufferSequence&,
const endpoint_type&, socket_base::message_flags,
Handler& handler)
{
asio::error_code ec = asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename Handler, typename IoExecutor>
void async_send_to(implementation_type&, const null_buffers&,
const endpoint_type&, socket_base::message_flags,
Handler& handler, const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename MutableBufferSequence>
std::size_t receive_from(implementation_type&, const MutableBufferSequence&,
endpoint_type&, socket_base::message_flags,
asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

std::size_t receive_from(implementation_type&, const null_buffers&,
endpoint_type&, socket_base::message_flags,
asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_receive_from(implementation_type&, const MutableBufferSequence&,
endpoint_type&, socket_base::message_flags, Handler& handler,
const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename Handler, typename IoExecutor>
void async_receive_from(implementation_type&, const null_buffers&,
endpoint_type&, socket_base::message_flags, Handler& handler,
const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename Socket>
asio::error_code accept(implementation_type&,
Socket&, endpoint_type*, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

template <typename Socket, typename Handler, typename IoExecutor>
void async_accept(implementation_type&, Socket&, endpoint_type*,
Handler& handler, const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
asio::post(io_ex, detail::bind_handler(handler, ec));
}

asio::error_code connect(implementation_type&,
const endpoint_type&, asio::error_code& ec)
{
ec = asio::error::operation_not_supported;
return ec;
}

template <typename Handler, typename IoExecutor>
void async_connect(implementation_type&, const endpoint_type&,
Handler& handler, const IoExecutor& io_ex)
{
asio::error_code ec = asio::error::operation_not_supported;
asio::post(io_ex, detail::bind_handler(handler, ec));
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
