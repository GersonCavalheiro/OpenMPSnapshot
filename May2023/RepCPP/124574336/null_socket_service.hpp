
#ifndef BOOST_ASIO_DETAIL_NULL_SOCKET_SERVICE_HPP
#define BOOST_ASIO_DETAIL_NULL_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_WINDOWS_RUNTIME)

#include <boost/asio/buffer.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/socket_base.hpp>
#include <boost/asio/detail/bind_handler.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
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

boost::system::error_code open(implementation_type&,
const protocol_type&, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

boost::system::error_code assign(implementation_type&, const protocol_type&,
const native_handle_type&, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

bool is_open(const implementation_type&) const
{
return false;
}

boost::system::error_code close(implementation_type&,
boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

native_handle_type release(implementation_type&,
boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

native_handle_type native_handle(implementation_type&)
{
return 0;
}

boost::system::error_code cancel(implementation_type&,
boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

bool at_mark(const implementation_type&,
boost::system::error_code& ec) const
{
ec = boost::asio::error::operation_not_supported;
return false;
}

std::size_t available(const implementation_type&,
boost::system::error_code& ec) const
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

boost::system::error_code listen(implementation_type&,
int, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

template <typename IO_Control_Command>
boost::system::error_code io_control(implementation_type&,
IO_Control_Command&, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

bool non_blocking(const implementation_type&) const
{
return false;
}

boost::system::error_code non_blocking(implementation_type&,
bool, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

bool native_non_blocking(const implementation_type&) const
{
return false;
}

boost::system::error_code native_non_blocking(implementation_type&,
bool, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

boost::system::error_code shutdown(implementation_type&,
socket_base::shutdown_type, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

boost::system::error_code bind(implementation_type&,
const endpoint_type&, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

template <typename Option>
boost::system::error_code set_option(implementation_type&,
const Option&, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

template <typename Option>
boost::system::error_code get_option(const implementation_type&,
Option&, boost::system::error_code& ec) const
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

endpoint_type local_endpoint(const implementation_type&,
boost::system::error_code& ec) const
{
ec = boost::asio::error::operation_not_supported;
return endpoint_type();
}

endpoint_type remote_endpoint(const implementation_type&,
boost::system::error_code& ec) const
{
ec = boost::asio::error::operation_not_supported;
return endpoint_type();
}

template <typename ConstBufferSequence>
std::size_t send(implementation_type&, const ConstBufferSequence&,
socket_base::message_flags, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

std::size_t send(implementation_type&, const null_buffers&,
socket_base::message_flags, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_send(implementation_type&, const ConstBufferSequence&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
boost::asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename Handler, typename IoExecutor>
void async_send(implementation_type&, const null_buffers&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
boost::asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename MutableBufferSequence>
std::size_t receive(implementation_type&, const MutableBufferSequence&,
socket_base::message_flags, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

std::size_t receive(implementation_type&, const null_buffers&,
socket_base::message_flags, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_receive(implementation_type&, const MutableBufferSequence&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
boost::asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename Handler, typename IoExecutor>
void async_receive(implementation_type&, const null_buffers&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
boost::asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename MutableBufferSequence>
std::size_t receive_with_flags(implementation_type&,
const MutableBufferSequence&, socket_base::message_flags,
socket_base::message_flags&, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

std::size_t receive_with_flags(implementation_type&,
const null_buffers&, socket_base::message_flags,
socket_base::message_flags&, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_receive_with_flags(implementation_type&,
const MutableBufferSequence&, socket_base::message_flags,
socket_base::message_flags&, Handler& handler, const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
boost::asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename Handler, typename IoExecutor>
void async_receive_with_flags(implementation_type&, const null_buffers&,
socket_base::message_flags, socket_base::message_flags&,
Handler& handler, const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
boost::asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename ConstBufferSequence>
std::size_t send_to(implementation_type&, const ConstBufferSequence&,
const endpoint_type&, socket_base::message_flags,
boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

std::size_t send_to(implementation_type&, const null_buffers&,
const endpoint_type&, socket_base::message_flags,
boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_send_to(implementation_type&, const ConstBufferSequence&,
const endpoint_type&, socket_base::message_flags,
Handler& handler)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
boost::asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename Handler, typename IoExecutor>
void async_send_to(implementation_type&, const null_buffers&,
const endpoint_type&, socket_base::message_flags,
Handler& handler, const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
boost::asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename MutableBufferSequence>
std::size_t receive_from(implementation_type&, const MutableBufferSequence&,
endpoint_type&, socket_base::message_flags,
boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

std::size_t receive_from(implementation_type&, const null_buffers&,
endpoint_type&, socket_base::message_flags,
boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_receive_from(implementation_type&, const MutableBufferSequence&,
endpoint_type&, socket_base::message_flags, Handler& handler,
const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
boost::asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename Handler, typename IoExecutor>
void async_receive_from(implementation_type&, const null_buffers&,
endpoint_type&, socket_base::message_flags, Handler& handler,
const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
boost::asio::post(io_ex, detail::bind_handler(
handler, ec, bytes_transferred));
}

template <typename Socket>
boost::system::error_code accept(implementation_type&,
Socket&, endpoint_type*, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

template <typename Socket, typename Handler, typename IoExecutor>
void async_accept(implementation_type&, Socket&, endpoint_type*,
Handler& handler, const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
boost::asio::post(io_ex, detail::bind_handler(handler, ec));
}

boost::system::error_code connect(implementation_type&,
const endpoint_type&, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

template <typename Handler, typename IoExecutor>
void async_connect(implementation_type&, const endpoint_type&,
Handler& handler, const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
boost::asio::post(io_ex, detail::bind_handler(handler, ec));
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
