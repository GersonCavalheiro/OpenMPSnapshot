
#ifndef BOOST_ASIO_DETAIL_WINRT_SSOCKET_SERVICE_BASE_HPP
#define BOOST_ASIO_DETAIL_WINRT_SSOCKET_SERVICE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_WINDOWS_RUNTIME)

#include <boost/asio/buffer.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/socket_base.hpp>
#include <boost/asio/detail/buffer_sequence_adapter.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/detail/winrt_async_manager.hpp>
#include <boost/asio/detail/winrt_socket_recv_op.hpp>
#include <boost/asio/detail/winrt_socket_send_op.hpp>

#if defined(BOOST_ASIO_HAS_IOCP)
# include <boost/asio/detail/win_iocp_io_context.hpp>
#else 
# include <boost/asio/detail/scheduler.hpp>
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
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

BOOST_ASIO_DECL winrt_ssocket_service_base(execution_context& context);

BOOST_ASIO_DECL void base_shutdown();

BOOST_ASIO_DECL void construct(base_implementation_type&);

BOOST_ASIO_DECL void base_move_construct(base_implementation_type& impl,
base_implementation_type& other_impl) BOOST_ASIO_NOEXCEPT;

BOOST_ASIO_DECL void base_move_assign(base_implementation_type& impl,
winrt_ssocket_service_base& other_service,
base_implementation_type& other_impl);

BOOST_ASIO_DECL void destroy(base_implementation_type& impl);

bool is_open(const base_implementation_type& impl) const
{
return impl.socket_ != nullptr;
}

BOOST_ASIO_DECL boost::system::error_code close(
base_implementation_type& impl, boost::system::error_code& ec);

BOOST_ASIO_DECL native_handle_type release(
base_implementation_type& impl, boost::system::error_code& ec);

native_handle_type native_handle(base_implementation_type& impl)
{
return impl.socket_;
}

boost::system::error_code cancel(base_implementation_type&,
boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

bool at_mark(const base_implementation_type&,
boost::system::error_code& ec) const
{
ec = boost::asio::error::operation_not_supported;
return false;
}

std::size_t available(const base_implementation_type&,
boost::system::error_code& ec) const
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

template <typename IO_Control_Command>
boost::system::error_code io_control(base_implementation_type&,
IO_Control_Command&, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

bool non_blocking(const base_implementation_type&) const
{
return false;
}

boost::system::error_code non_blocking(base_implementation_type&,
bool, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

bool native_non_blocking(const base_implementation_type&) const
{
return false;
}

boost::system::error_code native_non_blocking(base_implementation_type&,
bool, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

template <typename ConstBufferSequence>
std::size_t send(base_implementation_type& impl,
const ConstBufferSequence& buffers,
socket_base::message_flags flags, boost::system::error_code& ec)
{
return do_send(impl,
buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence>::first(buffers), flags, ec);
}

std::size_t send(base_implementation_type&, const null_buffers&,
socket_base::message_flags, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return 0;
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_send(base_implementation_type& impl,
const ConstBufferSequence& buffers, socket_base::message_flags flags,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef winrt_socket_send_op<ConstBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(buffers, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((scheduler_.context(),
*p.p, "socket", &impl, 0, "async_send"));

start_send_op(impl,
buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence>::first(buffers),
flags, p.p, is_continuation);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_send(base_implementation_type&, const null_buffers&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
boost::asio::post(io_ex,
detail::bind_handler(handler, ec, bytes_transferred));
}

template <typename MutableBufferSequence>
std::size_t receive(base_implementation_type& impl,
const MutableBufferSequence& buffers,
socket_base::message_flags flags, boost::system::error_code& ec)
{
return do_receive(impl,
buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence>::first(buffers), flags, ec);
}

std::size_t receive(base_implementation_type&, const null_buffers&,
socket_base::message_flags, boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
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

typedef winrt_socket_recv_op<MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(buffers, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((scheduler_.context(),
*p.p, "socket", &impl, 0, "async_receive"));

start_receive_op(impl,
buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence>::first(buffers),
flags, p.p, is_continuation);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_receive(base_implementation_type&, const null_buffers&,
socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
{
boost::system::error_code ec = boost::asio::error::operation_not_supported;
const std::size_t bytes_transferred = 0;
boost::asio::post(io_ex,
detail::bind_handler(handler, ec, bytes_transferred));
}

protected:
BOOST_ASIO_DECL std::size_t do_get_endpoint(
const base_implementation_type& impl, bool local,
void* addr, std::size_t addr_len, boost::system::error_code& ec) const;

BOOST_ASIO_DECL boost::system::error_code do_set_option(
base_implementation_type& impl,
int level, int optname, const void* optval,
std::size_t optlen, boost::system::error_code& ec);

BOOST_ASIO_DECL void do_get_option(
const base_implementation_type& impl,
int level, int optname, void* optval,
std::size_t* optlen, boost::system::error_code& ec) const;

BOOST_ASIO_DECL boost::system::error_code do_connect(
base_implementation_type& impl,
const void* addr, boost::system::error_code& ec);

BOOST_ASIO_DECL void start_connect_op(
base_implementation_type& impl, const void* addr,
winrt_async_op<void>* op, bool is_continuation);

BOOST_ASIO_DECL std::size_t do_send(
base_implementation_type& impl, const boost::asio::const_buffer& data,
socket_base::message_flags flags, boost::system::error_code& ec);

BOOST_ASIO_DECL void start_send_op(base_implementation_type& impl,
const boost::asio::const_buffer& data, socket_base::message_flags flags,
winrt_async_op<unsigned int>* op, bool is_continuation);

BOOST_ASIO_DECL std::size_t do_receive(
base_implementation_type& impl, const boost::asio::mutable_buffer& data,
socket_base::message_flags flags, boost::system::error_code& ec);

BOOST_ASIO_DECL void start_receive_op(base_implementation_type& impl,
const boost::asio::mutable_buffer& data, socket_base::message_flags flags,
winrt_async_op<Windows::Storage::Streams::IBuffer^>* op,
bool is_continuation);

#if defined(BOOST_ASIO_HAS_IOCP)
typedef class win_iocp_io_context scheduler_impl;
#else
typedef class scheduler scheduler_impl;
#endif
scheduler_impl& scheduler_;

winrt_async_manager& async_manager_;

boost::asio::detail::mutex mutex_;

base_implementation_type* impl_list_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/winrt_ssocket_service_base.ipp>
#endif 

#endif 

#endif 
