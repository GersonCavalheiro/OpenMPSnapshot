
#ifndef BOOST_ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_HPP
#define BOOST_ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_HAS_IOCP)

#include <boost/asio/buffer.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/socket_base.hpp>
#include <boost/asio/detail/buffer_sequence_adapter.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/detail/reactive_null_buffers_op.hpp>
#include <boost/asio/detail/reactive_socket_accept_op.hpp>
#include <boost/asio/detail/reactive_socket_connect_op.hpp>
#include <boost/asio/detail/reactive_socket_recvfrom_op.hpp>
#include <boost/asio/detail/reactive_socket_sendto_op.hpp>
#include <boost/asio/detail/reactive_socket_service_base.hpp>
#include <boost/asio/detail/reactor.hpp>
#include <boost/asio/detail/reactor_op.hpp>
#include <boost/asio/detail/socket_holder.hpp>
#include <boost/asio/detail/socket_ops.hpp>
#include <boost/asio/detail/socket_types.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename Protocol>
class reactive_socket_service :
public execution_context_service_base<reactive_socket_service<Protocol> >,
public reactive_socket_service_base
{
public:
typedef Protocol protocol_type;

typedef typename Protocol::endpoint endpoint_type;

typedef socket_type native_handle_type;

struct implementation_type :
reactive_socket_service_base::base_implementation_type
{
implementation_type()
: protocol_(endpoint_type().protocol())
{
}

protocol_type protocol_;
};

reactive_socket_service(execution_context& context)
: execution_context_service_base<
reactive_socket_service<Protocol> >(context),
reactive_socket_service_base(context)
{
}

void shutdown()
{
this->base_shutdown();
}

void move_construct(implementation_type& impl,
implementation_type& other_impl) BOOST_ASIO_NOEXCEPT
{
this->base_move_construct(impl, other_impl);

impl.protocol_ = other_impl.protocol_;
other_impl.protocol_ = endpoint_type().protocol();
}

void move_assign(implementation_type& impl,
reactive_socket_service_base& other_service,
implementation_type& other_impl)
{
this->base_move_assign(impl, other_service, other_impl);

impl.protocol_ = other_impl.protocol_;
other_impl.protocol_ = endpoint_type().protocol();
}

template <typename Protocol1>
void converting_move_construct(implementation_type& impl,
reactive_socket_service<Protocol1>&,
typename reactive_socket_service<
Protocol1>::implementation_type& other_impl)
{
this->base_move_construct(impl, other_impl);

impl.protocol_ = protocol_type(other_impl.protocol_);
other_impl.protocol_ = typename Protocol1::endpoint().protocol();
}

boost::system::error_code open(implementation_type& impl,
const protocol_type& protocol, boost::system::error_code& ec)
{
if (!do_open(impl, protocol.family(),
protocol.type(), protocol.protocol(), ec))
impl.protocol_ = protocol;
return ec;
}

boost::system::error_code assign(implementation_type& impl,
const protocol_type& protocol, const native_handle_type& native_socket,
boost::system::error_code& ec)
{
if (!do_assign(impl, protocol.type(), native_socket, ec))
impl.protocol_ = protocol;
return ec;
}

native_handle_type native_handle(implementation_type& impl)
{
return impl.socket_;
}

boost::system::error_code bind(implementation_type& impl,
const endpoint_type& endpoint, boost::system::error_code& ec)
{
socket_ops::bind(impl.socket_, endpoint.data(), endpoint.size(), ec);
return ec;
}

template <typename Option>
boost::system::error_code set_option(implementation_type& impl,
const Option& option, boost::system::error_code& ec)
{
socket_ops::setsockopt(impl.socket_, impl.state_,
option.level(impl.protocol_), option.name(impl.protocol_),
option.data(impl.protocol_), option.size(impl.protocol_), ec);
return ec;
}

template <typename Option>
boost::system::error_code get_option(const implementation_type& impl,
Option& option, boost::system::error_code& ec) const
{
std::size_t size = option.size(impl.protocol_);
socket_ops::getsockopt(impl.socket_, impl.state_,
option.level(impl.protocol_), option.name(impl.protocol_),
option.data(impl.protocol_), &size, ec);
if (!ec)
option.resize(impl.protocol_, size);
return ec;
}

endpoint_type local_endpoint(const implementation_type& impl,
boost::system::error_code& ec) const
{
endpoint_type endpoint;
std::size_t addr_len = endpoint.capacity();
if (socket_ops::getsockname(impl.socket_, endpoint.data(), &addr_len, ec))
return endpoint_type();
endpoint.resize(addr_len);
return endpoint;
}

endpoint_type remote_endpoint(const implementation_type& impl,
boost::system::error_code& ec) const
{
endpoint_type endpoint;
std::size_t addr_len = endpoint.capacity();
if (socket_ops::getpeername(impl.socket_,
endpoint.data(), &addr_len, false, ec))
return endpoint_type();
endpoint.resize(addr_len);
return endpoint;
}

boost::system::error_code shutdown(base_implementation_type& impl,
socket_base::shutdown_type what, boost::system::error_code& ec)
{
socket_ops::shutdown(impl.socket_, what, ec);
return ec;
}

template <typename ConstBufferSequence>
size_t send_to(implementation_type& impl, const ConstBufferSequence& buffers,
const endpoint_type& destination, socket_base::message_flags flags,
boost::system::error_code& ec)
{
typedef buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence> bufs_type;

if (bufs_type::is_single_buffer)
{
return socket_ops::sync_sendto1(impl.socket_, impl.state_,
bufs_type::first(buffers).data(),
bufs_type::first(buffers).size(), flags,
destination.data(), destination.size(), ec);
}
else
{
bufs_type bufs(buffers);
return socket_ops::sync_sendto(impl.socket_, impl.state_,
bufs.buffers(), bufs.count(), flags,
destination.data(), destination.size(), ec);
}
}

size_t send_to(implementation_type& impl, const null_buffers&,
const endpoint_type&, socket_base::message_flags,
boost::system::error_code& ec)
{
socket_ops::poll_write(impl.socket_, impl.state_, -1, ec);

return 0;
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_send_to(implementation_type& impl,
const ConstBufferSequence& buffers,
const endpoint_type& destination, socket_base::message_flags flags,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_socket_sendto_op<ConstBufferSequence,
endpoint_type, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, impl.socket_,
buffers, destination, flags, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_send_to"));

start_op(impl, reactor::write_op, p.p, is_continuation, true, false);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_send_to(implementation_type& impl, const null_buffers&,
const endpoint_type&, socket_base::message_flags,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_send_to(null_buffers)"));

start_op(impl, reactor::write_op, p.p, is_continuation, false, false);
p.v = p.p = 0;
}

template <typename MutableBufferSequence>
size_t receive_from(implementation_type& impl,
const MutableBufferSequence& buffers,
endpoint_type& sender_endpoint, socket_base::message_flags flags,
boost::system::error_code& ec)
{
typedef buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence> bufs_type;

std::size_t addr_len = sender_endpoint.capacity();
std::size_t bytes_recvd;
if (bufs_type::is_single_buffer)
{
bytes_recvd = socket_ops::sync_recvfrom1(impl.socket_,
impl.state_, bufs_type::first(buffers).data(),
bufs_type::first(buffers).size(), flags,
sender_endpoint.data(), &addr_len, ec);
}
else
{
bufs_type bufs(buffers);
bytes_recvd = socket_ops::sync_recvfrom(
impl.socket_, impl.state_, bufs.buffers(), bufs.count(),
flags, sender_endpoint.data(), &addr_len, ec);
}

if (!ec)
sender_endpoint.resize(addr_len);

return bytes_recvd;
}

size_t receive_from(implementation_type& impl, const null_buffers&,
endpoint_type& sender_endpoint, socket_base::message_flags,
boost::system::error_code& ec)
{
socket_ops::poll_read(impl.socket_, impl.state_, -1, ec);

sender_endpoint = endpoint_type();

return 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_receive_from(implementation_type& impl,
const MutableBufferSequence& buffers, endpoint_type& sender_endpoint,
socket_base::message_flags flags, Handler& handler,
const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_socket_recvfrom_op<MutableBufferSequence,
endpoint_type, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
int protocol = impl.protocol_.type();
p.p = new (p.v) op(success_ec_, impl.socket_, protocol,
buffers, sender_endpoint, flags, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_receive_from"));

start_op(impl,
(flags & socket_base::message_out_of_band)
? reactor::except_op : reactor::read_op,
p.p, is_continuation, true, false);
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_receive_from(implementation_type& impl, const null_buffers&,
endpoint_type& sender_endpoint, socket_base::message_flags flags,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_receive_from(null_buffers)"));

sender_endpoint = endpoint_type();

start_op(impl,
(flags & socket_base::message_out_of_band)
? reactor::except_op : reactor::read_op,
p.p, is_continuation, false, false);
p.v = p.p = 0;
}

template <typename Socket>
boost::system::error_code accept(implementation_type& impl,
Socket& peer, endpoint_type* peer_endpoint, boost::system::error_code& ec)
{
if (peer.is_open())
{
ec = boost::asio::error::already_open;
return ec;
}

std::size_t addr_len = peer_endpoint ? peer_endpoint->capacity() : 0;
socket_holder new_socket(socket_ops::sync_accept(impl.socket_,
impl.state_, peer_endpoint ? peer_endpoint->data() : 0,
peer_endpoint ? &addr_len : 0, ec));

if (new_socket.get() != invalid_socket)
{
if (peer_endpoint)
peer_endpoint->resize(addr_len);
peer.assign(impl.protocol_, new_socket.get(), ec);
if (!ec)
new_socket.release();
}

return ec;
}

template <typename Socket, typename Handler, typename IoExecutor>
void async_accept(implementation_type& impl, Socket& peer,
endpoint_type* peer_endpoint, Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_socket_accept_op<Socket, Protocol, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, impl.socket_, impl.state_,
peer, impl.protocol_, peer_endpoint, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_accept"));

start_accept_op(impl, p.p, is_continuation, peer.is_open());
p.v = p.p = 0;
}

#if defined(BOOST_ASIO_HAS_MOVE)
template <typename PeerIoExecutor, typename Handler, typename IoExecutor>
void async_move_accept(implementation_type& impl,
const PeerIoExecutor& peer_io_ex, endpoint_type* peer_endpoint,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_socket_move_accept_op<Protocol,
PeerIoExecutor, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, peer_io_ex, impl.socket_,
impl.state_, impl.protocol_, peer_endpoint, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_accept"));

start_accept_op(impl, p.p, is_continuation, false);
p.v = p.p = 0;
}
#endif 

boost::system::error_code connect(implementation_type& impl,
const endpoint_type& peer_endpoint, boost::system::error_code& ec)
{
socket_ops::sync_connect(impl.socket_,
peer_endpoint.data(), peer_endpoint.size(), ec);
return ec;
}

template <typename Handler, typename IoExecutor>
void async_connect(implementation_type& impl,
const endpoint_type& peer_endpoint,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef reactive_socket_connect_op<Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, impl.socket_, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "socket",
&impl, impl.socket_, "async_connect"));

start_connect_op(impl, p.p, is_continuation,
peer_endpoint.data(), peer_endpoint.size());
p.v = p.p = 0;
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
