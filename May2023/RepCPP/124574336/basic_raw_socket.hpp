
#ifndef BOOST_ASIO_BASIC_RAW_SOCKET_HPP
#define BOOST_ASIO_BASIC_RAW_SOCKET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <cstddef>
#include <boost/asio/basic_socket.hpp>
#include <boost/asio/detail/handler_type_requirements.hpp>
#include <boost/asio/detail/non_const_lvalue.hpp>
#include <boost/asio/detail/throw_error.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/error.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if !defined(BOOST_ASIO_BASIC_RAW_SOCKET_FWD_DECL)
#define BOOST_ASIO_BASIC_RAW_SOCKET_FWD_DECL

template <typename Protocol, typename Executor = any_io_executor>
class basic_raw_socket;

#endif 


template <typename Protocol, typename Executor>
class basic_raw_socket
: public basic_socket<Protocol, Executor>
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_raw_socket<Protocol, Executor1> other;
};

#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined native_handle_type;
#else
typedef typename basic_socket<Protocol,
Executor>::native_handle_type native_handle_type;
#endif

typedef Protocol protocol_type;

typedef typename Protocol::endpoint endpoint_type;


explicit basic_raw_socket(const executor_type& ex)
: basic_socket<Protocol, Executor>(ex)
{
}


template <typename ExecutionContext>
explicit basic_raw_socket(ExecutionContext& context,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: basic_socket<Protocol, Executor>(context)
{
}


basic_raw_socket(const executor_type& ex, const protocol_type& protocol)
: basic_socket<Protocol, Executor>(ex, protocol)
{
}


template <typename ExecutionContext>
basic_raw_socket(ExecutionContext& context, const protocol_type& protocol,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: basic_socket<Protocol, Executor>(context, protocol)
{
}


basic_raw_socket(const executor_type& ex, const endpoint_type& endpoint)
: basic_socket<Protocol, Executor>(ex, endpoint)
{
}


template <typename ExecutionContext>
basic_raw_socket(ExecutionContext& context, const endpoint_type& endpoint,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: basic_socket<Protocol, Executor>(context, endpoint)
{
}


basic_raw_socket(const executor_type& ex,
const protocol_type& protocol, const native_handle_type& native_socket)
: basic_socket<Protocol, Executor>(ex, protocol, native_socket)
{
}


template <typename ExecutionContext>
basic_raw_socket(ExecutionContext& context,
const protocol_type& protocol, const native_handle_type& native_socket,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: basic_socket<Protocol, Executor>(context, protocol, native_socket)
{
}

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

basic_raw_socket(basic_raw_socket&& other) BOOST_ASIO_NOEXCEPT
: basic_socket<Protocol, Executor>(std::move(other))
{
}


basic_raw_socket& operator=(basic_raw_socket&& other)
{
basic_socket<Protocol, Executor>::operator=(std::move(other));
return *this;
}


template <typename Protocol1, typename Executor1>
basic_raw_socket(basic_raw_socket<Protocol1, Executor1>&& other,
typename enable_if<
is_convertible<Protocol1, Protocol>::value
&& is_convertible<Executor1, Executor>::value
>::type* = 0)
: basic_socket<Protocol, Executor>(std::move(other))
{
}


template <typename Protocol1, typename Executor1>
typename enable_if<
is_convertible<Protocol1, Protocol>::value
&& is_convertible<Executor1, Executor>::value,
basic_raw_socket&
>::type operator=(basic_raw_socket<Protocol1, Executor1>&& other)
{
basic_socket<Protocol, Executor>::operator=(std::move(other));
return *this;
}
#endif 


~basic_raw_socket()
{
}


template <typename ConstBufferSequence>
std::size_t send(const ConstBufferSequence& buffers)
{
boost::system::error_code ec;
std::size_t s = this->impl_.get_service().send(
this->impl_.get_implementation(), buffers, 0, ec);
boost::asio::detail::throw_error(ec, "send");
return s;
}


template <typename ConstBufferSequence>
std::size_t send(const ConstBufferSequence& buffers,
socket_base::message_flags flags)
{
boost::system::error_code ec;
std::size_t s = this->impl_.get_service().send(
this->impl_.get_implementation(), buffers, flags, ec);
boost::asio::detail::throw_error(ec, "send");
return s;
}


template <typename ConstBufferSequence>
std::size_t send(const ConstBufferSequence& buffers,
socket_base::message_flags flags, boost::system::error_code& ec)
{
return this->impl_.get_service().send(
this->impl_.get_implementation(), buffers, flags, ec);
}


template <typename ConstBufferSequence,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) WriteHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
void (boost::system::error_code, std::size_t))
async_send(const ConstBufferSequence& buffers,
BOOST_ASIO_MOVE_ARG(WriteHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WriteHandler,
void (boost::system::error_code, std::size_t)>(
initiate_async_send(this), handler,
buffers, socket_base::message_flags(0));
}


template <typename ConstBufferSequence,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) WriteHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
void (boost::system::error_code, std::size_t))
async_send(const ConstBufferSequence& buffers,
socket_base::message_flags flags,
BOOST_ASIO_MOVE_ARG(WriteHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WriteHandler,
void (boost::system::error_code, std::size_t)>(
initiate_async_send(this), handler, buffers, flags);
}


template <typename ConstBufferSequence>
std::size_t send_to(const ConstBufferSequence& buffers,
const endpoint_type& destination)
{
boost::system::error_code ec;
std::size_t s = this->impl_.get_service().send_to(
this->impl_.get_implementation(), buffers, destination, 0, ec);
boost::asio::detail::throw_error(ec, "send_to");
return s;
}


template <typename ConstBufferSequence>
std::size_t send_to(const ConstBufferSequence& buffers,
const endpoint_type& destination, socket_base::message_flags flags)
{
boost::system::error_code ec;
std::size_t s = this->impl_.get_service().send_to(
this->impl_.get_implementation(), buffers, destination, flags, ec);
boost::asio::detail::throw_error(ec, "send_to");
return s;
}


template <typename ConstBufferSequence>
std::size_t send_to(const ConstBufferSequence& buffers,
const endpoint_type& destination, socket_base::message_flags flags,
boost::system::error_code& ec)
{
return this->impl_.get_service().send_to(this->impl_.get_implementation(),
buffers, destination, flags, ec);
}


template <typename ConstBufferSequence,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) WriteHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
void (boost::system::error_code, std::size_t))
async_send_to(const ConstBufferSequence& buffers,
const endpoint_type& destination,
BOOST_ASIO_MOVE_ARG(WriteHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WriteHandler,
void (boost::system::error_code, std::size_t)>(
initiate_async_send_to(this), handler, buffers,
destination, socket_base::message_flags(0));
}


template <typename ConstBufferSequence,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) WriteHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
void (boost::system::error_code, std::size_t))
async_send_to(const ConstBufferSequence& buffers,
const endpoint_type& destination, socket_base::message_flags flags,
BOOST_ASIO_MOVE_ARG(WriteHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WriteHandler,
void (boost::system::error_code, std::size_t)>(
initiate_async_send_to(this), handler, buffers, destination, flags);
}


template <typename MutableBufferSequence>
std::size_t receive(const MutableBufferSequence& buffers)
{
boost::system::error_code ec;
std::size_t s = this->impl_.get_service().receive(
this->impl_.get_implementation(), buffers, 0, ec);
boost::asio::detail::throw_error(ec, "receive");
return s;
}


template <typename MutableBufferSequence>
std::size_t receive(const MutableBufferSequence& buffers,
socket_base::message_flags flags)
{
boost::system::error_code ec;
std::size_t s = this->impl_.get_service().receive(
this->impl_.get_implementation(), buffers, flags, ec);
boost::asio::detail::throw_error(ec, "receive");
return s;
}


template <typename MutableBufferSequence>
std::size_t receive(const MutableBufferSequence& buffers,
socket_base::message_flags flags, boost::system::error_code& ec)
{
return this->impl_.get_service().receive(
this->impl_.get_implementation(), buffers, flags, ec);
}


template <typename MutableBufferSequence,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_receive(const MutableBufferSequence& buffers,
BOOST_ASIO_MOVE_ARG(ReadHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<ReadHandler,
void (boost::system::error_code, std::size_t)>(
initiate_async_receive(this), handler,
buffers, socket_base::message_flags(0));
}


template <typename MutableBufferSequence,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_receive(const MutableBufferSequence& buffers,
socket_base::message_flags flags,
BOOST_ASIO_MOVE_ARG(ReadHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<ReadHandler,
void (boost::system::error_code, std::size_t)>(
initiate_async_receive(this), handler, buffers, flags);
}


template <typename MutableBufferSequence>
std::size_t receive_from(const MutableBufferSequence& buffers,
endpoint_type& sender_endpoint)
{
boost::system::error_code ec;
std::size_t s = this->impl_.get_service().receive_from(
this->impl_.get_implementation(), buffers, sender_endpoint, 0, ec);
boost::asio::detail::throw_error(ec, "receive_from");
return s;
}


template <typename MutableBufferSequence>
std::size_t receive_from(const MutableBufferSequence& buffers,
endpoint_type& sender_endpoint, socket_base::message_flags flags)
{
boost::system::error_code ec;
std::size_t s = this->impl_.get_service().receive_from(
this->impl_.get_implementation(), buffers, sender_endpoint, flags, ec);
boost::asio::detail::throw_error(ec, "receive_from");
return s;
}


template <typename MutableBufferSequence>
std::size_t receive_from(const MutableBufferSequence& buffers,
endpoint_type& sender_endpoint, socket_base::message_flags flags,
boost::system::error_code& ec)
{
return this->impl_.get_service().receive_from(
this->impl_.get_implementation(), buffers, sender_endpoint, flags, ec);
}


template <typename MutableBufferSequence,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_receive_from(const MutableBufferSequence& buffers,
endpoint_type& sender_endpoint,
BOOST_ASIO_MOVE_ARG(ReadHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<ReadHandler,
void (boost::system::error_code, std::size_t)>(
initiate_async_receive_from(this), handler, buffers,
&sender_endpoint, socket_base::message_flags(0));
}


template <typename MutableBufferSequence,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_receive_from(const MutableBufferSequence& buffers,
endpoint_type& sender_endpoint, socket_base::message_flags flags,
BOOST_ASIO_MOVE_ARG(ReadHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<ReadHandler,
void (boost::system::error_code, std::size_t)>(
initiate_async_receive_from(this), handler,
buffers, &sender_endpoint, flags);
}

private:
basic_raw_socket(const basic_raw_socket&) BOOST_ASIO_DELETED;
basic_raw_socket& operator=(const basic_raw_socket&) BOOST_ASIO_DELETED;

class initiate_async_send
{
public:
typedef Executor executor_type;

explicit initiate_async_send(basic_raw_socket* self)
: self_(self)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename WriteHandler, typename ConstBufferSequence>
void operator()(BOOST_ASIO_MOVE_ARG(WriteHandler) handler,
const ConstBufferSequence& buffers,
socket_base::message_flags flags) const
{
BOOST_ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

detail::non_const_lvalue<WriteHandler> handler2(handler);
self_->impl_.get_service().async_send(
self_->impl_.get_implementation(), buffers, flags,
handler2.value, self_->impl_.get_executor());
}

private:
basic_raw_socket* self_;
};

class initiate_async_send_to
{
public:
typedef Executor executor_type;

explicit initiate_async_send_to(basic_raw_socket* self)
: self_(self)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename WriteHandler, typename ConstBufferSequence>
void operator()(BOOST_ASIO_MOVE_ARG(WriteHandler) handler,
const ConstBufferSequence& buffers, const endpoint_type& destination,
socket_base::message_flags flags) const
{
BOOST_ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

detail::non_const_lvalue<WriteHandler> handler2(handler);
self_->impl_.get_service().async_send_to(
self_->impl_.get_implementation(), buffers, destination,
flags, handler2.value, self_->impl_.get_executor());
}

private:
basic_raw_socket* self_;
};

class initiate_async_receive
{
public:
typedef Executor executor_type;

explicit initiate_async_receive(basic_raw_socket* self)
: self_(self)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename ReadHandler, typename MutableBufferSequence>
void operator()(BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
const MutableBufferSequence& buffers,
socket_base::message_flags flags) const
{
BOOST_ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

detail::non_const_lvalue<ReadHandler> handler2(handler);
self_->impl_.get_service().async_receive(
self_->impl_.get_implementation(), buffers, flags,
handler2.value, self_->impl_.get_executor());
}

private:
basic_raw_socket* self_;
};

class initiate_async_receive_from
{
public:
typedef Executor executor_type;

explicit initiate_async_receive_from(basic_raw_socket* self)
: self_(self)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename ReadHandler, typename MutableBufferSequence>
void operator()(BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
const MutableBufferSequence& buffers, endpoint_type* sender_endpoint,
socket_base::message_flags flags) const
{
BOOST_ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

detail::non_const_lvalue<ReadHandler> handler2(handler);
self_->impl_.get_service().async_receive_from(
self_->impl_.get_implementation(), buffers, *sender_endpoint,
flags, handler2.value, self_->impl_.get_executor());
}

private:
basic_raw_socket* self_;
};
};

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
