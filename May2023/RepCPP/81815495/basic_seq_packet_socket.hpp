
#ifndef ASIO_BASIC_SEQ_PACKET_SOCKET_HPP
#define ASIO_BASIC_SEQ_PACKET_SOCKET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <cstddef>
#include "asio/basic_socket.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

#if !defined(ASIO_BASIC_SEQ_PACKET_SOCKET_FWD_DECL)
#define ASIO_BASIC_SEQ_PACKET_SOCKET_FWD_DECL

template <typename Protocol, typename Executor = any_io_executor>
class basic_seq_packet_socket;

#endif 


template <typename Protocol, typename Executor>
class basic_seq_packet_socket
: public basic_socket<Protocol, Executor>
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_seq_packet_socket<Protocol, Executor1> other;
};

#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined native_handle_type;
#else
typedef typename basic_socket<Protocol,
Executor>::native_handle_type native_handle_type;
#endif

typedef Protocol protocol_type;

typedef typename Protocol::endpoint endpoint_type;


explicit basic_seq_packet_socket(const executor_type& ex)
: basic_socket<Protocol, Executor>(ex)
{
}


template <typename ExecutionContext>
explicit basic_seq_packet_socket(ExecutionContext& context,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: basic_socket<Protocol, Executor>(context)
{
}


basic_seq_packet_socket(const executor_type& ex,
const protocol_type& protocol)
: basic_socket<Protocol, Executor>(ex, protocol)
{
}


template <typename ExecutionContext>
basic_seq_packet_socket(ExecutionContext& context,
const protocol_type& protocol,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value,
defaulted_constraint
>::type = defaulted_constraint())
: basic_socket<Protocol, Executor>(context, protocol)
{
}


basic_seq_packet_socket(const executor_type& ex,
const endpoint_type& endpoint)
: basic_socket<Protocol, Executor>(ex, endpoint)
{
}


template <typename ExecutionContext>
basic_seq_packet_socket(ExecutionContext& context,
const endpoint_type& endpoint,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: basic_socket<Protocol, Executor>(context, endpoint)
{
}


basic_seq_packet_socket(const executor_type& ex,
const protocol_type& protocol, const native_handle_type& native_socket)
: basic_socket<Protocol, Executor>(ex, protocol, native_socket)
{
}


template <typename ExecutionContext>
basic_seq_packet_socket(ExecutionContext& context,
const protocol_type& protocol, const native_handle_type& native_socket,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: basic_socket<Protocol, Executor>(context, protocol, native_socket)
{
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

basic_seq_packet_socket(basic_seq_packet_socket&& other) ASIO_NOEXCEPT
: basic_socket<Protocol, Executor>(std::move(other))
{
}


basic_seq_packet_socket& operator=(basic_seq_packet_socket&& other)
{
basic_socket<Protocol, Executor>::operator=(std::move(other));
return *this;
}


template <typename Protocol1, typename Executor1>
basic_seq_packet_socket(basic_seq_packet_socket<Protocol1, Executor1>&& other,
typename constraint<
is_convertible<Protocol1, Protocol>::value
&& is_convertible<Executor1, Executor>::value
>::type = 0)
: basic_socket<Protocol, Executor>(std::move(other))
{
}


template <typename Protocol1, typename Executor1>
typename constraint<
is_convertible<Protocol1, Protocol>::value
&& is_convertible<Executor1, Executor>::value,
basic_seq_packet_socket&
>::type operator=(basic_seq_packet_socket<Protocol1, Executor1>&& other)
{
basic_socket<Protocol, Executor>::operator=(std::move(other));
return *this;
}
#endif 


~basic_seq_packet_socket()
{
}


template <typename ConstBufferSequence>
std::size_t send(const ConstBufferSequence& buffers,
socket_base::message_flags flags)
{
asio::error_code ec;
std::size_t s = this->impl_.get_service().send(
this->impl_.get_implementation(), buffers, flags, ec);
asio::detail::throw_error(ec, "send");
return s;
}


template <typename ConstBufferSequence>
std::size_t send(const ConstBufferSequence& buffers,
socket_base::message_flags flags, asio::error_code& ec)
{
return this->impl_.get_service().send(
this->impl_.get_implementation(), buffers, flags, ec);
}


template <typename ConstBufferSequence,
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
std::size_t)) WriteHandler
ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
void (asio::error_code, std::size_t))
async_send(const ConstBufferSequence& buffers,
socket_base::message_flags flags,
ASIO_MOVE_ARG(WriteHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WriteHandler,
void (asio::error_code, std::size_t)>(
initiate_async_send(this), handler, buffers, flags);
}


template <typename MutableBufferSequence>
std::size_t receive(const MutableBufferSequence& buffers,
socket_base::message_flags& out_flags)
{
asio::error_code ec;
std::size_t s = this->impl_.get_service().receive_with_flags(
this->impl_.get_implementation(), buffers, 0, out_flags, ec);
asio::detail::throw_error(ec, "receive");
return s;
}


template <typename MutableBufferSequence>
std::size_t receive(const MutableBufferSequence& buffers,
socket_base::message_flags in_flags,
socket_base::message_flags& out_flags)
{
asio::error_code ec;
std::size_t s = this->impl_.get_service().receive_with_flags(
this->impl_.get_implementation(), buffers, in_flags, out_flags, ec);
asio::detail::throw_error(ec, "receive");
return s;
}


template <typename MutableBufferSequence>
std::size_t receive(const MutableBufferSequence& buffers,
socket_base::message_flags in_flags,
socket_base::message_flags& out_flags, asio::error_code& ec)
{
return this->impl_.get_service().receive_with_flags(
this->impl_.get_implementation(), buffers, in_flags, out_flags, ec);
}


template <typename MutableBufferSequence,
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
std::size_t)) ReadHandler
ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (asio::error_code, std::size_t))
async_receive(const MutableBufferSequence& buffers,
socket_base::message_flags& out_flags,
ASIO_MOVE_ARG(ReadHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<ReadHandler,
void (asio::error_code, std::size_t)>(
initiate_async_receive_with_flags(this), handler,
buffers, socket_base::message_flags(0), &out_flags);
}


template <typename MutableBufferSequence,
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
std::size_t)) ReadHandler
ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (asio::error_code, std::size_t))
async_receive(const MutableBufferSequence& buffers,
socket_base::message_flags in_flags,
socket_base::message_flags& out_flags,
ASIO_MOVE_ARG(ReadHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<ReadHandler,
void (asio::error_code, std::size_t)>(
initiate_async_receive_with_flags(this),
handler, buffers, in_flags, &out_flags);
}

private:
basic_seq_packet_socket(const basic_seq_packet_socket&) ASIO_DELETED;
basic_seq_packet_socket& operator=(
const basic_seq_packet_socket&) ASIO_DELETED;

class initiate_async_send
{
public:
typedef Executor executor_type;

explicit initiate_async_send(basic_seq_packet_socket* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename WriteHandler, typename ConstBufferSequence>
void operator()(ASIO_MOVE_ARG(WriteHandler) handler,
const ConstBufferSequence& buffers,
socket_base::message_flags flags) const
{
ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

detail::non_const_lvalue<WriteHandler> handler2(handler);
self_->impl_.get_service().async_send(
self_->impl_.get_implementation(), buffers, flags,
handler2.value, self_->impl_.get_executor());
}

private:
basic_seq_packet_socket* self_;
};

class initiate_async_receive_with_flags
{
public:
typedef Executor executor_type;

explicit initiate_async_receive_with_flags(basic_seq_packet_socket* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename ReadHandler, typename MutableBufferSequence>
void operator()(ASIO_MOVE_ARG(ReadHandler) handler,
const MutableBufferSequence& buffers,
socket_base::message_flags in_flags,
socket_base::message_flags* out_flags) const
{
ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

detail::non_const_lvalue<ReadHandler> handler2(handler);
self_->impl_.get_service().async_receive_with_flags(
self_->impl_.get_implementation(), buffers, in_flags,
*out_flags, handler2.value, self_->impl_.get_executor());
}

private:
basic_seq_packet_socket* self_;
};
};

} 

#include "asio/detail/pop_options.hpp"

#endif 
