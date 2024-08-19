
#ifndef ASIO_SSL_STREAM_HPP
#define ASIO_SSL_STREAM_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/async_result.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/non_const_lvalue.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/ssl/context.hpp"
#include "asio/ssl/detail/buffered_handshake_op.hpp"
#include "asio/ssl/detail/handshake_op.hpp"
#include "asio/ssl/detail/io.hpp"
#include "asio/ssl/detail/read_op.hpp"
#include "asio/ssl/detail/shutdown_op.hpp"
#include "asio/ssl/detail/stream_core.hpp"
#include "asio/ssl/detail/write_op.hpp"
#include "asio/ssl/stream_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {


template <typename Stream>
class stream :
public stream_base,
private noncopyable
{
public:
typedef SSL* native_handle_type;

struct impl_struct
{
SSL* ssl;
};

typedef typename remove_reference<Stream>::type next_layer_type;

typedef typename next_layer_type::lowest_layer_type lowest_layer_type;

typedef typename lowest_layer_type::executor_type executor_type;

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

template <typename Arg>
stream(Arg&& arg, context& ctx)
: next_layer_(ASIO_MOVE_CAST(Arg)(arg)),
core_(ctx.native_handle(), next_layer_.lowest_layer().get_executor())
{
}
#else 
template <typename Arg>
stream(Arg& arg, context& ctx)
: next_layer_(arg),
core_(ctx.native_handle(), next_layer_.lowest_layer().get_executor())
{
}
#endif 

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

stream(stream&& other)
: next_layer_(ASIO_MOVE_CAST(Stream)(other.next_layer_)),
core_(ASIO_MOVE_CAST(detail::stream_core)(other.core_))
{
}


stream& operator=(stream&& other)
{
if (this != &other)
{
next_layer_ = ASIO_MOVE_CAST(Stream)(other.next_layer_);
core_ = ASIO_MOVE_CAST(detail::stream_core)(other.core_);
}
return *this;
}
#endif 


~stream()
{
}


executor_type get_executor() ASIO_NOEXCEPT
{
return next_layer_.lowest_layer().get_executor();
}


native_handle_type native_handle()
{
return core_.engine_.native_handle();
}


const next_layer_type& next_layer() const
{
return next_layer_;
}


next_layer_type& next_layer()
{
return next_layer_;
}


lowest_layer_type& lowest_layer()
{
return next_layer_.lowest_layer();
}


const lowest_layer_type& lowest_layer() const
{
return next_layer_.lowest_layer();
}


void set_verify_mode(verify_mode v)
{
asio::error_code ec;
set_verify_mode(v, ec);
asio::detail::throw_error(ec, "set_verify_mode");
}


ASIO_SYNC_OP_VOID set_verify_mode(
verify_mode v, asio::error_code& ec)
{
core_.engine_.set_verify_mode(v, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


void set_verify_depth(int depth)
{
asio::error_code ec;
set_verify_depth(depth, ec);
asio::detail::throw_error(ec, "set_verify_depth");
}


ASIO_SYNC_OP_VOID set_verify_depth(
int depth, asio::error_code& ec)
{
core_.engine_.set_verify_depth(depth, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename VerifyCallback>
void set_verify_callback(VerifyCallback callback)
{
asio::error_code ec;
this->set_verify_callback(callback, ec);
asio::detail::throw_error(ec, "set_verify_callback");
}


template <typename VerifyCallback>
ASIO_SYNC_OP_VOID set_verify_callback(VerifyCallback callback,
asio::error_code& ec)
{
core_.engine_.set_verify_callback(
new detail::verify_callback<VerifyCallback>(callback), ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


void handshake(handshake_type type)
{
asio::error_code ec;
handshake(type, ec);
asio::detail::throw_error(ec, "handshake");
}


ASIO_SYNC_OP_VOID handshake(handshake_type type,
asio::error_code& ec)
{
detail::io(next_layer_, core_, detail::handshake_op(type), ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename ConstBufferSequence>
void handshake(handshake_type type, const ConstBufferSequence& buffers)
{
asio::error_code ec;
handshake(type, buffers, ec);
asio::detail::throw_error(ec, "handshake");
}


template <typename ConstBufferSequence>
ASIO_SYNC_OP_VOID handshake(handshake_type type,
const ConstBufferSequence& buffers, asio::error_code& ec)
{
detail::io(next_layer_, core_,
detail::buffered_handshake_op<ConstBufferSequence>(type, buffers), ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code))
HandshakeHandler
ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(HandshakeHandler,
void (asio::error_code))
async_handshake(handshake_type type,
ASIO_MOVE_ARG(HandshakeHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<HandshakeHandler,
void (asio::error_code)>(
initiate_async_handshake(this), handler, type);
}


template <typename ConstBufferSequence,
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
std::size_t)) BufferedHandshakeHandler
ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(BufferedHandshakeHandler,
void (asio::error_code, std::size_t))
async_handshake(handshake_type type, const ConstBufferSequence& buffers,
ASIO_MOVE_ARG(BufferedHandshakeHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<BufferedHandshakeHandler,
void (asio::error_code, std::size_t)>(
initiate_async_buffered_handshake(this), handler, type, buffers);
}


void shutdown()
{
asio::error_code ec;
shutdown(ec);
asio::detail::throw_error(ec, "shutdown");
}


ASIO_SYNC_OP_VOID shutdown(asio::error_code& ec)
{
detail::io(next_layer_, core_, detail::shutdown_op(), ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code))
ShutdownHandler
ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(ShutdownHandler,
void (asio::error_code))
async_shutdown(
ASIO_MOVE_ARG(ShutdownHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<ShutdownHandler,
void (asio::error_code)>(
initiate_async_shutdown(this), handler);
}


template <typename ConstBufferSequence>
std::size_t write_some(const ConstBufferSequence& buffers)
{
asio::error_code ec;
std::size_t n = write_some(buffers, ec);
asio::detail::throw_error(ec, "write_some");
return n;
}


template <typename ConstBufferSequence>
std::size_t write_some(const ConstBufferSequence& buffers,
asio::error_code& ec)
{
return detail::io(next_layer_, core_,
detail::write_op<ConstBufferSequence>(buffers), ec);
}


template <typename ConstBufferSequence,
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
std::size_t)) WriteHandler
ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
void (asio::error_code, std::size_t))
async_write_some(const ConstBufferSequence& buffers,
ASIO_MOVE_ARG(WriteHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WriteHandler,
void (asio::error_code, std::size_t)>(
initiate_async_write_some(this), handler, buffers);
}


template <typename MutableBufferSequence>
std::size_t read_some(const MutableBufferSequence& buffers)
{
asio::error_code ec;
std::size_t n = read_some(buffers, ec);
asio::detail::throw_error(ec, "read_some");
return n;
}


template <typename MutableBufferSequence>
std::size_t read_some(const MutableBufferSequence& buffers,
asio::error_code& ec)
{
return detail::io(next_layer_, core_,
detail::read_op<MutableBufferSequence>(buffers), ec);
}


template <typename MutableBufferSequence,
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
std::size_t)) ReadHandler
ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (asio::error_code, std::size_t))
async_read_some(const MutableBufferSequence& buffers,
ASIO_MOVE_ARG(ReadHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<ReadHandler,
void (asio::error_code, std::size_t)>(
initiate_async_read_some(this), handler, buffers);
}

private:
class initiate_async_handshake
{
public:
typedef typename stream::executor_type executor_type;

explicit initiate_async_handshake(stream* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename HandshakeHandler>
void operator()(ASIO_MOVE_ARG(HandshakeHandler) handler,
handshake_type type) const
{
ASIO_HANDSHAKE_HANDLER_CHECK(HandshakeHandler, handler) type_check;

asio::detail::non_const_lvalue<HandshakeHandler> handler2(handler);
detail::async_io(self_->next_layer_, self_->core_,
detail::handshake_op(type), handler2.value);
}

private:
stream* self_;
};

class initiate_async_buffered_handshake
{
public:
typedef typename stream::executor_type executor_type;

explicit initiate_async_buffered_handshake(stream* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename BufferedHandshakeHandler, typename ConstBufferSequence>
void operator()(ASIO_MOVE_ARG(BufferedHandshakeHandler) handler,
handshake_type type, const ConstBufferSequence& buffers) const
{
ASIO_BUFFERED_HANDSHAKE_HANDLER_CHECK(
BufferedHandshakeHandler, handler) type_check;

asio::detail::non_const_lvalue<
BufferedHandshakeHandler> handler2(handler);
detail::async_io(self_->next_layer_, self_->core_,
detail::buffered_handshake_op<ConstBufferSequence>(type, buffers),
handler2.value);
}

private:
stream* self_;
};

class initiate_async_shutdown
{
public:
typedef typename stream::executor_type executor_type;

explicit initiate_async_shutdown(stream* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename ShutdownHandler>
void operator()(ASIO_MOVE_ARG(ShutdownHandler) handler) const
{
ASIO_HANDSHAKE_HANDLER_CHECK(ShutdownHandler, handler) type_check;

asio::detail::non_const_lvalue<ShutdownHandler> handler2(handler);
detail::async_io(self_->next_layer_, self_->core_,
detail::shutdown_op(), handler2.value);
}

private:
stream* self_;
};

class initiate_async_write_some
{
public:
typedef typename stream::executor_type executor_type;

explicit initiate_async_write_some(stream* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename WriteHandler, typename ConstBufferSequence>
void operator()(ASIO_MOVE_ARG(WriteHandler) handler,
const ConstBufferSequence& buffers) const
{
ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

asio::detail::non_const_lvalue<WriteHandler> handler2(handler);
detail::async_io(self_->next_layer_, self_->core_,
detail::write_op<ConstBufferSequence>(buffers), handler2.value);
}

private:
stream* self_;
};

class initiate_async_read_some
{
public:
typedef typename stream::executor_type executor_type;

explicit initiate_async_read_some(stream* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename ReadHandler, typename MutableBufferSequence>
void operator()(ASIO_MOVE_ARG(ReadHandler) handler,
const MutableBufferSequence& buffers) const
{
ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

asio::detail::non_const_lvalue<ReadHandler> handler2(handler);
detail::async_io(self_->next_layer_, self_->core_,
detail::read_op<MutableBufferSequence>(buffers), handler2.value);
}

private:
stream* self_;
};

Stream next_layer_;
detail::stream_core core_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
